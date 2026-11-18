"""Single-stage deterministic training; all assets load strictly offline."""
import argparse
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import yaml

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .checkpoints import (identifiers, resources_for, ARCHITECTURE, RESOURCES, TRAINING_RECIPE, atomic_save, capture_rng,
                          load_checkpoint, load_initial_weights, loss_fingerprint, restore_rng, validate_training_checkpoint,
                          load_v6_initialization, load_v6_weights, validate_resume_topology)
from .data import AsyncClipDataset, balanced_sample_key, collate_clips
from .early_stopping import EarlyStopping, early_stopped_checkpoint
from .losses import TemporalReference, normalize_statistics, sum_statistics
from .metrics import validate
from .progress import TrainingProgress
from .models import build_model, parameter_groups
from .runtime import (GlobalSampleSampler, StageObjective, add_runtime_arguments,
                      append_json, apply_runtime_arguments, json_ready, load_config,
                      ensure_stage_can_extend, merge_config, portable_config, positive_int, query_indices,
                      resolve_manifest, seed_everything, stage_loader_settings, validate_config, write_json,
                      scheduled_counts, observation_counts, statistic_denominators, effective_update_objective)


def _checkpoint_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def distributed_context(device):
    requested = torch.device(device)
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank, local_rank = int(os.environ.get('RANK', '0')), int(os.environ.get('LOCAL_RANK', '0'))
    if requested.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but no visible CUDA device is available')
        requested = torch.device('cuda', local_rank if world_size > 1 else (requested.index or 0))
        torch.cuda.set_device(requested)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group('nccl' if requested.type == 'cuda' else 'gloo',
                                **({'device_id': requested} if requested.type == 'cuda' else {}))
    if dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
    return requested, rank, world_size


def _reference(model, checkpoint, best_a=None, fusion_config=None, *, ablation=None):
    if checkpoint is not None and checkpoint.get('temporal_reference') is not None:
        reference = TemporalReference(model.ir_motion, model.vis_motion, fusion_config=fusion_config,
                                      ablation=ablation)
        reference.load_state_dict(checkpoint['temporal_reference'], strict=True)
        return reference
    if best_a is not None:
        reference_checkpoint = load_checkpoint(best_a, initialization=True)
        if reference_checkpoint['stage'] != 'a':
            raise ValueError('--best-a must refer to a stage-A checkpoint')
        validate_training_checkpoint(reference_checkpoint, reference_checkpoint['config'], initialization=True)
        if reference_checkpoint['config']['fusion'] != fusion_config:
            raise ValueError('best-A reference fusion recipe mismatch')
        if reference_checkpoint['config'].get('model', {}).get('ablation') != ablation:
            raise ValueError('Stage-A reference ablation configuration mismatch')
        return TemporalReference.from_best_a(model, reference_checkpoint, fusion_config=fusion_config,
                                            ablation=ablation)
    if checkpoint is not None and checkpoint['stage'] == 'a':
        return TemporalReference.from_best_a(model, checkpoint, fusion_config=fusion_config,
                                            ablation=ablation)
    return None


def temporal_weight_factor(completed_steps, warmup_steps=500):
    """First update uses 0.1 of the final weights; update 501 uses full weights."""
    if warmup_steps == 0:
        return 1.
    return .1 + .9 * min(max(completed_steps / warmup_steps, 0.), 1.)


def joint_learning_rate(completed_steps, settings):
    """Pure step-based schedule: resume has no hidden scheduler state to reset."""
    fraction = min(max(completed_steps / max(settings['max_steps'] - 1, 1), 0.), 1.)
    initial, minimum = settings['task_lr'], settings.get('task_lr_min', 5e-6)
    return minimum + .5 * (initial - minimum) * (1. + math.cos(math.pi * fraction))


def train_stage(config, stage, *, model=None, datasets=None, resume=None,
                init_from=None, init_from_v6=None, best_a=None, device=None, validate_only=False,
                allow_world_size_change=False, init_from_pretrained=False):
    """Python injection accepts a real model with explicit test backbones only.

    CLI entry points never inject a model/dataset and always run preflight first.
    """
    config = deepcopy(config)
    validate_config(config)
    if stage != 'joint' or sum(value is not None for value in (resume, init_from, init_from_v6))+int(init_from_pretrained) > 1:
        raise ValueError('Specify joint and only one of resume/init_from/init_from_v6')
    if stage in ('b', 'c', 'joint') and not init_from_pretrained and all(value is None for value in (resume, init_from, init_from_v6)):
        raise ValueError(f'Fresh joint training requires --init-from or --init-from-v6')
    validation_enabled=config.get('validation_enabled',True)
    if validate_only and not validation_enabled:
        raise ValueError('--validate-only requires validation_enabled=true')
    if config.get('initialization')=='pretrained_backbones_only' and not (init_from_pretrained or resume):
        raise ValueError('This recipe requires --init-from-pretrained or its own --resume')
    # Each DDP rank performs many small CPU camera fits; avoid nested CV thread pools.
    import cv2
    cv2.setNumThreads(1)
    device, rank, world_size = distributed_context(device or 'cuda')
    destination = Path(config['run_dir']).expanduser().resolve() / stage
    occupied = [False]
    if rank == 0 and resume is None:
        occupied[0] = destination.is_symlink() or (destination.exists() and
                       (not destination.is_dir() or any(destination.iterdir())))
    if world_size > 1:
        dist.broadcast_object_list(occupied, src=0)
    if occupied[0]:
        raise FileExistsError(f'Stage destination is occupied; use explicit --resume or a fresh run: {destination}')
    settings = config['stages'][stage]
    loader_settings = stage_loader_settings(config, stage, device)
    batch_size, accumulation = settings['batch_size'], settings['accumulation']
    effective_clips = batch_size * accumulation * world_size
    seed_everything(config['seed'])
    manifest = resolve_manifest(config)
    splits=('train','val') if validation_enabled else ('train',)
    if datasets is None:
        datasets = {split: AsyncClipDataset(config['data']['root'], manifest, split,
                                            config['data']['crop_size'], training=split == 'train')
                    for split in splits}
    for split in splits:
        if datasets[split].manifest['fingerprint'] != manifest['fingerprint']:
            raise ValueError('Dataset manifest identity mismatch')
        if not datasets[split].groups:
            raise ValueError(f'{split} split has no usable windows')
    source_checkpoint, initialization_metadata = None, None
    resume_topology_history = []
    if init_from_v6 is not None:
        source_checkpoint, initialization_metadata = load_v6_initialization(
            init_from_v6, config, manifest_identity=manifest['fingerprint'])
    if resume is not None or init_from is not None:
        source_checkpoint = load_checkpoint(resume or init_from, manifest_identity=manifest['fingerprint'],
                                            initialization=resume is None)
        validate_training_checkpoint(source_checkpoint, config, initialization=resume is None)
        if resume is not None and source_checkpoint['model_config'] != config['model']:
            raise ValueError('Checkpoint model architecture configuration mismatch')
        if init_from is not None and stage in ('b', 'c'):
            expected_stage = 'a' if stage == 'b' else 'b'
            if source_checkpoint['stage'] not in (expected_stage, stage):
                raise ValueError(f'Fresh stage {stage.upper()} requires the previous stage or same-stage weights-only fork')
        if resume is not None:
            if source_checkpoint['stage'] != stage:
                raise ValueError('Same-stage resume requires matching stage')
            resume_topology_history = deepcopy(source_checkpoint.get('resume_topology_history', []))
            transition = validate_resume_topology(source_checkpoint, config, world_size,
                                                  allow_world_size_change=allow_world_size_change)
            if transition is not None:
                transition['source_checkpoint'] = str(Path(resume).resolve())
                transition['source_checkpoint_sha256'] = _checkpoint_sha256(resume)
                resume_topology_history.append(transition)
            if early_stopped_checkpoint(source_checkpoint):
                raise ValueError('This stage already completed by early stopping; use pipeline --resume-run '
                                 'to inspect the completed run, or --init-from in a fresh run')
            ensure_stage_can_extend(config['run_dir'], stage, source_checkpoint['step'], settings['max_steps'])
    if source_checkpoint is not None and initialization_metadata is None:
        initialization_metadata = deepcopy(source_checkpoint.get('initialization_metadata'))
    model = model if model is not None else build_model(config['model'], config.get('pretrained_root'))
    if init_from_pretrained:
        from .models.initialization import initialize_task_parameters
        initialization_metadata=initialize_task_parameters(model,config['seed'])
        initialization_metadata['asset_fingerprints']=deepcopy(config.get('asset_fingerprints',{}))
    if resume and config.get('initialization')=='pretrained_backbones_only' and (initialization_metadata or {}).get('kind')!='pretrained_backbones_only':
        raise ValueError('Fresh-task recipe cannot resume inherited task weights')
    if source_checkpoint is not None:
        if init_from_v6 is not None:
            load_v6_weights(model, source_checkpoint)
        elif resume is not None:
            model.load_state_dict(source_checkpoint['model'], strict=True)
        else:
            load_initial_weights(model, source_checkpoint, config['model'])
    # Stale gradients from an earlier in-process stage are not optimizer state.
    model.zero_grad(set_to_none=True)
    groups = parameter_groups(model, stage)
    for group in groups:
        group['lr'] = settings[group['name'] + '_lr']
    model.to(device)
    from .models.ablation import mode_of
    mode = mode_of(config)
    architecture, recipe = identifiers(config)
    optimizer = torch.optim.AdamW(groups, weight_decay=config['optimizer']['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=False)  # bf16 does not require scaling.
    reference = (_reference(model, source_checkpoint, best_a, config['fusion'],
                            ablation=config['model'].get('ablation'))
                 if stage in ('b', 'c', 'joint') else None)
    if init_from_pretrained:
        reference=TemporalReference(model.ir_motion,model.vis_motion,fusion_config=config['fusion'],
                                    ablation=config['model'].get('ablation'))
    if stage in ('c', 'joint') and reference is None:
        raise ValueError('Joint training requires the frozen temporal reference carried by the initialization checkpoint')
    if stage == 'b' and reference is None:
        raise ValueError('Stage B must carry best-A reference; use a source with temporal_reference or provide --best-a')
    if reference is not None:
        reference.to(device).eval()
    objective = StageObjective(model, stage, config['fusion']).to(device)
    distributed = world_size > 1
    wrapped = DistributedDataParallel(objective, device_ids=[device.index] if device.type == 'cuda' else None,
                                       find_unused_parameters=True) if distributed else objective
    step, logical_count, best_score = 0, 0, None
    training_mix = {}
    if resume is not None:
        optimizer.load_state_dict(source_checkpoint['optimizer'])
        # Batch/accumulation/max_steps are caller-overridable; restored optimizer LRs
        # deliberately retain the resumed schedule/state, rather than restarting it.
        scaler.load_state_dict(source_checkpoint['scaler'])
        step, logical_count = source_checkpoint['step'], source_checkpoint['logical_sample_count']
        best_score = source_checkpoint['best_score']
        training_mix = source_checkpoint.get('training_mix', {})
        if rank < source_checkpoint['world_size']:
            restore_rng(source_checkpoint['rng_by_rank'][rank])
        else:
            # A new rank has no previous local RNG stream. This explicit seed
            # is recorded in the topology history; optimizer and global cursor
            # are restored identically on every rank.
            seed_everything(transition['added_rank_seeds'][str(rank)])
    if step > settings['max_steps']:
        raise ValueError('Checkpoint step exceeds configured max_steps')
    early_stopping = EarlyStopping(settings['early_stopping'], config['validate_every'],
                                   saved_state=source_checkpoint.get('early_stopping') if resume else None,
                                   resume_step=step if resume else None, best_score=best_score)
    termination_reason = None
    run_dir = Path(config['run_dir']).expanduser().resolve() / stage
    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / 'resolved_paths.json', {'data_root': str(Path(config['data']['root']).resolve()),
                                                    'pretrained_root': config.get('pretrained_root'),
                                                    'run_dir': str(run_dir)})
    if distributed:
        dist.barrier()
    latest_path, best_path = run_dir / 'latest.pt', run_dir / 'best.pt'
    if resume is not None and best_score is not None:
        source_best = best_path if best_path.exists() else Path(resume).parent / 'best.pt'
        if not source_best.is_file():
            if config.get('checkpoint_policy', 'best') == 'best':
                raise FileNotFoundError('Resume requires the paired best.pt snapshot; copy the full stage/run directory')
        else:
            best_snapshot = load_checkpoint(source_best, manifest_identity=manifest['fingerprint'])
            validate_training_checkpoint(best_snapshot, config)
            if best_snapshot['stage'] != stage or best_snapshot['model_config'] != config['model']:
                raise ValueError('Resume paired best checkpoint has a different stage/model')
            if rank == 0 and not best_path.exists():
                atomic_save(best_snapshot, best_path)
    if distributed:
        dist.barrier()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    last_terms, last_teacher_diagnostics, last_grad, throughput = {}, {}, 0., 0.
    last_timing = {}
    progress = TrainingProgress(stage, settings['max_steps'], initial=step, rank=rank)
    def status(phase, validation=None, error=None):
        if rank != 0:
            return
        memory = ({'allocated_bytes': torch.cuda.memory_allocated(device),
                   'reserved_bytes': torch.cuda.memory_reserved(device),
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(device)}
                  if device.type == 'cuda' else {'allocated_bytes': 0, 'reserved_bytes': 0, 'peak_allocated_bytes': 0})
        record = {'phase': phase, 'stage': stage, 'step': step, 'max_steps': settings['max_steps'],
                  **({'resume_topology_history': deepcopy(resume_topology_history)} if resume_topology_history else {}),
                  'training_recipe': recipe, 'loss_fingerprint': loss_fingerprint(config),
                  'initialization_metadata': deepcopy(initialization_metadata),
                  'segmentation_enabled': config['model']['target_gate']['segmentation_enabled'],
                  'ablation_mode': mode,
                  'logical_sample_count': logical_count, 'best_score': best_score,
                  'validation_only': bool(validate_only),
                  'early_stopping': early_stopping.state_dict(), 'termination_reason': termination_reason,
                  'loss': last_terms, 'teacher_diagnostics': last_teacher_diagnostics,
                  'learning_rates': {g['name']: g['lr'] for g in optimizer.param_groups},
                  'temporal_weight_factor': temporal_weight_factor(step, settings.get('temporal_warmup_steps', 500)),
                  'grad_norm': last_grad, 'clips_per_second': throughput,
                  'timing': last_timing,
                  'data_loading': dict(loader_settings),
                  'queries_per_second': throughput * (3 if stage in ('c', 'joint') else 1),
                  'batch_size_per_rank': batch_size, 'accumulation_steps': accumulation,
                  'world_size': world_size, 'effective_clips': effective_clips,
                  'effective_queries': effective_clips * (3 if stage in ('c', 'joint') else 1),
                  'memory': memory, 'source_protocol_counts': training_mix,
                  'manifest_identity': manifest['fingerprint']}
        if validation is not None:
            record['validation'] = {key: validation[key] for key in ('score', 'components', 'missing', 'observed')}
            for key in ('selection_eligible', 'support', 'fusion_diagnostics'):
                if key in validation:
                    record['validation'][key] = validation[key]
        if error is not None:
            record['error'] = str(error)
        write_json(run_dir / 'status.json', record)
        append_json(run_dir / 'metrics.jsonl', record)
        progress.status(record)

    def snapshot(completed=False):
        local_rng = capture_rng()
        states = [None] * world_size
        if distributed:
            dist.all_gather_object(states, local_rng)
        else:
            states[0] = local_rng
        if rank != 0:
            return None
        return {'format_version': 1, 'architecture': architecture,
                **({'resume_topology_history': deepcopy(resume_topology_history)} if resume_topology_history else {}),
                'training_recipe': recipe, 'loss_fingerprint': loss_fingerprint(config),
                'initialization_metadata': deepcopy(initialization_metadata),
                'segmentation_enabled': config['model']['target_gate']['segmentation_enabled'],
                  'ablation_mode': mode,
                'resources': resources_for(config), 'asset_fingerprints': deepcopy(config.get('asset_fingerprints', {})),
                'model_config': deepcopy(config['model']),
                'config': portable_config(config), 'manifest_identity': manifest['fingerprint'],
                'model': model.state_dict(),
                'temporal_reference': reference.state_dict() if reference is not None else None,
                'stage': stage, 'step': step, 'logical_sample_count': logical_count,
                'best_score': best_score, 'optimizer': optimizer.state_dict(),
                'schedule': {'kind': 'cosine_joint_v1', 'completed_steps': step,
                             'max_steps': settings['max_steps'], 'initial_lr': settings['task_lr'],
                             'minimum_lr': settings.get('task_lr_min', 5e-6),
                             'temporal_warmup_steps': settings.get('temporal_warmup_steps', 500)},
                'scaler': scaler.state_dict(), 'rng_by_rank': states, 'world_size': world_size,
                'completed': completed, 'training_mix': dict(training_mix),
                'early_stopping': early_stopping.state_dict(), 'termination_reason': termination_reason}

    def validation_and_save(completed=False):
        nonlocal best_score, termination_reason
        def validation_status(phase, result):
            # Propagate a rank-zero write failure to every rank.
            status_error = [None]
            try:
                status(phase, result)
            except Exception as exc:
                status_error[0] = f'{type(exc).__name__}: {exc}'
            if distributed:
                dist.broadcast_object_list(status_error, src=0)
            if status_error[0] is not None:
                raise RuntimeError(f'Validation status write failed: {status_error[0]}')
        report, error = None, None
        if rank == 0:
            try:
                with progress.validation() as callback:
                    report = validate(model, datasets['val'], stage, clips=config['validation_clips'],
                                      seed=config['seed'], device=device, reference=reference,
                                      fusion_config=config['fusion'], progress_callback=callback,
                                      output_dir=run_dir / 'validation' / f'step_{step:08}')
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
        payload = [report, error]
        if distributed:
            dist.broadcast_object_list(payload, src=0)
        report, error = payload
        if error is not None:
            raise RuntimeError(f'Validation failed: {error}')
        # All ranks receive the same validation score and make the same stop decision
        # before gathering checkpoint RNG state or entering the next training step.
        eligible = bool(report.get('selection_eligible', True))
        if eligible and not validate_only:
            early_stopping.update(step, report['score'])
        improved = eligible and (best_score is None or report['score'] < best_score)
        if improved:
            best_score = report['score']
        termination_reason = (None if validate_only else
                              'max_steps' if completed else
                              'early_stopping' if early_stopping.stopped else None)
        completed = False if validate_only else completed or early_stopping.stopped
        state = snapshot(completed)
        publication_error = [None]
        if rank == 0:
            try:
                atomic_save(state, latest_path)
                if improved:
                    atomic_save(state, best_path)
            except Exception as exc:
                publication_error[0] = f'{type(exc).__name__}: {exc}'
        if distributed:
            dist.broadcast_object_list(publication_error, src=0)
        if publication_error[0] is not None:
            raise RuntimeError(f'Checkpoint save failed: {publication_error[0]}')
        validation_status('validated' if validate_only else 'completed' if completed else 'validation', report)
        return report

    def save_without_validation(completed=False):
        nonlocal termination_reason
        termination_reason='max_steps' if completed else None
        state=snapshot(completed)
        error=[None]
        if rank==0:
            try:atomic_save(state,latest_path)
            except Exception as exc:error[0]=f'{type(exc).__name__}: {exc}'
        if distributed:dist.broadcast_object_list(error,src=0)
        if error[0]:raise RuntimeError('Checkpoint save failed: '+error[0])
        status('completed' if completed else 'training')

    loader = batches = batch = None
    try:
        from .models.ablation import describe_model
        if rank == 0:
            print(json.dumps(describe_model(model, objective.loss), allow_nan=False), flush=True)
            print(json.dumps({'initialization_metadata':initialization_metadata,'validation_enabled':validation_enabled,
                              'checkpoint_policy':config.get('checkpoint_policy','best'),'max_steps':settings['max_steps']},allow_nan=False),flush=True)
        status('startup')
        if validation_enabled:validation_and_save(completed=step == settings['max_steps'])
        else:save_without_validation(completed=step == settings['max_steps'])
        if not validate_only and step < settings['max_steps'] and not early_stopping.stopped:
            remaining = settings['max_steps'] - step
            sampler = GlobalSampleSampler(datasets['train'], logical_count,
                                           remaining * effective_clips, batch_size, rank, world_size, config['seed'],
                                           stage=stage)
            generator = torch.Generator().manual_seed(config['seed'] + rank)
            loader = DataLoader(datasets['train'], batch_size=batch_size, sampler=sampler,
                                collate_fn=collate_clips, num_workers=loader_settings['workers'],
                                pin_memory=loader_settings['pin_memory'], generator=generator,
                                **({'prefetch_factor': loader_settings['prefetch_factor']}
                                   if loader_settings['workers'] else {}))
            batches = iter(loader)
            model.train()
            while step < settings['max_steps'] and not early_stopping.stopped:
                started = time.perf_counter()
                data_wait_seconds = 0.
                for group in optimizer.param_groups:
                    group['lr'] = joint_learning_rate(step, settings)
                optimizer.zero_grad(set_to_none=True)
                totals = []
                counts = scheduled_counts(datasets['train'], stage, logical_count, effective_clips, config['seed'])
                weights = dict(objective.loss.weights)
                if stage in ('c', 'joint'):
                    ramp = temporal_weight_factor(step, settings.get('temporal_warmup_steps', 500))
                    weights['temporal'] = objective.loss.temporal_weight * ramp
                    weights['temporal_lowfreq'] = objective.loss.temporal_lowfreq_weight * ramp
                for micro in range(accumulation):
                    data_started = time.perf_counter()
                    batch = next(batches)
                    data_wait_seconds += time.perf_counter() - data_started
                    start = logical_count + micro * world_size * batch_size + rank * batch_size
                    queries = query_indices(stage, range(start, start + batch_size), config['seed'])
                    if observation_counts(batch, stage, queries) != scheduled_counts(
                            datasets['train'], stage, start, batch_size, config['seed']):
                        raise ValueError('Scheduled observation counts disagree with actual selected model inputs')
                    synchronize = nullcontext() if not distributed or micro == accumulation - 1 else wrapped.no_sync()
                    with synchronize:
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                            enabled=device.type == 'cuda' and config['precision'] == 'bf16'):
                            statistics = wrapped(batch, queries=queries, reference=reference, return_stats=True)
                            loss, terms = effective_update_objective(statistics, weights, counts, world_size)
                        valid = torch.isfinite(loss).to(dtype=torch.int32)
                        if distributed:
                            dist.all_reduce(valid, op=dist.ReduceOp.MIN)
                        if not valid.item():
                            raise FloatingPointError('Nonfinite loss')
                        scaler.scale(loss).backward()
                    totals.append({name: (numerator.detach(), denominator.detach())
                                   for name, (numerator, denominator) in statistics.items()})
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_([p for group in optimizer.param_groups for p in group['params']],
                                                           config['optimizer']['grad_clip'])
                valid = torch.isfinite(grad_norm).to(dtype=torch.int32)
                if distributed:
                    dist.all_reduce(valid, op=dist.ReduceOp.MIN)
                if not valid.item():
                    raise FloatingPointError('Nonfinite gradient norm')
                scaler.step(optimizer)
                scaler.update()
                statistics = sum_statistics(totals)
                if distributed:
                    values = torch.stack([value for pair in statistics.values() for value in pair])
                    dist.all_reduce(values)
                    statistics = {name: (values[2*i], values[2*i+1]) for i, name in enumerate(statistics)}
                normalizers = statistic_denominators(statistics, counts)
                denominators_ok = torch.stack([value[1] == normalizers[name]
                                               for name, value in statistics.items()]).all()
                if not denominators_ok.item():
                    raise ValueError('Global effective-update denominator mismatch')
                normalized = normalize_statistics(statistics, weights)
                diagnostic_names = [name for name in statistics if name.startswith('teacher_')]
                # One device-to-host copy for all reported scalars. Keep teacher
                # denominators so unsupported diagnostics still serialize as null.
                scalars = torch.stack([grad_norm, *normalized.values(),
                                      *[value for name in diagnostic_names for value in statistics[name]]])
                host = scalars.detach().cpu().tolist()
                last_grad = host[0]
                last_terms = dict(zip(normalized, host[1:1 + len(normalized)]))
                diagnostic_values = host[1 + len(normalized):]
                last_teacher_diagnostics = {
                    name: diagnostic_values[2*i] / diagnostic_values[2*i+1]
                    if diagnostic_values[2*i+1] > 0 else None
                    for i, name in enumerate(diagnostic_names)}
                for index in range(logical_count, logical_count + effective_clips):
                    key = balanced_sample_key(datasets['train'], index, config['seed'])
                    name = datasets['train'].windows[key.window_index]['source'] + '/' + key.protocol
                    training_mix[name] = training_mix.get(name, 0) + 1
                logical_count += effective_clips
                step += 1
                step_seconds = max(time.perf_counter() - started, 1e-9)
                throughput = effective_clips / step_seconds
                last_timing = {'step_seconds': step_seconds, 'data_wait_seconds': data_wait_seconds,
                               'data_wait_fraction': data_wait_seconds / step_seconds}
                progress.fields['state'] = 'training'
                progress.update(step, loss=last_terms.get('total'), clips_per_second=throughput,
                                memory_bytes=torch.cuda.memory_reserved(device)
                                if device.type == 'cuda' and rank == 0 else None)
                complete = step == settings['max_steps']
                if not validation_enabled:
                    if complete or step % config['save_every']==0:save_without_validation(completed=complete)
                elif complete or step % config['validate_every'] == 0:
                    validation_and_save(completed=complete)
                elif step % config['save_every'] == 0:
                    state = snapshot()
                    if rank == 0:
                        atomic_save(state, latest_path)
                if not complete and not early_stopping.stopped and step % config['log_every'] == 0:
                    status('training')
        if validation_enabled and best_score is None and not validate_only:
            raise RuntimeError('No evaluable validation checkpoint: inspect target/temporal support; '
                               'a zero-support score cannot select best weights or start the next stage')
    except Exception as exc:
        status('failed', error=exc)
        raise
    except KeyboardInterrupt:
        status('interrupted')
        raise
    finally:
        # Close the loader and retire the last pinned batch before the pipeline
        # releases retired cached host memory.
        batches = loader = batch = None
        progress.close()
    if distributed:
        dist.barrier()
    return {'latest': str(latest_path), 'best': str(best_path) if validation_enabled else None, 'stage': stage,
            'step': step, 'logical_sample_count': logical_count, 'best_score': best_score,
            'termination_reason': termination_reason, 'validation_only': bool(validate_only),
            'phase': 'validated' if validate_only else 'completed'}


def main(argv=None):
    from .preflight import preflight
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    parser.add_argument('--stage', default='joint', choices=('joint',))
    parser.add_argument('--validate-only', action='store_true',
                        help='run sampled validation and save a resumable checkpoint without updating weights')
    initialize = parser.add_mutually_exclusive_group()
    initialize.add_argument('--resume')
    initialize.add_argument('--init-from')
    initialize.add_argument('--init-from-v6', help='Explicit v6 weights-only migration into fresh SAFI-Net joint training')
    initialize.add_argument('--init-from-pretrained',action='store_true')
    parser.add_argument('--best-a')
    parser.add_argument('--batch-size', type=positive_int)
    parser.add_argument('--accumulation-steps', type=positive_int)
    parser.add_argument('--max-steps', type=positive_int)
    parser.add_argument('--allow-world-size-change', action='store_true',
                        help='Explicitly resize DDP on resume while retaining effective clips per update')
    args = parser.parse_args(argv)
    resume_checkpoint = load_checkpoint(args.resume) if args.resume else None
    saved = resume_checkpoint['config'] if resume_checkpoint is not None else {}
    config = merge_config(load_config(), saved)
    if args.config:
        with Path(args.config).open() as handle:
            config = merge_config(config, yaml.safe_load(handle) or {})
    config = apply_runtime_arguments(config, args)
    for argument, field in (('batch_size', 'batch_size'), ('accumulation_steps', 'accumulation'), ('max_steps', 'max_steps')):
        if getattr(args, argument) is not None:
            config['stages'][args.stage][field] = getattr(args, argument)
    validate_config(config)
    if resume_checkpoint is not None:
        validate_training_checkpoint(resume_checkpoint, config)
    preflight(config, device=args.device)
    try:
        train_stage(config, args.stage, resume=args.resume, init_from=args.init_from, init_from_v6=args.init_from_v6,
                    best_a=args.best_a, device=args.device, validate_only=args.validate_only,
                    allow_world_size_change=args.allow_world_size_change,init_from_pretrained=args.init_from_pretrained)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
