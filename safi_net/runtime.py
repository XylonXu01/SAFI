"""Shared deterministic sampling, configuration and finite runtime reporting."""
import argparse
from copy import deepcopy
import gc
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch import nn
from torch.utils.data import Sampler
import yaml

from .checkpoints import ARCHITECTURE
from .data import balanced_sample_key
from .early_stopping import normalize_early_stopping
from .losses import LossComputer, normalize_statistics, sum_statistics, normalize_training_fusion_config, luminance, build_target_supervision, target_supervision_statistics
from .paths import PACKAGE_ROOT, resolve_paths
from .split_manifest import manifest_fingerprint
from .types import TemporalInputs
from .temporal import canonicalize_inputs


def release_stage_memory(device='cuda'):
    """Release retired loader/model buffers before a new stage uses new shapes.

    CUDA's pinned-host allocator otherwise retains the previous stage's large
    batch buffers. Only unused cached blocks are released, after outstanding
    transfers finish; checkpoint state and live tensors remain intact.
    """
    device = torch.device(device or 'cuda')
    gc.collect()
    if device.type != 'cuda' or not torch.cuda.is_initialized():
        return {'cuda_initialized': False}
    torch.cuda.synchronize(device)
    get_stats = getattr(torch.cuda.memory, 'host_memory_stats', None)
    before = get_stats() if get_stats else {}
    empty_host = getattr(getattr(torch, 'accelerator', None), 'empty_host_cache', None)
    if empty_host is not None:
        empty_host()
    torch.cuda.empty_cache()
    after = get_stats() if get_stats else {}
    return {'cuda_initialized': True, 'host_cache_release_supported': empty_host is not None,
            'pinned_owned_bytes_before': before.get('allocated_bytes.current'),
            'pinned_owned_bytes_after': after.get('allocated_bytes.current'),
            'pinned_active_bytes_after': after.get('active_bytes.current')}


def merge_config(base, updates):
    result = deepcopy(base) if isinstance(base, dict) else {}
    for key, value in updates.items():
        result[key] = merge_config(result.get(key, {}), value) if isinstance(value, dict) else deepcopy(value)
    return result


def load_config(path=None, overrides=None):
    with (PACKAGE_ROOT / 'configs/base.yaml').open() as handle:
        config = yaml.safe_load(handle)
    config = merge_config(config, {'model': {'base_channels': 32, 'semantic_dim': 128,
                                            'transport': {'radius': 2, 'mode': 'softmax',
                                                          'epsilon': .05, 'rho': 1., 'iterations': 10}},
                                   'workers': config.get('workers', 4)})
    if path:
        with Path(path).open() as handle:
            config = merge_config(config, yaml.safe_load(handle) or {})
    config = merge_config(config, overrides or {})
    from .relocation import resolved_values
    config = resolved_values(config)
    validate_config(config)
    return config


def validate_config(config):
    if type(config.get('validation_enabled',True)) is not bool:
        raise ValueError('validation_enabled must be boolean')
    if config.get('initialization','legacy') not in ('legacy','pretrained_backbones_only'):
        raise ValueError('Unknown initialization policy')
    if not config.get('validation_enabled',True):
        if config.get('checkpoint_policy','best')!='latest' or config['stages']['joint'].get('early_stopping',{}).get('enabled',True):
            raise ValueError('Training without validation requires latest checkpoints and disabled early stopping')
    from .compat.identity import normalize_config
    config['architecture'] = normalize_config(config)['architecture']
    if config['architecture'] != ARCHITECTURE:
        raise ValueError('Unsupported architecture')
    if config['precision'] not in ('bf16', 'float32'):
        raise ValueError('precision must be bf16 or float32')
    if config.get('checkpoint_policy', 'best') not in ('best', 'latest'):
        raise ValueError('checkpoint_policy must be best or latest')
    for field in ('save_every', 'log_every', 'validate_every', 'validation_clips'):
        if not isinstance(config[field], int) or config[field] < 1:
            raise ValueError(f'{field} must be a positive integer')
    if not isinstance(config['workers'], int) or config['workers'] < 0:
        raise ValueError('workers must be nonnegative')
    if type(config.setdefault('pin_memory', True)) is not bool:
        raise ValueError('pin_memory must be a boolean')
    prefetch = config.setdefault('prefetch_factor', 1)
    if type(prefetch) is not int or prefetch < 1:
        raise ValueError('prefetch_factor must be a positive integer')
    for stage in ('joint',):
        stage_workers = config['stages'][stage].get('workers')
        if stage_workers is not None and (type(stage_workers) is not int or stage_workers < 0):
            raise ValueError(f'{stage}.workers must be nonnegative or null to inherit')
        stage_pin = config['stages'][stage].get('pin_memory')
        if stage_pin is not None and type(stage_pin) is not bool:
            raise ValueError(f'{stage}.pin_memory must be a boolean or null to inherit')
        for field in ('max_steps', 'batch_size', 'accumulation'):
            value = config['stages'][stage][field]
            if not isinstance(value, int) or value < 1:
                raise ValueError(f'{stage}.{field} must be a positive integer')
        config['stages'][stage]['early_stopping'] = normalize_early_stopping(
            config['stages'][stage].get('early_stopping'), stage)
    if config['optimizer']['grad_clip'] <= 0 or config['optimizer']['weight_decay'] < 0:
        raise ValueError('Invalid optimizer settings')
    config['fusion'] = normalize_training_fusion_config(config.get('fusion'))
    if config['fusion']['residual_scale'] != .25:
        raise ValueError('SAFI-Net architecture requires fusion.residual_scale=.25')
    gate = config['model'].setdefault('target_gate', {})
    if gate.get('segmentation_enabled', False):
        raise ValueError('SAFI-Net full-image training does not support segmentation')
    gate['segmentation_enabled'] = False
    config['model']['fusion_region'] = 'all'
    settings = config['stages']['joint']
    if type(settings.setdefault('temporal_warmup_steps', 500)) is not int or settings['temporal_warmup_steps'] < 0:
        raise ValueError('joint.temporal_warmup_steps must be nonnegative')
    for key, default in (('task_lr', 1e-5), ('task_lr_min', 5e-6)):
        value = settings.setdefault(key, default)
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'joint.{key} must be finite and positive')
    if settings['task_lr_min'] > settings['task_lr']:
        raise ValueError('joint.task_lr_min must not exceed task_lr')
    if 'ablation' in config['model']:
        from .models.ablation import normalize_ablation
        value = normalize_ablation(config['model']['ablation'])
        if value['mode'] == 'full' and set(config['model']['ablation']) <= {'mode'}:
            config['model'].pop('ablation')
        else:
            config['model']['ablation'] = value
    from .models.person import normalize_target_classes
    perception = config['model'].setdefault('perception', {})
    perception['target_classes'] = list(normalize_target_classes(perception.get('target_classes')))
    config.pop('video_validation', None)


def stage_loader_settings(config, stage, device='cuda'):
    """Resolve stage overrides; global worker/pinning flags remain defaults.

    Missing or null stage fields inherit the global setting. CPU loaders never
    pin CUDA host memory, and zero workers have no prefetch queue.
    """
    if stage not in ('a', 'b', 'c', 'joint'):
        raise ValueError('Unknown loader training stage')
    settings = config['stages'][stage]
    workers = settings.get('workers')
    workers = config['workers'] if workers is None else workers
    pin_memory = settings.get('pin_memory')
    pin_memory = config.get('pin_memory', True) if pin_memory is None else pin_memory
    return {'workers': workers, 'pin_memory': bool(pin_memory and torch.device(device).type == 'cuda'),
            'prefetch_factor': config.get('prefetch_factor', 1) if workers else None}


def resolve_manifest(config):
    manifest = config['data'].get('manifest') or PACKAGE_ROOT / 'artifacts/split_async.json'
    if not isinstance(manifest, dict):
        with Path(manifest).open() as handle:
            manifest = json.load(handle)
    if manifest.get('version') != 1 or manifest.get('fingerprint') != manifest_fingerprint(manifest):
        raise ValueError('Manifest version/fingerprint mismatch')
    return manifest


def portable_config(config):
    result = deepcopy(config)
    result.pop('pretrained_root', None)
    result.pop('run_dir', None)
    result['data'].pop('root', None)
    result['data'].pop('manifest', None)
    return result


def ensure_stage_can_extend(run_dir, stage, step, max_steps):
    if step >= max_steps or stage == 'joint':
        return
    successors = [later for later in 'abc'['abc'.index(stage) + 1:]
                  if any((Path(run_dir) / later / name).exists() for name in ('latest.pt', 'best.pt'))]
    if successors:
        raise ValueError(f'Cannot extend stage {stage} with successor checkpoints {successors}; '
                         'use --init-from in a fresh run for a weights-only fork')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


class GlobalSampleSampler(Sampler):
    """Partition a fixed global clip stream in rank-contiguous microbatches."""
    def __init__(self, dataset, start, count, batch_size, rank=0, world_size=1, seed=3407, stage=None):
        if start < 0 or count < 0 or batch_size < 1 or not 0 <= rank < world_size:
            raise ValueError('Invalid global sampler bounds')
        if count % (batch_size * world_size):
            raise ValueError('Global sample count must contain complete distributed microbatches')
        if stage not in (None, 'a', 'b', 'c', 'joint'):
            raise ValueError('Unknown sampler training stage')
        self.dataset, self.start, self.count = dataset, start, count
        self.batch_size, self.rank, self.world_size, self.seed = batch_size, rank, world_size, seed
        self.stage = stage

    def indices(self):
        width = self.batch_size * self.world_size
        for offset in range(0, self.count, width):
            for local in range(self.batch_size):
                yield self.start + offset + self.rank * self.batch_size + local

    def __iter__(self):
        for index in self.indices():
            key = balanced_sample_key(self.dataset, index, self.seed)
            if self.stage == 'b':
                key = replace(key, query_index=query_indices('b', [index], self.seed)[0])
            yield key

    def __len__(self):
        return self.count // self.world_size


def query_indices(stage, global_indices, seed=3407):
    if stage in ('c', 'joint'):
        return None
    choices = (0, 2) if stage == 'a' else (0, 1, 2)
    return [random.Random(f'{seed}:query:{i}').choice(choices) for i in global_indices]


def select_queries(batch, queries):
    if 'selected_queries' in batch:
        _validate_selected_queries(batch, queries)
        return batch['inputs'][0], batch['ir_targets'][:, 0], batch['vis_targets'][:, 0]
    selected = TemporalInputs.cat([
        batch['inputs'][query].select(slice(row, row+1))
        for row, query in enumerate(queries)])
    ir = torch.stack([batch['ir_targets'][row, query] for row, query in enumerate(queries)])
    vis = torch.stack([batch['vis_targets'][row, query] for row, query in enumerate(queries)])
    return selected, ir, vis


def _validate_selected_queries(batch, queries):
    selected = batch['selected_queries']
    if queries is None or list(queries) != selected:
        raise ValueError('Worker-selected queries disagree with the deterministic training schedule')
    if (len(selected) != batch['ir_targets'].shape[0] or len(batch['inputs']) != 1
            or batch['ir_targets'].shape[1] != 1 or batch['vis_targets'].shape[1] != 1):
        raise ValueError('Invalid worker-selected query batch')


def scheduled_counts(dataset, stage, start, count, seed=3407):
    """Observation-only denominators for a contiguous logical sample interval."""
    result = {'ir': 0, 'vis': 0, 'queries': count * (3 if stage in ('c', 'joint') else 1), 'clips': count}
    for index in range(start, start + count):
        key = balanced_sample_key(dataset, index, seed)
        queries = (0, 1, 2) if stage in ('c', 'joint') else query_indices(stage, [index], seed)
        missing = sum(query != 1 for query in queries)
        result['ir'] += missing
        result['vis'] += missing if key.protocol == 'same' else 0
    return result


def observation_counts(batch, stage, queries=None):
    # Counting observations needs clocks only. Selecting complete image/context
    # tensors here duplicated the large pixel selection performed by the loss.
    observed = [torch.stack(((value.ir_valid & (value.ir_times == value.query_time[:, None])).any(1),
                             (value.vis_valid & (value.vis_times == value.query_time[:, None])).any(1)), 1)
                for value in batch['inputs']]
    if 'selected_queries' in batch:
        if stage in ('c', 'joint'):
            raise ValueError('Joint training requires all three queries')
        _validate_selected_queries(batch, queries)
    elif stage not in ('c', 'joint'):
        flags = torch.stack(observed, 1)
        if queries is None or len(queries) != flags.shape[0]:
            raise ValueError('A query is required for every sample')
        observed = [flags[torch.arange(flags.shape[0], device=flags.device),
                          torch.as_tensor(queries, device=flags.device)]]
    return {'ir': sum(int((~flags[:, 0]).sum()) for flags in observed),
            'vis': sum(int((~flags[:, 1]).sum()) for flags in observed),
            'queries': sum(flags.shape[0] for flags in observed), 'clips': batch['ir_targets'].shape[0]}


def statistic_denominators(statistics, counts):
    def denominator(name):
        if name in ('reference_l1', 'reference_gradient'):
            return counts['vis']
        if name in {prefix + modality for prefix in ('interp_', 'gradient_', 'motion_') for modality in ('ir', 'vis')}:
            return counts[name.rsplit('_', 1)[-1]]
        return counts['clips' if name in ('temporal', 'temporal_support', 'temporal_lowfreq', 'temporal_lowfreq_support') else 'queries']
    # These data-dependent diagnostics never enter the weighted objective. Keep
    # their additive eligible-pair denominator through accumulation/all-reduce.
    pair_diagnostics = {prefix + suffix for prefix in ('temporal', 'temporal_lowfreq')
                        for suffix in ('_pair_support', '_coverage', '_visibility')}
    return {name: value[1] if name in pair_diagnostics or name.startswith('teacher_') else value[1].new_tensor(denominator(name))
            for name, value in statistics.items()}


def effective_update_objective(statistics, weights, counts, world_size=1):
    terms = normalize_statistics(statistics, weights, statistic_denominators(statistics, counts))
    return terms['total'] * world_size, terms


class StageObjective(nn.Module):
    """One DDP forward owns all three joint queries and temporal losses."""
    def __init__(self, model, stage, fusion_config=None):
        super().__init__()
        self.model = model
        self.fusion_config = normalize_training_fusion_config(fusion_config)
        self.loss = LossComputer(stage, self.fusion_config,
                                 ablation=getattr(model, 'config', {}).get('ablation'))
        self.stage = stage

    def forward(self, batch, queries=None, reference=None, return_stats=False):
        device = next(self.model.parameters()).device
        predictions, fused_predictions, masks, observed_vis, per_query = [], [], [], [], []
        gate_cache = {'cache_motion_pairs': False} if self.stage == 'b' else {}
        if self.stage in ('c', 'joint'):
            if 'selected_queries' in batch:
                raise ValueError('Joint training requires all three queries')
            # Reuse these GPU tensors for both per-query and temporal losses.
            ir_targets = batch['ir_targets'].to(device, non_blocking=True)
            vis_targets = batch['vis_targets'].to(device, non_blocking=True)
            selections = [(value, ir_targets[:, q], vis_targets[:, q])
                          for q, value in enumerate(batch['inputs'])]
        else:
            selections = [select_queries(batch, queries)]
        teachers = batch.get('teacher_inputs')
        if teachers is not None and self.stage not in ('c', 'joint') and 'selected_queries' not in batch:
            teachers = [TemporalInputs.cat([teachers[query].select(slice(row, row+1))
                        for row, query in enumerate(queries)])]
        teacher_cache = {'cache_motion_pairs': False} if self.stage == 'b' else {}
        for query_index, (inputs, ir, vis) in enumerate(selections):
            inputs = canonicalize_inputs(inputs.to(device, non_blocking=True))
            ir, vis = ir.to(device, non_blocking=True), vis.to(device, non_blocking=True)
            output = self.model(inputs, gate_cache=gate_cache)
            # Targets are visible to this training-only helper, never to model.forward.
            supervision = (build_target_supervision(self.model, inputs, vis, reference,
                            teacher_inputs=teachers[query_index].to(device, non_blocking=True) if teachers is not None else None,
                            cache=teacher_cache) if self.stage != 'a' else None)
            target_mask = supervision['mask'] if supervision is not None else None
            terms = self.loss(output, inputs, ir, vis, return_stats=True, target_mask=target_mask,
                              target_support=supervision['support'] if supervision is not None else None)
            if supervision is not None:
                terms.update(target_supervision_statistics(supervision))
            if self.stage in ('c', 'joint'):
                predictions.append(luminance(output.fused) - luminance(output.vis_reference.detach()))
                fused_predictions.append(output.fused)
                masks.append(target_mask.detach())
                observed_vis.append(output.observed[:, 1])
            per_query.append(terms)
        statistics = sum_statistics(per_query)
        weights = dict(self.loss.weights)
        if self.stage in ('c', 'joint'):
            if reference is None:
                raise ValueError('Joint training requires the frozen temporal reference')
            if self.loss.ablation['mode'] == 'a4_no_tcl':
                zero = fused_predictions[0].float().sum() * 0
                count = zero.new_tensor(len(ir_targets))
                temporal = {key: (zero, count) for key in ('temporal', 'temporal_lowfreq',
                            'temporal_support', 'temporal_lowfreq_support')}
            else:
                temporal = reference(torch.stack(predictions, 1), ir_targets,
                                     vis_targets, target_masks=torch.stack(masks, 1),
                                     observed_vis=torch.stack(observed_vis, 1),
                                     fused_predictions=torch.stack(fused_predictions, 1), return_statistics=True)
            statistics.update(temporal)
            weights['temporal'] = self.loss.temporal_weight
            weights['temporal_lowfreq'] = self.loss.temporal_lowfreq_weight
        if return_stats:
            return statistics
        terms = normalize_statistics(statistics, weights)
        return terms['total'], terms


def json_ready(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite runtime value cannot be reported as JSON')
    return value


def write_json(path, value):
    path = Path(path)
    contents = json.dumps(json_ready(value), indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(contents)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_json(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(json_ready(value), allow_nan=False) + '\n')


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('must be a nonnegative integer')
    return value


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError('must be a finite nonnegative number')
    return value


def add_runtime_arguments(parser):
    parser.add_argument('--config')
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--pretrained-root')
    parser.add_argument('--manifest')
    parser.add_argument('--run-dir')
    parser.add_argument('--device', default='cuda', help='cpu, cuda, or an explicit CUDA device such as cuda:2')
    parser.add_argument('--precision', choices=('bf16', 'float32'))
    parser.add_argument('--segmentation', choices=('off',), help='SAFI-Net uses full-image fusion without segmentation')
    parser.add_argument('--checkpoint-policy', choices=('best', 'latest'),
                        help='checkpoint used to initialize each following pipeline stage')
    parser.add_argument('--crop-size', type=positive_int)
    parser.add_argument('--workers', type=nonnegative_int,
                        help='default loader workers for stages without a stage-specific override')
    parser.add_argument('--pin-memory', action=argparse.BooleanOptionalAction, default=None,
                        help='default CUDA loader pinning for stages without a stage-specific override')
    parser.add_argument('--prefetch-factor', type=positive_int)
    for stage in ('joint',):
        parser.add_argument('--workers-' + stage, type=nonnegative_int,
                            help=f'override loader workers for stage {stage.upper()}')
        parser.add_argument('--pin-memory-' + stage, action=argparse.BooleanOptionalAction, default=None,
                            help=f'override CUDA loader pinning for stage {stage.upper()}')
    for name in ('validation-clips', 'save-every', 'log-every', 'validate-every'):
        parser.add_argument('--' + name, type=positive_int)
    parser.add_argument('--early-stopping', action=argparse.BooleanOptionalAction, default=None,
                        help='enable or disable joint validation-score early stopping')
    parser.add_argument('--early-stopping-patience', type=positive_int,
                        help='number of validations without significant improvement before stopping')
    parser.add_argument('--early-stopping-min-delta', type=nonnegative_float,
                        help='minimum absolute validation-score decrease for significant improvement')
    for stage in ('joint',):
        parser.add_argument('--early-stopping-min-steps-' + stage, type=nonnegative_int,
                            help=f'do not count early-stopping failures through this stage-{stage.upper()} step')


def apply_runtime_arguments(config, args):
    config = deepcopy(config)
    roots = resolve_paths(args.data_root,
                          args.pretrained_root if args.pretrained_root is not None else config.get('pretrained_root'),
                          args.run_dir if args.run_dir is not None else config.get('run_dir'))
    config['data']['root'] = str(roots.data_root)
    config['pretrained_root'], config['run_dir'] = str(roots.pretrained_root), str(roots.run_dir)
    if getattr(args, 'segmentation', None) is not None:
        config['model'].setdefault('target_gate', {})['segmentation_enabled'] = args.segmentation == 'on'
    if args.manifest is not None:
        config['data']['manifest'] = str(Path(args.manifest).expanduser().resolve())
    if args.crop_size is not None:
        config['data']['crop_size'] = args.crop_size
    for field in ('workers', 'pin_memory', 'prefetch_factor', 'precision', 'checkpoint_policy', 'validation_clips', 'save_every', 'log_every', 'validate_every'):
        if getattr(args, field, None) is not None:
            config[field] = getattr(args, field)
    for stage in ('joint',):
        for field in ('workers', 'pin_memory'):
            override = getattr(args, field + '_' + stage, None)
            if override is not None:
                config['stages'][stage][field] = override
    for stage in ('joint',):
        settings = normalize_early_stopping(config['stages'][stage].get('early_stopping'), stage)
        for field, argument in (
                ('enabled', 'early_stopping'), ('patience', 'early_stopping_patience'),
                ('min_delta', 'early_stopping_min_delta'),
                ('min_steps', 'early_stopping_min_steps_' + stage)):
            if getattr(args, argument, None) is not None:
                settings[field] = getattr(args, argument)
        config['stages'][stage]['early_stopping'] = normalize_early_stopping(settings, stage)
    return config
