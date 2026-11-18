"""Fixed source/protocol-balanced validation, separated missing and observed metrics."""
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from .temporal import canonicalize_inputs
from .checkpoints import preserve_rng
from .data import SampleKey, balanced_sample_key, collate_clips
from .losses import (LossComputer, missing_metrics, normalize_statistics, sum_statistics,
                     luminance, thermal_delta_target, validate_target_mask, build_target_supervision,
                     target_supervision_statistics)
from .fusion_guidance import thermal_saliency
from .runtime import json_ready, write_json


def validation_keys(dataset, clips=150, seed=3407):
    if clips < 1 or not dataset.groups:
        raise ValueError('Validation requires positive clips and a nonempty split')
    return [balanced_sample_key(dataset, index, seed) for index in range(clips)]


def _preview(path, output, ir, vis):
    tiles = [output.fused, output.ir_aux.repeat(1, 3, 1, 1), output.vis_aux,
             ir.repeat(1, 3, 1, 1), vis]
    image = torch.cat(tiles, -1)[0].float().clamp(0, 1).permute(1, 2, 0)
    Image.fromarray((image.cpu().numpy() * 255).round().astype(np.uint8)).save(path)


@torch.no_grad()
def fusion_diagnostics(output, ir, vis, target_mask=None, fusion_config=None, *, target_support=None):
    """Fixed IR-only evaluation masks, independent of the trained loss recipe.

    Return additive (sum of per-image means, valid-image count) statistics.
    These are source-consistency measurements, not fusion ground-truth scores.
    """
    seed = thermal_saliency(ir.float())
    background = 1 - seed
    fused, visible = output.fused.float(), vis.float()
    y, v = luminance(fused), luminance(visible)
    delta = y - v

    def masked(error, mask):
        mass = mask.flatten(1).sum(1)
        values = (error * mask).flatten(1).sum(1) / mass.clamp_min(1e-6)
        valid = mass > 1e-6
        return (values * valid).sum(), valid.sum().to(values)

    dx = (y[..., :, 1:] - y[..., :, :-1]) - (v[..., :, 1:] - v[..., :, :-1])
    dy = (y[..., 1:, :] - y[..., :-1, :]) - (v[..., 1:, :] - v[..., :-1, :])
    mx = torch.minimum(background[..., :, 1:], background[..., :, :-1])
    my = torch.minimum(background[..., 1:, :], background[..., :-1, :])
    gradients = torch.cat((dx.abs().flatten(1), dy.abs().flatten(1)), 1)
    masks = torch.cat((mx.flatten(1), my.flatten(1)), 1)

    def fractions(value):
        clipped = ((value <= 0) | (value >= 1)).any(1).float().flatten(1).mean(1)
        return clipped

    saturation = fractions(fused)
    saturation_delta = saturation - fractions(visible)
    result = {
        'background_luma_mae': masked(delta.abs(), background),
        'background_luma_bias': masked(delta, background),
        'background_gradient_mae': masked(gradients, masks),
        'thermal_roi_mae': masked((y - ir.float()).abs(), seed),
        'saturated_pixel_fraction': (saturation.sum(), saturation.new_tensor(saturation.numel())),
        'saturation_change_vs_vis': (saturation_delta.sum(), saturation.new_tensor(saturation.numel())),
    }
    zero = y.new_zeros(())
    teacher_keys = ('teacher_target_coverage', 'teacher_target_nonempty_fraction',
                    'teacher_gate_miss_fraction', 'teacher_applied_luma_mae', 'teacher_pre_gate_luma_mae',
                    'teacher_background_mae')
    result.update({name: (zero, zero) for name in teacher_keys})
    if target_mask is not None:
        target_mask = validate_target_mask(target_mask, ir)
        target_support = torch.ones_like(target_mask) if target_support is None else validate_target_mask(target_support, ir)
        desired = thermal_delta_target(ir, vis, fusion_config)
        actual = y-luminance(output.vis_reference)
        coverage = target_mask.flatten(1).mean(1)
        valid = target_mask.flatten(1).sum(1) > 1e-6
        supported = target_support.flatten(1).sum(1) > 1e-6
        result['teacher_target_coverage'] = ((coverage*supported).sum(), supported.sum().to(coverage))
        result['teacher_target_nonempty_fraction'] = ((valid & supported).float().sum(), supported.sum().to(coverage))
        result['teacher_gate_miss_fraction'] = masked((output.target_gate <= 1e-6).float(), target_mask)
        result['teacher_applied_luma_mae'] = masked((actual-target_mask*desired).abs(), target_mask)
        result['teacher_background_mae'] = masked(
            (fused-output.vis_reference.float()).abs().mean(1, keepdim=True), (1-target_mask)*target_support)
        if getattr(output, 'pre_gate_delta', None) is not None:
            result['teacher_pre_gate_luma_mae'] = masked((output.pre_gate_delta.float()-desired).abs(), target_mask)
    gate = getattr(output, 'target_gate', None)
    if gate is not None:
        for name in ('target_gate', 'person_mask', 'vehicle_mask', 'semantic_mask',
                     'motion_mask', 'ir_motion_mask', 'vis_motion_mask'):
            value = getattr(output, name, None)
            if value is not None:
                means = value.float().flatten(1).mean(1)
                result[name + '_coverage'] = (means.sum(), means.new_tensor(means.numel()))
        classes = getattr(output, 'class_masks', None)
        if classes is not None:
            from .models.person import TARGET_CLASSES
            if classes.shape != (fused.shape[0], len(TARGET_CLASSES), *fused.shape[-2:]):
                raise ValueError('Diagnostic class masks must be [B,6,H,W] in the configured target class order')
            for channel, name in enumerate(TARGET_CLASSES):
                means = classes[:, channel].float().flatten(1).mean(1)
                result['class_' + name + '_coverage'] = (means.sum(), means.new_tensor(means.numel()))
        motion_reports = (getattr(output, 'motion_diagnostics', None) or
                          (getattr(output, 'gate_diagnostics', None) or {}).get('motion', {}))
        for modality in ('ir', 'vis'):
            reports = motion_reports.get(modality, [])
            good = y.new_tensor([bool(row.get('camera_ok', False)) for row in reports])
            support = y.new_tensor([row.get('support_fraction', 0.) for row in reports])
            result[modality + '_camera_success_fraction'] = (good.sum(), good.new_tensor(len(reports)))
            result[modality + '_motion_support_fraction'] = (support.sum(), support.new_tensor(len(reports)))
        reference = output.vis_reference.float()
        outside = (gate <= 0).float()
        result['vis_reference_background_mae'] = masked((fused-reference).abs().mean(1, keepdim=True), outside)
        maxima = ((fused-reference).abs() * outside).flatten(1).amax(1)
        result['vis_reference_background_max_abs'] = (maxima.sum(), maxima.new_tensor(maxima.numel()))
        result['target_mean_brightening'] = masked(y - luminance(reference), gate.float())
        for name in ('person', 'vehicle'):
            region = getattr(output, name + '_mask', None)
            if region is not None:
                result[name + '_mean_brightening'] = masked(y - luminance(reference), region.float())
        target_edges = torch.cat((torch.minimum(gate[..., :, 1:], gate[..., :, :-1]).flatten(1),
                                  torch.minimum(gate[..., 1:, :], gate[..., :-1, :]).flatten(1)), 1)
        result['target_detail_error'] = masked(gradients, target_edges)
        semantic = getattr(output, 'semantic_mask', None)
        semantic = gate if semantic is None else semantic
        empty = semantic.float().flatten(1).amax(1) <= 1e-6
        result['semantic_gate_empty_fraction'] = (empty.float().sum(), y.new_tensor(empty.numel()))
        reports = (getattr(output, 'gate_diagnostics', None) or {}).get('pairs', [])
        required = y.new_tensor([item.get('alignment_required', False) for item in reports])
        rejected = y.new_tensor([not item.get('geometry_ok', False) for item in reports])
        valid = y.new_tensor([item.get('geometry_valid_fraction', 0.) for item in reports])
        result['alignment_required_fraction'] = (required.sum(), y.new_tensor(len(reports)))
        result['geometry_rejection_rate'] = ((required * rejected).sum(), required.sum())
        result['geometry_valid_fraction'] = ((required * valid).sum(), required.sum())
    return result



def _guidance_preview(path, output, ir, vis, reference):
    gate = getattr(output, 'target_gate', None)
    if gate is None:
        gate = torch.zeros_like(ir)
    person = getattr(output, 'person_mask', None)
    vehicle = getattr(output, 'vehicle_mask', None)
    person = torch.zeros_like(ir) if person is None else person
    vehicle = torch.zeros_like(ir) if vehicle is None else vehicle
    reference_vis = getattr(output, 'vis_reference', None)
    reference_vis = vis if reference_vis is None else reference_vis
    delta = (luminance(output.fused) - luminance(reference_vis)).clamp(-1, 1)
    heat = torch.cat((delta.clamp_min(0), torch.zeros_like(delta), (-delta).clamp_min(0)), 1)
    tiles = [output.fused, person.repeat(1, 3, 1, 1), vehicle.repeat(1, 3, 1, 1),
             gate.repeat(1, 3, 1, 1), heat]
    array = (torch.cat(tiles, -1)[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    h, w = array.shape[:2]
    canvas = Image.new('RGB', (w, h + 24), 'white')
    canvas.paste(Image.fromarray(array), (0, 24))
    draw = ImageDraw.Draw(canvas)
    for i, label in enumerate(('Fused', 'Person', 'Vehicle', 'Target gate', 'Applied delta Y')):
        draw.text((i * (w // 5) + 3, 5), label, fill='black')
    canvas.save(path)


def _empty_accumulator():
    return {'clips': 0, 'components': {}, '_statistics': None, '_fusion_diagnostics': None,
            'alignment_reason_counts': {},
            'missing': {'ir_count': 0, 'vis_count': 0, 'ir_psnr': 0., 'ir_ssim': 0.,
                        'vis_psnr': 0., 'vis_ssim': 0., 'vis_reference_count': 0,
                        'vis_reference_psnr': 0., 'vis_reference_ssim': 0.},
            'observed': {'ir_count': 0, 'vis_count': 0, 'ir_max_abs_error': 0., 'vis_max_abs_error': 0.}}


@torch.no_grad()
def _finalize(row, weights):
    statistics = row.pop('_statistics')
    row['components'] = {key: float(value) for key, value in normalize_statistics(statistics, weights).items()}
    # Training averages over scheduled queries/clips for exact DDP accumulation.
    # Reports separately expose evaluable support and never label empty support zero error.
    row['support'] = {}
    for name in ('target_support', 'target_detail_support', 'target_coverage', 'background_support', 'temporal_support',
                 'temporal_pair_support', 'temporal_coverage', 'temporal_visibility',
                 'temporal_lowfreq_support', 'temporal_lowfreq_pair_support',
                 'temporal_lowfreq_coverage', 'temporal_lowfreq_visibility'):
        if name in statistics:
            numerator, denominator = statistics[name]
            row['support'][name] = {'sum': float(numerator), 'count': int(denominator),
                                    'mean': float(numerator/denominator) if denominator > 0 else None}
    for name in ('reference_l1', 'reference_gradient', 'applied_luma', 'gate_miss'):
        if name in statistics:
            total, count = statistics[name]
            row['components'][name] = float(total/count) if count > 0 else None
    for support, terms in (('target_support', ('target_luma', 'applied_luma', 'gate_miss')),
                           ('target_detail_support', ('target_detail',)),
                           ('background_support', ('background',)),
                           ('temporal_support', ('temporal',)),
                           ('temporal_lowfreq_support', ('temporal_lowfreq',))):
        if support in statistics and statistics[support][0] == 0:
            for name in terms:
                if name in row['components']:
                    row['components'][name] = None
    for name, (_, count) in statistics.items():
        if name in row['components'] and count == 0:
            row['components'][name] = None
    diagnostics = row.pop('_fusion_diagnostics') or {}
    row['fusion_diagnostics'] = {name: {'mean': float(total / count) if count > 0 else None,
                                       'images': int(count)}
                                 for name, (total, count) in diagnostics.items()}
    for modality in ('ir', 'vis', 'vis_reference'):
        count = row['missing'][modality + '_count']
        for measure in ('psnr', 'ssim'):
            row['missing'][modality + '_' + measure] = (
                row['missing'][modality + '_' + measure] / count if count else None)
    return row


def selection_eligible(stage, support, weights):
    """An unavailable active loss cannot improve best by disappearing from total."""
    if stage == 'a':
        return True
    required = ['target_support']
    if stage in ('c', 'joint'):
        required.extend(name + '_support' for name in ('temporal', 'temporal_lowfreq')
                        if weights.get(name, 0.) > 0.)
    return all(support.get(name, {}).get('sum', 0.) > 0 for name in required)


@torch.no_grad()
def validate(model, dataset, stage, clips=150, seed=3407, device='cpu', reference=None,
             output_dir=None, fusion_config=None, progress_callback=None, *, sample_keys=None):
    """Every clip contributes equally within its source/protocol group.

    The lower-is-better score averages the represented source/protocol objectives. Image quality
    denominators count only missing images; absent metrics serialize as null.
    Explicit sample_keys support paired or exhaustive held-out evaluation; the
    default balanced training-validation sampler is unchanged.
    """
    if stage not in ('a', 'b', 'c', 'joint'):
        raise ValueError('Validation stage must be a, b, c or joint')
    if stage in ('c', 'joint') and reference is None:
        raise ValueError('Stage C validation requires its portable temporal reference')
    destination = Path(output_dir) if output_dir is not None else None
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
    groups, aggregate, audit = {}, _empty_accumulator(), []
    ablation = getattr(model, 'config', {}).get('ablation')
    loss = LossComputer(stage, fusion_config=fusion_config, ablation=ablation)
    weights = loss.validation_weights
    if stage in ('c', 'joint'):
        weights['temporal'] = loss.temporal_weight
        weights['temporal_lowfreq'] = loss.temporal_lowfreq_weight
    keys = validation_keys(dataset, clips, seed) if sample_keys is None else list(sample_keys)
    if not keys:
        raise ValueError('Evaluation requires at least one sample key')
    if any(not isinstance(key, SampleKey) or key.protocol not in ('same', 'vis_fast')
           or not 0 <= key.window_index < len(dataset) for key in keys):
        raise ValueError('Invalid explicit evaluation sample key')
    if progress_callback is not None:
        progress_callback(0, len(keys))
    group_count = 2 * len(dataset.groups)
    training = model.training
    try:
        with preserve_rng():
            model.eval()
            for index, key in enumerate(keys):
                sample = dataset[key]
                batch = collate_clips([sample])
                query = (index // group_count) % (2 if stage == 'a' else 3)
                selected = list(range(3)) if stage in ('c', 'joint') else [query * 2 if stage == 'a' else query]
                group_name = sample['source'] + '/' + sample['protocol']
                row = groups.setdefault(group_name, _empty_accumulator())
                terms_by_query, predictions, fused_predictions, masks, observation_flags = [], [], [], [], []
                gate_cache, teacher_cache = {}, {}
                for q in selected:
                    inputs = canonicalize_inputs(batch['inputs'][q].to(device))
                    ir = batch['ir_targets'][:, q].to(device)
                    vis = batch['vis_targets'][:, q].to(device)
                    cold_cache = not gate_cache
                    if torch.device(device).type == 'cuda':
                        torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    output = model(inputs, gate_cache=gate_cache)
                    if torch.device(device).type == 'cuda':
                        torch.cuda.synchronize(device)
                    elapsed_ms = 1000 * (time.perf_counter() - started)
                    teachers = batch.get('teacher_inputs')
                    supervision = (build_target_supervision(model, inputs, vis, reference,
                                    teacher_inputs=teachers[q].to(device) if teachers is not None else None,
                                    cache=teacher_cache) if stage != 'a' else None)
                    target_mask = supervision['mask'] if supervision is not None else None
                    target_support = supervision['support'] if supervision is not None else None
                    terms = loss(output, inputs, ir, vis, return_stats=True, target_mask=target_mask,
                                 target_support=target_support)
                    diagnostics = fusion_diagnostics(output, ir, vis, target_mask, loss.fusion_config,
                                                     target_support=target_support)
                    if supervision is not None:
                        terms.update(target_supervision_statistics(supervision))
                        diagnostics.update(target_supervision_statistics(supervision))
                    elapsed = ir.new_tensor(elapsed_ms, dtype=torch.float32)
                    one, zero = elapsed.new_ones(()), elapsed.new_zeros(())
                    diagnostics['model_forward_ms'] = (elapsed, one)
                    diagnostics['model_forward_cold_cache_ms'] = ((elapsed, one) if cold_cache else (zero, zero))
                    diagnostics['model_forward_reused_cache_ms'] = ((zero, zero) if cold_cache else (elapsed, one))
                    terms_by_query.append(terms)
                    observation_flags.append(output.observed)
                    if stage in ('c', 'joint'):
                        predictions.append(luminance(output.fused) - luminance(output.vis_reference))
                        fused_predictions.append(output.fused)
                        masks.append(target_mask)
                    missing = missing_metrics(output, ir, vis)
                    for accumulator in (row, aggregate):
                        for alignment in (getattr(output, 'gate_diagnostics', None) or {}).get('pairs', []):
                            reason = alignment.get('reason', 'unknown')
                            counts = accumulator['alignment_reason_counts']
                            counts[reason] = counts.get(reason, 0) + 1
                        previous = accumulator['_fusion_diagnostics']
                        accumulator['_fusion_diagnostics'] = (sum_statistics([previous, diagnostics])
                                                               if previous is not None else diagnostics)
                        for modality in ('ir', 'vis', 'vis_reference'):
                            count = missing[modality + '_count']
                            accumulator['missing'][modality + '_count'] += count
                            for measure in ('psnr', 'ssim'):
                                name = modality + '_' + measure
                                accumulator['missing'][name] += float(missing[name]) * count
                            if modality == 'vis_reference':
                                continue
                            observed = output.observed[:, 0 if modality == 'ir' else 1]
                            prediction = output.ir_aux if modality == 'ir' else output.vis_aux
                            target = ir if modality == 'ir' else vis
                            accumulator['observed'][modality + '_count'] += int(observed.sum())
                            if observed.any():
                                name = modality + '_max_abs_error'
                                accumulator['observed'][name] = max(accumulator['observed'][name],
                                                                    float((prediction[observed]-target[observed]).abs().max()))
                    if destination and index < group_count:
                        _preview(destination / f'{index:03}_{q+1}_{sample["protocol"]}.png', output, ir, vis)
                        _guidance_preview(destination / f'{index:03}_{q+1}_{sample["protocol"]}_diagnostics.png',
                                          output, ir, vis, None)
                statistics = sum_statistics(terms_by_query)
                if stage in ('c', 'joint'):
                    temporal = reference(torch.stack(predictions, 1), batch['ir_targets'].to(device),
                                         batch['vis_targets'].to(device), target_masks=torch.stack(masks, 1),
                                         observed_vis=torch.stack(observation_flags, 1)[:, :, 1],
                                         fused_predictions=torch.stack(fused_predictions, 1), return_statistics=True)
                    statistics.update(temporal)
                for accumulator in (row, aggregate):
                    accumulator['clips'] += 1
                    existing = accumulator['_statistics']
                    accumulator['_statistics'] = sum_statistics([existing, statistics]) if existing is not None else statistics
                audit.append({'source': sample['source'], 'protocol': sample['protocol'],
                              'sequence_id': sample['sequence_id'], 'window_index': key.window_index,
                              'sample_id': sample['sample_id'], 'seed': key.seed,
                              'query_positions': [q+1 for q in selected]})
                if progress_callback is not None:
                    progress_callback(index + 1, len(keys))
    finally:
        model.train(training)
    groups = {name: _finalize(row, weights) for name, row in sorted(groups.items())}
    report = _finalize(aggregate, weights)
    report.update(stage=stage, split=dataset.split, seed=seed, groups=groups,
                  score=sum(row['components']['total'] for row in groups.values()) / len(groups),
                  samples=audit, manifest_identity=dataset.manifest['fingerprint'],
                  diagnostics_reference='full_image_thermal_gt_v7',
                  segmentation_enabled=getattr(model, 'segmentation_enabled', True),
                  fusion_config=loss.fusion_config,
                  timing_protocol='batch=1 model forward only; CUDA synchronized; includes initial warmup; '
                                  'cold/reused cache reported separately; validation has no explicit autocast',
                  geometry_rate_denominator='queries requiring VIS alignment; exact VIS observations excluded')
    report['selection_eligible'] = selection_eligible(stage, report['support'], weights)
    report['support_note'] = ('Training losses average scheduled queries/clips; support reports count independent '
                              'full-image targets and visible temporal evidence. Null means unevaluable, not zero error. '
                              'No-target or entirely invisible validation cannot select a fusion checkpoint.')
    report['temporal_protocol'] = 'full_image_frozen_ir_delta_and_vis_lowfreq_teacher_v7'
    if ablation is not None:
        report['ablation'] = dict(ablation)
        report['temporal_protocol'] = (
            'a_reference_ir_flow_gt_vis_confidence_teacher_union_symmetric_residual'
            if ablation['temporal'] == 'standard' else
            'a_reference_ir_flow_gt_vis_confidence_teacher_union_observed_anchor')
    report = json_ready(report)
    if destination:
        write_json(destination / 'metrics.json', report)
    return report
