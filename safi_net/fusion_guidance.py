"""Detached, target-only guidance for preserving visible backgrounds during fusion.

None of these references is an inference input. DINO supplies local similarity,
not a semantic class prediction or a photometric quality score.
"""
import math
from collections.abc import Mapping
from numbers import Integral, Real

import torch
import torch.nn.functional as F


DEFAULT_FUSION_CONFIG = {
    'mode': 'dino_ir',
    'seed_margin': 0.02,
    'median_margin': 0.05,
    'seed_scale_floor': 0.05,
    'affinity_temperature': 0.1,
    'propagation_steps': 2,
    'propagation_decay': 0.8,
    'background_luma_weight': 1.0,
    'background_detail_weight': 0.25,
    'interp_vis_weight': 3.0,
}


def normalize_fusion_config(config=None):
    """Return a validated, complete configuration; None selects the v2 defaults."""
    if config is not None and not isinstance(config, Mapping):
        raise ValueError('Fusion configuration must be a mapping')
    config = {} if config is None else dict(config)
    unknown = set(config) - set(DEFAULT_FUSION_CONFIG)
    if unknown:
        raise ValueError(f'Unknown fusion configuration keys: {sorted(unknown)}')
    result = {**DEFAULT_FUSION_CONFIG, **config}
    if result['mode'] not in ('dino_ir', 'ir_only', 'legacy'):
        raise ValueError('Fusion mode must be dino_ir, ir_only or legacy')
    for key in DEFAULT_FUSION_CONFIG:
        if key == 'mode':
            continue
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f'Fusion {key} must be a finite number')
        if key == 'propagation_steps':
            if not isinstance(value, Integral) or value < 0:
                raise ValueError('Fusion propagation_steps must be a nonnegative integer')
            result[key] = int(value)
        else:
            result[key] = float(value)
            if key in ('seed_scale_floor', 'affinity_temperature'):
                if value <= 0:
                    raise ValueError(f'Fusion {key} must be positive')
            elif key in ('seed_margin', 'median_margin', 'propagation_decay'):
                if not 0 <= value <= 1:
                    raise ValueError(f'Fusion {key} must be between 0 and 1')
            elif value < 0:
                raise ValueError(f'Fusion {key} must be nonnegative')
    if result['mode'] == 'legacy':
        result.update(background_luma_weight=0.0, background_detail_weight=0.0,
                      interp_vis_weight=1.0)
    return result


@torch.no_grad()
def thermal_saliency(ir_gt, config=None):
    """Soft thermal evidence; flat images have no confidently hot pixels."""
    config = normalize_fusion_config(config)
    if ir_gt.ndim != 4 or ir_gt.shape[1] != 1 or min(ir_gt.shape) < 1:
        raise ValueError('IR targets must have shape [B,1,H,W] with nonempty dimensions')
    ir = ir_gt.float()
    quantiles = torch.quantile(ir.flatten(1), ir.new_tensor([0.5, 0.8, 0.99]), dim=1)
    median, high, peak = (q[:, None, None, None] for q in quantiles)
    if config['mode'] == 'legacy':
        return (ir >= high).float()
    scale = (peak - high).clamp_min(config['seed_scale_floor'])
    seed = ((ir - high - config['seed_margin']) / scale).clamp(0, 1)
    return seed * ((ir - median) > config['median_margin'])


def _local_affinity(features, temperature):
    """Nine-neighbor affinities without materializing nine full feature tensors."""
    features = F.normalize(features.float(), dim=1, eps=1e-6)
    height, width = features.shape[-2:]
    padded = F.pad(features, (1, 1, 1, 1))
    valid = F.pad(torch.ones_like(features[:, :1]), (1, 1, 1, 1))
    weights = []
    for row in range(3):
        for column in range(3):
            neighbor = padded[..., row:row + height, column:column + width]
            similarity = (features * neighbor).sum(1, keepdim=True).clamp(-1, 1)
            present = valid[..., row:row + height, column:column + width]
            weights.append(((similarity - 1) / temperature).exp() * present)
    weights = torch.cat(weights, 1)
    return weights / weights.sum(1, keepdim=True).clamp_min(1e-12)


@torch.no_grad()
def build_fusion_reference(ir_gt, vis_gt, dino, fusion_config):
    """Return detached supervision masks; only dino_ir invokes the backbone."""
    config = normalize_fusion_config(fusion_config)
    if config['mode'] == 'legacy':
        return None
    if (vis_gt.ndim != 4 or vis_gt.shape[1] != 3
            or vis_gt.shape[0] != ir_gt.shape[0]
            or vis_gt.shape[-2:] != ir_gt.shape[-2:]
            or vis_gt.device != ir_gt.device):
        raise ValueError('VIS targets must have shape [B,3,H,W] matching IR targets')
    seed = thermal_saliency(ir_gt, config)
    protected = seed
    if config['mode'] == 'dino_ir':
        if dino is None:
            raise ValueError('dino_ir guidance requires a frozen DINO backbone')
        levels = dino(vis_gt)
        if not isinstance(levels, (tuple, list)) or not levels:
            raise ValueError('DINO must return a nonempty sequence of feature maps')
        features = levels[-1]
        if (features.ndim != 4 or features.shape[0] != ir_gt.shape[0]
                or features.device != ir_gt.device or min(features.shape) < 1):
            raise ValueError('DINO features must have shape [B,C,h,w] on the target device')
        initial = F.adaptive_max_pool2d(seed, features.shape[-2:])
        propagated = initial
        weights = _local_affinity(features, config['affinity_temperature'])
        for _ in range(config['propagation_steps']):
            neighbors = F.unfold(propagated, kernel_size=3, padding=1).reshape_as(weights)
            value = (weights * neighbors).sum(1, keepdim=True)
            propagated = torch.maximum(initial, config['propagation_decay'] * value)
        region = F.interpolate(propagated, size=seed.shape[-2:], mode='bilinear',
                               align_corners=False)
        protected = torch.maximum(seed, region)
    return {'ir_saliency': seed.detach(),
            'background_mask': (1 - protected).clamp(0, 1).detach()}
