"""Validated experiment switches shared by model, objectives and launchers."""
from collections.abc import Mapping
import math

import torch


ABLATION_DEFAULTS = {
    'mode': 'full',
    'experiment_id': 'baseline',
    'semantic_features': True,
    'segmentation_gate': True,
    'aggregation': 'proposed',
    'temporal': 'proposed',
    'background_weight': 0.0,
    'cascade': False,
}


MODES = ('full', 'a1_no_dino', 'a2_separate_weights', 'a3_plain_decoder', 'a4_no_tcl')


def mode_of(config=None):
    value = config or {}
    if 'model' in value:
        value = value['model']
    value = value.get('ablation', value) if isinstance(value, Mapping) else value
    mode = value.get('mode', 'full') if isinstance(value, Mapping) else 'full'
    if mode not in MODES:
        raise ValueError(f'Unknown ablation mode: {mode}')
    return mode


def normalize_ablation(config=None):
    """Missing switches preserve the original model and training recipe."""
    if config is not None and not isinstance(config, Mapping):
        raise ValueError('model.ablation must be a mapping')
    config = {} if config is None else dict(config)
    unknown = set(config) - set(ABLATION_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown ablation settings: {sorted(unknown)}')
    result = {**ABLATION_DEFAULTS, **config}
    mode = mode_of(config)
    if mode != 'full':
        expected = {**ABLATION_DEFAULTS, 'mode': mode, 'semantic_features': mode != 'a1_no_dino'}
        if any(config[k] != expected[k] for k in config if k != 'mode'):
            raise ValueError('experiments/ablation modes are independent; legacy switches cannot be stacked')
        result = expected
    if not isinstance(result['experiment_id'], str) or not result['experiment_id'].strip():
        raise ValueError('ablation.experiment_id must be a nonempty string')
    for field in ('semantic_features', 'segmentation_gate', 'cascade'):
        if not isinstance(result[field], bool):
            raise ValueError(f'ablation.{field} must be boolean')
    for field in ('aggregation', 'temporal'):
        if result[field] not in ('proposed', 'standard'):
            raise ValueError(f'ablation.{field} must be proposed or standard')
    weight = result['background_weight']
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError('ablation.background_weight must be finite and nonnegative')
    result['background_weight'] = float(weight)
    return result


def masked_source_softmax(logits, valid):
    """Return zero, rather than NaN, where no source can sample a pixel."""
    valid = valid.to(dtype=torch.bool)
    masked = logits.float().masked_fill(~valid, -torch.inf)
    any_valid = valid.any(1, keepdim=True)
    safe = torch.where(any_valid, masked, torch.zeros_like(masked))
    return safe.softmax(1) * valid


def describe_model(model, objective=None):
    mode = mode_of(model.config)
    weights = dict(objective.weights) if objective is not None else {}
    if objective is not None:
        weights.update(temporal=objective.temporal_weight, temporal_lowfreq=objective.temporal_lowfreq_weight)
        weights.setdefault('match', 0.)
    return {'ablation_mode': mode, 'semantics_enabled': mode != 'a1_no_dino',
            'weight_mode': 'separate' if mode == 'a2_separate_weights' else 'shared',
            'decoder': 'plain_rgb' if mode == 'a3_plain_decoder' else 'TED',
            'loss_form': 'rgb_residual_channel_mean' if mode == 'a3_plain_decoder' else 'scalar_luminance',
            'loss_weights': weights, 'loss_enabled': {k:v != 0 for k,v in weights.items()},
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'added_weight_parameters': sum(p.numel() for n,p in model.named_parameters() if n.startswith('interpolation_weight_heads.'))}
