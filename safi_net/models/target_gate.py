"""Semantic-or-motion gating; disabling segmentation retains independent motion."""
from collections.abc import Mapping
from copy import deepcopy

import torch
from torch import nn
import torch.nn.functional as F

from ..temporal import time_context
from .person import TARGET_CLASSES
from .motion_gate import MotionEvidence, motion_configuration


DEFAULT_GATE_CONFIG = {'policy': 'semantic_or_motion', 'segmentation_enabled': True, 'motion': {}}


def _configuration(config):
    if config is not None and not isinstance(config, Mapping):
        raise ValueError('Target gate configuration must be a mapping')
    config = {} if config is None else dict(config)
    unknown = set(config) - set(DEFAULT_GATE_CONFIG)
    if unknown:
        raise ValueError(f'Unknown target gate configuration: {sorted(unknown)}')
    result = deepcopy({**DEFAULT_GATE_CONFIG, **config})
    if result['policy'] != 'semantic_or_motion':
        raise ValueError('v6 target gate policy must be semantic_or_motion; use the explicit v5 migration utility')
    if not isinstance(result['segmentation_enabled'], bool):
        raise ValueError('target_gate.segmentation_enabled must be a boolean')
    result['motion'] = motion_configuration(result['motion'])
    return result


def inward_feather(mask):
    """Smooth only inward: no positive value may appear outside the input mask."""
    return torch.minimum(mask, F.avg_pool2d(mask, 3, stride=1, padding=1))


class TargetGate(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = _configuration(config)
        self.motion = MotionEvidence(self.config['motion'])

    @property
    def segmentation_enabled(self):
        return self.config['segmentation_enabled']

    def set_segmentation_enabled(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError('segmentation_enabled must be a boolean')
        self.config['segmentation_enabled'] = enabled

    @torch.no_grad()
    def forward(self, inputs, class_masks, ir_motion, vis_motion, cache=None, *, context=None):
        batch, _, _, height, width = inputs.vis.shape
        if (class_masks.shape != (batch, len(TARGET_CLASSES), height, width)
                or class_masks.device != inputs.vis.device):
            raise ValueError('Reference semantic masks must be finite [B,6,H,W] probabilities on the input device')
        if not (torch.isfinite(class_masks).all() & (class_masks >= 0).all() & (class_masks <= 1).all()):
            raise ValueError('Reference semantic masks must be finite [B,6,H,W] probabilities on the input device')
        with torch.autocast(device_type=inputs.vis.device.type, enabled=False):
            classes = class_masks.detach().float() if self.segmentation_enabled else torch.zeros_like(class_masks).float()
            people = classes[:, :1]
            vehicles = classes[:, 1:].amax(1, keepdim=True)
            semantic = classes.amax(1, keepdim=True)
            context = time_context(inputs) if context is None else context
            observed = context['observed'][:, 1].cpu().tolist()
            reports = [{'mask_source': 'observed_vis' if bool(seen) else 'interpolated_vis_reference',
                        'geometry_checked': False, 'alignment_required': False,
                        'reason': 'reference_semantics' if self.segmentation_enabled else 'segmentation_disabled'}
                       for seen in observed]
            motion = self.motion(inputs, ir_motion, vis_motion, cache=cache)
            diagnostics = {'pairs': reports, 'motion': motion['diagnostics'],
                           'segmentation_enabled': self.segmentation_enabled}
            return {'class_masks': classes, 'person_mask': people, 'vehicle_mask': vehicles,
                    'semantic_mask': semantic, **motion,
                    'motion_diagnostics': motion['diagnostics'],
                    'target_gate': inward_feather(torch.maximum(semantic, motion['motion_mask'])),
                    'diagnostics': diagnostics}
