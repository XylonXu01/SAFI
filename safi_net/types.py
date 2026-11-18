from dataclasses import dataclass, fields
from typing import Any, Sequence

import torch

SOURCE_KEYS = ('ir0', 'vis0', 'ir1', 'vis1')


@dataclass
class MotionContext:
    """Observation-only full-FOV images and query-pixel-to-context geometry."""
    ir: torch.Tensor
    vis: torch.Tensor
    spatial_valid: torch.Tensor
    transform: torch.Tensor
    frame_step: torch.Tensor

    def to(self, device, non_blocking=False):
        return MotionContext(**{f.name: getattr(self, f.name).to(device, non_blocking=non_blocking)
                                for f in fields(self)})

    def pin_memory(self):
        """Let DataLoader's pinning thread reach tensors in this custom batch."""
        return MotionContext(**{f.name: getattr(self, f.name).pin_memory() for f in fields(self)})

    def select(self, index):
        if isinstance(index, int):
            index = slice(index, index + 1)
        return MotionContext(**{f.name: getattr(self, f.name)[index] for f in fields(self)})

    @classmethod
    def cat(cls, items):
        if not items:
            raise ValueError('Cannot concatenate empty motion contexts')
        return cls(**{f.name: torch.cat([getattr(x, f.name) for x in items]) for f in fields(cls)})


@dataclass
class TemporalInputs:
    ir: torch.Tensor
    vis: torch.Tensor
    ir_times: torch.Tensor
    vis_times: torch.Tensor
    ir_valid: torch.Tensor
    vis_valid: torch.Tensor
    query_time: torch.Tensor
    motion_context: MotionContext | None = None

    def to(self, device: torch.device | str, non_blocking=False) -> 'TemporalInputs':
        return TemporalInputs(**{f.name: (getattr(self, f.name).to(device, non_blocking=non_blocking)
                                         if getattr(self, f.name) is not None else None)
                                 for f in fields(self)})

    def pin_memory(self) -> 'TemporalInputs':
        return TemporalInputs(**{f.name: (getattr(self, f.name).pin_memory()
                                         if getattr(self, f.name) is not None else None)
                                 for f in fields(self)})

    def select(self, index) -> 'TemporalInputs':
        if isinstance(index, int):
            index = slice(index, index + 1)
        return TemporalInputs(**{
            f.name: (None if getattr(self, f.name) is None else
                     getattr(self, f.name).select(index) if f.name == 'motion_context' else
                     getattr(self, f.name)[index]) for f in fields(self)})

    @classmethod
    def cat(cls, items: Sequence['TemporalInputs']) -> 'TemporalInputs':
        if not items:
            raise ValueError('Cannot concatenate empty TemporalInputs')
        values = {}
        for f in fields(cls):
            parts = [getattr(item, f.name) for item in items]
            if f.name == 'motion_context':
                if all(x is None for x in parts):
                    values[f.name] = None
                elif any(x is None for x in parts):
                    raise ValueError('Cannot mix inputs with and without full-FOV context')
                else:
                    values[f.name] = MotionContext.cat(parts)
            else:
                values[f.name] = torch.cat(parts)
        return cls(**values)


@dataclass
class AsyncOutput:
    fused: torch.Tensor
    ir_aux: torch.Tensor
    vis_aux: torch.Tensor
    ir_base: torch.Tensor
    vis_base: torch.Tensor
    flows: dict[str, torch.Tensor]
    reliability: torch.Tensor
    visibility: torch.Tensor
    source_weights: torch.Tensor
    observed: torch.Tensor
    transport_details: Any = None
    vis_reference: torch.Tensor | None = None
    person_mask: torch.Tensor | None = None
    motion_mask: torch.Tensor | None = None
    target_gate: torch.Tensor | None = None
    fusion_delta: torch.Tensor | None = None
    gate_diagnostics: Any = None
    vehicle_mask: torch.Tensor | None = None
    semantic_mask: torch.Tensor | None = None
    class_masks: torch.Tensor | None = None
    pre_gate_delta: torch.Tensor | None = None
    ir_motion_mask: torch.Tensor | None = None
    vis_motion_mask: torch.Tensor | None = None
    motion_diagnostics: Any = None
    fusion_representation: str = 'thermal_delta'
    interpolation_weights: tuple = ()
