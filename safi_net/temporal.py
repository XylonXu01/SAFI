"""Selection and relative clocks, independent of targets and model weights."""
import math
from dataclasses import replace

import torch

from .types import TemporalInputs


def canonicalize_inputs(inputs: TemporalInputs) -> TemporalInputs:
    """Give exact observations one slot identity, independently of neighbours.

    Bracketed queries retain chronological endpoints. At exact queries only
    the real query image is active: slot zero holds it and slot one is padded.
    The returned object is private; source tensors and clocks are not mutated.
    All source-indexed supervision must use this representation as well.
    """
    time_context(inputs)  # Validate the original chronological input contract.
    updates, selections = {}, {}
    for name in ('ir', 'vis'):
        frames, times, valid = (getattr(inputs, name + suffix) for suffix in ('', '_times', '_valid'))
        exact = valid & (times == inputs.query_time[:, None])
        seen = exact.any(1)
        index = exact.to(torch.long).argmax(1)
        chosen = frames[torch.arange(len(frames), device=frames.device), index]
        canonical = torch.stack((chosen, torch.zeros_like(chosen)), 1)
        updates[name] = torch.where(seen[:, None, None, None, None], canonical, frames)
        clocks = torch.stack((inputs.query_time, torch.zeros_like(inputs.query_time)), 1)
        updates[name + '_times'] = torch.where(seen[:, None], clocks, times)
        mask = torch.stack((torch.ones_like(seen), torch.zeros_like(seen)), 1)
        updates[name + '_valid'] = torch.where(seen[:, None], mask, valid)
        selections[name] = (seen, index)
    if inputs.motion_context is not None:
        # Retain geometry for callers that still attach observation context;
        # it must not silently disagree with the normalized image slots.
        context_updates = {}
        for name, (seen, index) in selections.items():
            frames = getattr(inputs.motion_context, name)
            chosen = frames[torch.arange(len(frames), device=frames.device), index]
            canonical = torch.stack((chosen, torch.zeros_like(chosen)), 1)
            context_updates[name] = torch.where(seen[:, None, None, None, None], canonical, frames)
        updates['motion_context'] = replace(inputs.motion_context, **context_updates)
    return replace(inputs, **updates)


def _clock(times, device=None):
    times = torch.as_tensor(times, dtype=torch.float64, device=device)
    if times.ndim != 1 or not times.numel() or not torch.isfinite(times).all():
        raise ValueError('Clock must be a nonempty finite vector')
    if (times[1:] <= times[:-1]).any():
        raise ValueError('Clock must be strictly increasing')
    return times


def observation_indices(times, query_time):
    """Select chronological real context, keeping a neighbour at exact queries."""
    times = _clock(times)
    query = torch.as_tensor(query_time, dtype=torch.float64, device=times.device)
    if query.ndim or not torch.isfinite(query):
        raise ValueError('Query must be a finite scalar')
    if query < times[0] or query > times[-1]:
        raise ValueError('Query has no observation brackets')
    right = int(torch.searchsorted(times, query))
    if times[right] != query:
        return [right - 1, right]
    if len(times) == 1:
        return [right, None]
    if right == 0:
        return [0, 1]
    if right == len(times) - 1:
        return [right - 1, right]
    # Strictly closer future context wins; equal distances retain the past.
    if times[right + 1] - query < query - times[right - 1]:
        return [right, right + 1]
    return [right - 1, right]


def select_observations(frames, times, query_time):
    if frames.ndim != 4 or not frames.shape[0]:
        raise ValueError('Frames must have shape [N,C,H,W]')
    times = _clock(times, frames.device)
    if times.numel() != len(frames):
        raise ValueError('Frame and clock lengths differ')
    indices = observation_indices(times, query_time)
    if indices[1] is None:
        return (torch.stack([frames[indices[0]], torch.zeros_like(frames[indices[0]])]),
                torch.stack([times[indices[0]], times.new_zeros(())]),
                torch.tensor([True, False], device=frames.device))
    return (frames[indices], times[indices],
            torch.ones(2, dtype=torch.bool, device=frames.device))


def select_inputs(ir, vis, ir_times, vis_times, query_time):
    a, at, av = select_observations(ir, ir_times, query_time)
    b, bt, bv = select_observations(vis, vis_times, query_time)
    x = TemporalInputs(a[None], b[None], at[None], bt[None], av[None], bv[None],
                       torch.as_tensor([query_time], dtype=torch.float64, device=ir.device))
    time_context(x)
    return x


def time_context(inputs: TemporalInputs):
    x = inputs
    if x.ir.ndim != 5 or x.vis.ndim != 5:
        raise ValueError('Images must be [B,2,C,H,W]')
    batch = x.ir.shape[0]
    if batch < 1 or x.ir.shape[1:3] != (2, 1) or x.vis.shape[:3] != (batch, 2, 3):
        raise ValueError('Expected two IR grayscale and two VIS RGB slots')
    if x.ir.shape[-2:] != x.vis.shape[-2:] or min(x.ir.shape[-2:]) < 1:
        raise ValueError('IR and VIS spatial dimensions must agree')
    if not x.ir.is_floating_point() or not x.vis.is_floating_point():
        raise ValueError('Image tensors must be floating point')
    tensors = (x.ir, x.vis, x.ir_times, x.vis_times, x.ir_valid, x.vis_valid, x.query_time)
    if any(t.device != x.ir.device for t in tensors):
        raise ValueError('All inputs must share one device')
    if x.query_time.shape != (batch,) or x.query_time.dtype != torch.float64:
        raise ValueError('Query times must be float64 [B]')
    if not torch.isfinite(x.query_time).all():
        raise ValueError('Query times must be finite')
    active, observed, spans, taus, offsets = [], [], [], [], []
    for times, valid in ((x.ir_times, x.ir_valid), (x.vis_times, x.vis_valid)):
        if times.shape != (batch, 2) or times.dtype != torch.float64:
            raise ValueError('Observation times must be float64 [B,2]')
        if valid.shape != (batch, 2) or valid.dtype != torch.bool:
            raise ValueError('Observation masks must be bool [B,2]')
        if not torch.isfinite(times).all() or not valid.any(1).all():
            raise ValueError('Each modality requires finite times and a valid observation')
        pair = valid.all(1)
        if (pair & (times[:, 1] <= times[:, 0])).any():
            raise ValueError('Valid observation timestamps must be strictly increasing')
        exact = valid & (times == x.query_time[:, None])
        seen = exact.any(1)
        bracket = pair & (times[:, 0] < x.query_time) & (x.query_time < times[:, 1])
        if not (seen | bracket).all():
            raise ValueError('Query has no observation brackets')
        mask = torch.where(seen[:, None], exact, valid)
        span = torch.where(seen, 0., times[:, 1] - times[:, 0])
        tau = torch.where(seen, 0., (x.query_time - times[:, 0]) / span.clamp_min(torch.finfo(times.dtype).tiny))
        active.append(mask)
        observed.append(seen)
        spans.append(span)
        taus.append(tau)
        offsets.append(torch.where(mask, times - x.query_time[:, None], 0.))
    span = torch.stack(spans, dim=1)
    scale = span.max(1).values
    scale = torch.where(scale > 0, scale, 1.)
    # Stack along slots then modality to retain ir0,vis0,ir1,vis1 ordering.
    return {'observed': torch.stack(observed, 1), 'tau': torch.stack(taus, 1),
            'effective_valid': torch.stack(active, 2).flatten(1),
            'offsets': torch.stack(offsets, 2).flatten(1) / scale[:, None],
            'spans': span / scale[:, None]}


def output_timeline(ir_times, vis_times, fps_ir, fps_vis):
    if not all(math.isfinite(f) and f > 0 for f in (fps_ir, fps_vis)):
        raise ValueError('Frame rates must be positive finite values')
    ratio = fps_vis / fps_ir
    if not (math.isclose(ratio, 1., rel_tol=1e-8) or math.isclose(ratio, 2., rel_tol=1e-8)):
        raise ValueError('Only same rate or VIS2x is supported')
    ir, vis = _clock(ir_times), _clock(vis_times)
    low, high = max(ir[0], vis[0]), min(ir[-1], vis[-1])
    if low > high:
        raise ValueError('Empty common temporal support')
    if math.isclose(ratio, 2., rel_tol=1e-8):
        timeline = vis
    else:
        timeline = torch.cat([ir, ir[:-1] + (ir[1:] - ir[:-1]) / 2]).sort().values
    return timeline[(timeline >= low) & (timeline <= high)]
