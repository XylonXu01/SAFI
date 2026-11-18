"""Full-image supervision and joint anti-flicker loss contracts."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from safi_net.losses import (LossComputer, TemporalReference, build_target_supervision,
                                   luminance, thermal_delta_target)
from safi_net.types import TemporalInputs


class ZeroMotion(nn.Module):
    def __init__(self, displacement=0.):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(float(displacement)))

    def forward(self, a, b, tau):
        return a.new_ones(len(a), 4, *a.shape[-2:]) * self.bias


def sequence(batch=1, height=16, width=16):
    ir = torch.full((batch, 3, 1, height, width), .6)
    vis = torch.full((batch, 3, 3, height, width), .2)
    reference = TemporalReference(ZeroMotion(), ZeroMotion())
    target = thermal_delta_target(ir.flatten(0, 1), vis.flatten(0, 1)).reshape_as(ir)
    return reference, ir, vis, target


def stats(reference, delta, ir, vis, fused, observed=None):
    return reference(delta, ir, vis, target_masks=torch.ones_like(ir), observed_vis=observed,
                     fused_predictions=fused, return_statistics=True)


def test_full_image_supervision_never_calls_region_teachers():
    class NoTeacher:
        segmentation_enabled = False
        def reference_target_masks(self, image):
            raise AssertionError('PMT is not part of v7')
    vis = torch.zeros(2, 3, 8, 8, requires_grad=True)
    out = build_target_supervision(NoTeacher(), None, vis, reference=object(), teacher_inputs=object())
    assert torch.equal(out['mask'], torch.ones(2, 1, 8, 8))
    assert torch.equal(out['support'], out['mask'])
    assert not out['mask'].requires_grad
    assert not out['segmentation_enabled']
    assert out['semantic_mask'].count_nonzero() == out['ir_motion_mask'].count_nonzero() == 0


def test_joint_temporal_defaults_and_invalid_mode():
    loss = LossComputer('joint')
    assert loss.temporal_weight == loss.temporal_lowfreq_weight == .1
    assert loss.fusion_config['mode'] == 'full_image'
    with pytest.raises(ValueError, match='full_image'):
        LossComputer('joint', {'mode': 'target_gated'})


def test_joint_supervises_hot_static_pixels_and_zero_lift_background():
    h = 8
    times = torch.tensor([[0., 2.]], dtype=torch.float64)
    valid = torch.ones(1, 2, dtype=torch.bool)
    inputs = TemporalInputs(torch.zeros(1, 2, 1, h, h), torch.full((1, 2, 3, h, h), .2),
                            times, times, valid, valid, torch.ones(1, dtype=torch.float64))
    vis = torch.full((1, 3, h, h), .2)
    ir = torch.zeros(1, 1, h, h)
    ir[..., :4] = .6
    delta = torch.full_like(ir, .1, requires_grad=True)
    fused = vis + delta
    out = SimpleNamespace(fused=fused, ir_aux=ir, vis_aux=vis, vis_reference=vis,
        pre_gate_delta=delta, target_gate=torch.ones_like(ir),
        flows={name: torch.zeros(1, 2, h, h) for name in ('ir0', 'vis0', 'ir1', 'vis1')},
        reliability=torch.full((1, 4, h, h), .5), visibility=torch.full((1, 4, h, h), .5),
        source_weights=torch.full((1, 5, h, h), .2), transport_details=None)
    result = LossComputer('joint')(out, inputs, ir, vis, target_mask=torch.zeros_like(ir),
                                  target_support=torch.zeros_like(ir), return_stats=True)
    result['target_luma'][0].backward()
    assert (delta.grad[..., :4] < 0).all()
    assert (delta.grad[..., 4:] > 0).all()
    assert result['target_coverage'][0] == result['target_support'][0] == 1


def test_alternating_delta_has_gradient_without_dimming_observed_anchor():
    reference, ir, vis, target = sequence()
    delta = target.clone()
    delta[:, 0] -= .05
    delta[:, 2] -= .05
    delta.requires_grad_()
    observed = torch.tensor([[False, True, False]])
    result = stats(reference, delta, ir, vis, vis + delta, observed)
    assert result['temporal'][0].item() == pytest.approx(.05)
    result['temporal'][0].backward()
    assert delta.grad[:, 1].count_nonzero() == 0
    assert (delta.grad[:, (0, 2)] < 0).all()
    assert all(not p.requires_grad and p.grad is None for p in reference.parameters())


def test_lowfreq_detects_reference_flicker_when_delta_is_constant():
    reference, ir, vis, target = sequence()
    fused = vis + target
    fused[:, 1] += .05
    fused.requires_grad_()
    result = stats(reference, target, ir, vis, fused)
    assert result['temporal'][0].abs() < 1e-6
    assert result['temporal_lowfreq'][0].item() == pytest.approx(.05)
    result['temporal_lowfreq'][0].backward()
    assert fused.grad[:, 1].gt(0).all()
    assert fused.grad[:, (0, 2)].lt(0).all()
    assert result['temporal_lowfreq_pair_support'][0] == 4
    assert result['temporal_lowfreq_support'][0] == 1


def test_true_thermal_light_and_motion_changes_are_not_flicker():
    reference, ir, vis, _ = sequence()
    for frame in range(3):
        ir[:, frame].fill_(.4 + frame * .15)
        vis[:, frame].fill_(.1 + frame * .05)
        vis[:, frame, :, 3:7, 2 + frame:6 + frame] += .1
    target = thermal_delta_target(ir.flatten(0, 1), vis.flatten(0, 1)).reshape_as(ir)
    result = stats(reference, target, ir, vis, vis + target)
    assert result['temporal'][0].abs() < 1e-6
    assert result['temporal_lowfreq'][0].abs() < 1e-6


def test_lowfreq_confidence_is_not_cancelled_by_its_denominator():
    reference, ir, vis, target = sequence()
    fused = vis + target
    fused[:, 1] += .05
    high = stats(reference, target, ir, vis, fused)['temporal_lowfreq'][0]
    vis[:, 1] += .3
    target = thermal_delta_target(ir.flatten(0, 1), vis.flatten(0, 1)).reshape_as(ir)
    fused = vis + target
    fused[:, 1] += .05
    low = stats(reference, target, ir, vis, fused)['temporal_lowfreq'][0]
    assert low.item() == pytest.approx(high.item() * torch.exp(torch.tensor(-3.)).item(), rel=1e-5)


def test_no_visible_geometry_has_zero_loss_support_and_finite_gradients():
    _, ir, vis, target = sequence(height=7, width=11)
    reference = TemporalReference(ZeroMotion(1000.), ZeroMotion(1000.))
    fused = (vis + target).requires_grad_()
    result = stats(reference, target, ir, vis, fused)
    assert result['temporal_support'][0] == result['temporal_lowfreq_support'][0] == 0
    assert result['temporal_lowfreq'][0] == 0
    result['temporal_lowfreq'][0].backward()
    assert torch.isfinite(fused.grad).all() and fused.grad.count_nonzero() == 0


def test_lowfreq_requires_matching_rgb_shape():
    reference, ir, vis, target = sequence()
    with pytest.raises(ValueError, match='RGB predictions'):
        stats(reference, target, ir, vis, target)
