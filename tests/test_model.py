"""Contract tests use explicit small backbones; the task network remains real."""
from dataclasses import replace
import importlib.util
import inspect

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from safi_net.types import TemporalInputs


class TestMotion(nn.Module):
    __test__ = False

    def __init__(self, value=0.05):
        super().__init__()
        self.shift = nn.Parameter(torch.tensor(value))
        self.calls = []

    def forward(self, a, b, tau):
        self.calls.append((a.detach().clone(), b.detach().clone(), tau.detach().clone()))
        fields = torch.stack((tau, tau * 0, -(1 - tau), tau * 0), 1)
        return fields[:, :, None, None].expand(-1, -1, *a.shape[-2:]) * self.shift

    def interpolate_with_flow(self, a, b, tau):
        flow = self(a, b, tau)
        blend = tau[:, None, None, None]
        return flow, a * (1-blend) + b * blend


class TestSemantic(nn.Module):
    __test__ = False
    output_dim = 8

    def __init__(self):
        super().__init__()
        self.project = nn.Conv2d(3, 8, 1)
        self.seen = []

    def forward(self, x):
        self.seen.append(x.detach().clone())
        value = self.project(F.adaptive_avg_pool2d(x, (2, 2)))
        return (value,) * 4


def model():
    from safi_net.models import build_model
    torch.manual_seed(17)
    return build_model({'base_channels': 4, 'semantic_dim': 8, 'transport': {'radius': 1}},
                       test_backbones=(TestMotion(0.1), TestMotion(-0.2), TestSemantic()))


def inputs(exact_ir=False, exact_vis=False):
    torch.manual_seed(41)
    ir, vis = torch.rand(1, 2, 1, 32, 32), torch.rand(1, 2, 3, 32, 32)
    return TemporalInputs(ir, vis,
                          torch.tensor([[1., 4.]], dtype=torch.float64) if exact_ir else torch.tensor([[0., 4.]], dtype=torch.float64),
                          torch.tensor([[1., 2.]], dtype=torch.float64) if exact_vis else torch.tensor([[0., 2.]], dtype=torch.float64),
                          torch.tensor([[True, not exact_ir]]),
                          torch.tensor([[True, not exact_vis]]), torch.tensor([1.], dtype=torch.float64))


def test_model_api_exists():
    assert importlib.util.find_spec('safi_net.models') is not None, 'model implementation missing'


@pytest.mark.parametrize('ir_exact,vis_exact', [(True, True), (True, False), (False, True)])
def test_observed_aux_identity_survives_nonzero_heads(ir_exact, vis_exact):
    net, batch = model(), inputs(ir_exact, vis_exact)
    with torch.no_grad():
        net.decoder.ir.weight.fill_(0.1)
        net.decoder.vis.weight.fill_(0.1)
        net.decoder.ir.bias.fill_(0.3)
        net.decoder.vis.bias.fill_(0.3)
    out = net(batch)
    if ir_exact:
        assert torch.equal(out.ir_aux, batch.ir[:, 0])
        assert torch.equal(out.ir_base, batch.ir[:, 0])
    if vis_exact:
        assert torch.equal(out.vis_aux, batch.vis[:, 0])
        assert torch.equal(out.vis_base, batch.vis[:, 0])


def test_independent_clocks_and_motion_inputs():
    net, batch = model(), inputs()
    out = net(batch)
    assert torch.equal(net.ir_motion.calls[0][0], batch.ir[:, 0].repeat(1, 3, 1, 1))
    assert torch.equal(net.vis_motion.calls[0][0], batch.vis[:, 0])
    assert torch.equal(net.ir_motion.calls[0][2], torch.tensor([0.25]))
    assert torch.equal(net.vis_motion.calls[0][2], torch.tensor([0.5]))
    assert not torch.equal(out.flows['ir0'], out.flows['vis0'])


def test_exact_vis_bypasses_pair_semantics_and_invalid_evidence():
    net, batch = model(), inputs(True, True)
    out = net(batch)
    assert not net.vis_motion.calls
    assert len(net.dino.seen) == 1
    assert torch.equal(net.dino.seen[0], batch.vis[:, 0])
    other = replace(batch, ir=batch.ir.clone(), vis=batch.vis.clone())
    other.ir[:, 1].fill_(1e4)
    other.vis[:, 1].fill_(-1e4)
    again = net(other)
    assert torch.equal(out.fused, again.fused)
    assert torch.count_nonzero(out.source_weights[:, [2, 3]]) == 0
    assert torch.count_nonzero(out.visibility[:, [2, 3]]) == 0
    assert out.transport_details is None


def test_fast_vis_real_observation_changes_output_and_time_affine_invariance():
    net, batch = model(), inputs(False, True)
    out = net(batch)
    shifted = replace(batch, ir_times=batch.ir_times.double()*7+31,
                      vis_times=batch.vis_times.double()*7+31, query_time=batch.query_time*7+31)
    assert torch.allclose(out.fused, net(shifted).fused, atol=1e-5, rtol=0)
    changed = replace(batch, vis=batch.vis.clone())
    changed.vis[:, 0] = 1 - changed.vis[:, 0]
    assert not torch.allclose(out.fused, net(changed).fused)
    assert list(inspect.signature(net.forward).parameters) == ['inputs', 'gate_cache']


def test_stage_groups_and_missing_ir_gradient():
    from safi_net.models import configure_stage, parameter_groups
    from safi_net.losses import LossComputer
    net, batch = model(), inputs(False, True)
    for stage, lrs in [('a', [1e-4, 1e-5, 1e-5]), ('b', [1e-5]), ('c', [5e-6])]:
        configure_stage(net, stage)
        groups = parameter_groups(net, stage)
        assert [g['lr'] for g in groups] == lrs
        assert len({id(p) for g in groups for p in g['params']}) == sum(len(g['params']) for g in groups)
        assert not any(p.requires_grad for p in net.dino.parameters())
        assert any(p.requires_grad for p in net.vis_motion.parameters()) == (stage == 'a')
        assert net.ir_motion.shift.requires_grad == (stage == 'a')
    configure_stage(net, 'a')
    loss, _ = LossComputer('a')(net(batch), batch, torch.rand_like(batch.ir[:, 0]), batch.vis[:, 0])
    loss.backward()
    assert net.ir_motion.shift.grad is not None and net.ir_motion.shift.grad.abs() > 0
    assert all(p.grad is None for p in net.dino.parameters())
    assert all(p.grad is None for p in net.vis_motion.parameters())


def test_exact_visible_chroma_is_used_after_nonzero_aux_head():
    net, batch = model(), inputs(False, True)
    batch.vis = batch.vis*0.4+0.3
    with torch.no_grad():
        net.decoder.vis.bias.copy_(torch.tensor([0.3, -0.4, 0.7]))
        net.decoder.fused.bias.fill_(0.1)
    out = net(batch)
    delta = out.fused-batch.vis[:, 0]
    assert torch.allclose(delta[:, 0], delta[:, 1], atol=1e-6)
    assert torch.allclose(delta[:, 1], delta[:, 2], atol=1e-6)


def test_exact_second_slot_and_mixed_batch_have_no_neighbor_evidence():
    net = model()
    exact = inputs(True, True)
    exact = replace(exact, ir=exact.ir.flip(1), vis=exact.vis.flip(1),
                    ir_times=exact.ir_times.flip(1), vis_times=exact.vis_times.flip(1),
                    ir_valid=exact.ir_valid.flip(1), vis_valid=exact.vis_valid.flip(1))
    batch = TemporalInputs.cat([exact, inputs()])
    out = net(batch)
    assert torch.equal(out.ir_aux[:1], exact.ir[:, 1])
    assert torch.equal(out.vis_aux[:1], exact.vis[:, 1])
    assert torch.count_nonzero(out.source_weights[:1, 2:4]) == 0
    assert out.transport_details['batch_indices'].tolist() == [1]


def test_hidden_targets_only_change_loss_and_no_inference_state():
    from safi_net.losses import LossComputer
    net, batch = model(), inputs()
    before = net(batch)
    loss = LossComputer('b')
    a, _ = loss(before, batch, torch.zeros_like(before.ir_aux), torch.zeros_like(before.vis_aux),
                target_mask=torch.ones_like(before.ir_aux))
    b, _ = loss(before, batch, torch.ones_like(before.ir_aux), torch.ones_like(before.vis_aux),
                target_mask=torch.ones_like(before.ir_aux))
    assert not torch.equal(a, b)
    assert torch.equal(before.fused, net(batch).fused)
