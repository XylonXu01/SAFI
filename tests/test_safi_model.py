"""Whole-image SAFI-Net and exact-observation invariants on the real task network."""
from dataclasses import fields, replace

import pytest
import torch

from safi_net.models import build_model, configure_stage, parameter_groups
from safi_net.temporal import canonicalize_inputs, select_inputs, time_context
from safi_net.types import MotionContext, TemporalInputs
from test_model import TestMotion, TestSemantic, inputs


torch.set_num_threads(2)


class FeatureOnlySemantic(TestSemantic):
    def __init__(self):
        super().__init__()
        self.feature_calls = 0

    def encode_features(self, images):
        self.feature_calls += 1
        return self(images)

    def encode_masks(self, images):
        raise AssertionError('SAFI-Net must not execute PMT')

    def encode_observation(self, images):
        raise AssertionError('SAFI-Net must use feature-only DINO')


def make_model():
    torch.manual_seed(19)
    net = build_model({'base_channels': 4, 'semantic_dim': 8,
                       'transport': {'radius': 1},
                       'target_gate': {'policy': 'full_image', 'segmentation_enabled': False}},
                      test_backbones=(TestMotion(), TestMotion(), FeatureOnlySemantic()))
    with torch.no_grad():
        net.decoder.fused.bias.fill_(3.)
    return net


def exact_in_slot(batch, slot):
    """Same observed image, arbitrary neighbour, legal chronological clocks."""
    changes = {}
    for name in ('ir', 'vis'):
        frames = getattr(batch, name)
        changes[name] = frames if slot == 0 else frames.flip(1)
        changes[name + '_times'] = torch.tensor([[1., 2.]] if slot == 0 else [[0., 1.]], dtype=torch.float64)
        changes[name + '_valid'] = torch.ones(1, 2, dtype=torch.bool)
    return replace(batch, **changes)


@pytest.mark.parametrize('slot', [0, 1])
def test_canonical_exact_is_private_idempotent_and_padded(slot):
    batch = exact_in_slot(inputs(True, True), slot)
    copies = {field.name: getattr(batch, field.name).clone() for field in fields(batch)
              if isinstance(getattr(batch, field.name), torch.Tensor)}
    out = canonicalize_inputs(batch)
    again = canonicalize_inputs(out)
    for name in ('ir', 'vis'):
        assert torch.equal(getattr(out, name)[:, 0], getattr(batch, name)[:, slot])
        assert not getattr(out, name)[:, 1].count_nonzero()
        assert torch.equal(getattr(out, name + '_valid'), torch.tensor([[True, False]]))
        assert torch.equal(getattr(out, name + '_times'), torch.tensor([[1., 0.]], dtype=torch.float64))
    for name, old in copies.items():
        assert torch.equal(getattr(batch, name), old)
        assert torch.equal(getattr(out, name), getattr(again, name))
    assert time_context(out)['effective_valid'].tolist() == [[True, True, False, False]]


def test_canonical_missing_pairs_keep_chronological_images_and_clocks():
    batch = inputs()
    out = canonicalize_inputs(batch)
    for name in ('ir', 'vis', 'ir_times', 'vis_times', 'ir_valid', 'vis_valid'):
        assert torch.equal(getattr(out, name), getattr(batch, name))


def test_canonical_mixed_batch_and_full_fov_context_agree():
    exact = exact_in_slot(inputs(True, True), 1)
    missing = inputs()
    batch = TemporalInputs.cat([exact, missing])
    batch.motion_context = MotionContext(batch.ir.clone(), batch.vis.clone(),
                                         torch.ones(2, 1, 32, 32), torch.eye(3).repeat(2, 1, 1),
                                         torch.ones(2))
    out = canonicalize_inputs(batch)
    assert torch.equal(out.ir, out.motion_context.ir)
    assert torch.equal(out.vis, out.motion_context.vis)
    assert torch.equal(out.ir[1], missing.ir[0])
    assert torch.equal(out.vis[0, 0], exact.vis[0, 1])
    assert torch.equal(out.motion_context.transform, batch.motion_context.transform)


def test_exact_slot_invariance_in_all_source_indexed_outputs():
    net = make_model().eval()
    batch = inputs(True, True)
    with torch.no_grad():
        a = net(exact_in_slot(batch, 0))
        b = net(exact_in_slot(batch, 1))
    for name in ('fused', 'vis_reference', 'ir_aux', 'vis_aux', 'pre_gate_delta',
                 'source_weights', 'reliability', 'visibility'):
        assert torch.equal(getattr(a, name), getattr(b, name)), name
    for key in a.flows:
        assert torch.equal(a.flows[key], b.flows[key])


@pytest.mark.parametrize('query', [0., 1., 2.])
def test_same_and_vis_fast_shared_observation_invariance_including_endpoints(query):
    net = make_model().eval()
    torch.manual_seed(71)
    ir, vis = torch.rand(3, 1, 32, 32), torch.rand(5, 3, 32, 32)
    same = select_inputs(ir, vis[::2], [0., 1., 2.], [0., 1., 2.], query)
    fast = select_inputs(ir, vis, [0., 1., 2.], [0., .5, 1., 1.5, 2.], query)
    with torch.no_grad():
        assert torch.equal(net(same).fused, net(fast).fused)


def test_single_observation_is_supported():
    net = make_model().eval()
    batch = inputs(True, True)
    selected = select_inputs(batch.ir[0, :1], batch.vis[0, :1], [1.], [1.], 1.)
    with torch.no_grad():
        output = net(selected)
    assert torch.equal(output.vis_reference, batch.vis[:, 0])
    assert output.target_gate.min() == 1


def test_full_image_output_without_any_segmentation_or_motion_gate():
    net = make_model()
    assert not hasattr(net, 'target_gate')
    output = net(inputs())
    assert net.dino.feature_calls > 0
    assert output.target_gate.min() == output.target_gate.max() == 1
    assert torch.equal(output.fused, output.vis_reference + output.pre_gate_delta)
    assert torch.equal(output.fusion_delta, output.pre_gate_delta)
    assert output.fusion_delta.count_nonzero() > 0
    assert output.pre_gate_delta.min() >= 0 and output.pre_gate_delta.max() <= .25
    assert output.fused.min() >= 0 and output.fused.max() <= 1
    for name in ('person_mask', 'vehicle_mask', 'semantic_mask', 'motion_mask',
                 'ir_motion_mask', 'vis_motion_mask', 'class_masks'):
        assert not getattr(output, name).count_nonzero()
    assert net.segmentation_enabled is False
    assert net.set_segmentation_enabled(False) is net
    with pytest.raises(ValueError, match='no segmentation head'):
        net.set_segmentation_enabled(True)


def test_joint_freezes_backbones_but_trains_fusion():
    net = make_model()
    configure_stage(net, 'joint')
    net.train()
    for module in (net.dino, net.ir_motion, net.vis_motion):
        assert not module.training
        assert not any(p.requires_grad for p in module.parameters())
    groups = parameter_groups(net, 'joint')
    assert len(groups) == 1 and groups[0]['lr'] == 1e-5
    out = net(inputs())
    out.fused.mean().backward()
    assert net.decoder.fused.bias.grad is not None
    assert all(p.grad is None for module in (net.dino, net.ir_motion, net.vis_motion)
               for p in module.parameters())


def test_builder_never_passes_a_pmt_checkpoint(monkeypatch):
    import safi_net.models as models
    calls = []
    monkeypatch.setattr(models, 'RIFEBackbone', lambda path: TestMotion())
    def semantic(path, *, person_checkpoint, person_config):
        calls.append(person_checkpoint)
        return FeatureOnlySemantic()
    monkeypatch.setattr(models, 'DINOv3Backbone', semantic)
    model = models.build_model({'base_channels': 4, 'semantic_dim': 8})
    assert calls == [None]
    assert model.architecture_id == 'safi_net'
