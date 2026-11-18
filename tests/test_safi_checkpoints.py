"""SAFI-Net migration preserves shared weights, rejects incompatible recipes, resets state."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from safi_net import checkpoints as ckpt
from safi_net import _legacy_v6_checkpoints as legacy


def migration_case():
    target = yaml.safe_load((Path(__file__).parents[1] / 'configs/base.yaml').read_text())
    target['asset_fingerprints'] = {'rife/flownet.pkl': 'rife-sha', 'dinov3/model.safetensors': 'dino-sha'}
    source_config = deepcopy(target)
    source_config['architecture'] = legacy.ARCHITECTURE
    source_config['model'].pop('fusion_region')
    source_config['model']['target_gate'] = {'policy': 'semantic_or_motion', 'segmentation_enabled': True}
    source_config['fusion']['mode'] = 'target_gated'
    source_config['fusion'].pop('temporal_lowfreq_weight')
    source_config['asset_fingerprints']['person/pmt_small_coco.safetensors'] = 'pmt-sha'
    model = nn.Module()
    model.decoder = nn.Module()
    model.decoder.fused = nn.Linear(2, 1)
    with torch.no_grad():
        model.decoder.fused.weight.fill_(.4)
        model.decoder.fused.bias.fill_(.2)
    source = {
        'format_version': 1, 'architecture': legacy.ARCHITECTURE,
        'training_recipe': legacy.TRAINING_RECIPE, 'stage': 'b', 'step': 3000,
        'manifest_identity': 'manifest', 'config': source_config,
        'model_config': deepcopy(source_config['model']), 'resources': deepcopy(legacy.RESOURCES),
        'asset_fingerprints': deepcopy(source_config['asset_fingerprints']),
        'model': deepcopy(model.state_dict()), 'temporal_reference': {'ir_motion.weight': torch.ones(1)},
        'optimizer': {'must_not_restore': True}, 'best_score': -999,
        'early_stopping': {'stopped': True}, 'logical_sample_count': 999999,
    }
    source['model']['dino.person_head.unneeded.weight'] = torch.ones(3)
    source['loss_fingerprint'] = legacy.loss_fingerprint(source_config)
    return model, target, source


def test_v6_migration_preserves_fused_weights_and_identifies_reset(tmp_path):
    model, target, source = migration_case()
    path = tmp_path / 'v6.pt'
    torch.save(source, path)
    loaded, metadata = ckpt.load_v6_initialization(path, target, 'manifest')
    with torch.no_grad():
        model.decoder.fused.weight.zero_()
        model.decoder.fused.bias.zero_()
    ckpt.load_v6_weights(model, loaded)
    for name, value in model.state_dict().items():
        assert torch.equal(value, source['model'][name])
    assert metadata['source_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata['source_stage'] == 'b' and metadata['source_step'] == 3000
    for name in ('reset_optimizer', 'reset_steps', 'reset_best_and_early_stopping', 'preserve_fused_head'):
        assert metadata[name] is True
    assert metadata['discarded_parameter_prefix'] == 'dino.person_head.'
    with pytest.raises(ValueError, match='architecture mismatch'):
        ckpt.load_checkpoint(path)
    with pytest.raises(ValueError, match='Architecture/training recipe mismatch'):
        ckpt.validate_training_checkpoint(source, target)


@pytest.mark.parametrize('mutation,reason', [
    (lambda source: source.update(manifest_identity='other'), 'manifest identity'),
    (lambda source: source.update(stage='a'), 'B/C'),
    (lambda source: source.update(temporal_reference=None), 'temporal reference'),
    (lambda source: source.update(loss_fingerprint='bad'), 'fingerprint'),
    (lambda source: source['asset_fingerprints'].update({'rife/flownet.pkl': 'other'}), 'fingerprint'),
    (lambda source: source.update(resources={}), 'resource'),
])
def test_v6_migration_rejects_incompatible_source(tmp_path, mutation, reason):
    _, target, source = migration_case()
    mutation(source)
    path = tmp_path / 'bad.pt'
    torch.save(source, path)
    with pytest.raises(ValueError, match=reason):
        ckpt.load_v6_initialization(path, target, 'manifest')


def test_v6_migration_rejects_changed_shared_architecture_and_pretrained_assets(tmp_path):
    _, target, source = migration_case()
    path = tmp_path / 'v6.pt'
    torch.save(source, path)
    bad = deepcopy(target)
    bad['model']['semantic_dim'] = 100
    with pytest.raises(ValueError, match='shared model configuration'):
        ckpt.load_v6_initialization(path, bad, 'manifest')
    bad = deepcopy(target)
    bad['asset_fingerprints']['rife/flownet.pkl'] = 'other'
    with pytest.raises(ValueError, match='Shared pretrained fingerprint'):
        ckpt.load_v6_initialization(path, bad, 'manifest')


@pytest.mark.parametrize('mutation,reason', [
    (lambda source: source['model'].pop('decoder.fused.bias'), 'migration keys'),
    (lambda source: source['model'].update({'unexpected.weight': torch.ones(1)}), 'migration keys'),
    (lambda source: source['model'].update({'decoder.fused.weight': torch.ones(3, 2)}), 'shape mismatch'),
])
def test_v6_migration_allows_only_pmt_parameter_removal(mutation, reason):
    model, _, source = migration_case()
    mutation(source)
    with pytest.raises(ValueError, match=reason):
        ckpt.load_v6_weights(model, source)


def v7_case():
    _, config, _ = migration_case()
    source = {'format_version': 1, 'architecture': ckpt.ARCHITECTURE,
              'training_recipe': ckpt.TRAINING_RECIPE, 'stage': 'joint',
              'config': deepcopy(config), 'model_config': deepcopy(config['model']),
              'segmentation_enabled': False, 'asset_fingerprints': deepcopy(config['asset_fingerprints']),
              'loss_fingerprint': ckpt.loss_fingerprint(config)}
    return config, source


def test_safi_exact_resume_allows_batch_change_but_not_objective_or_schedule_change():
    config, source = v7_case()
    changed = deepcopy(config)
    changed['stages']['joint'].update(batch_size=16, accumulation=4)
    ckpt.validate_training_checkpoint(source, changed)
    for key, value in [('task_lr', 2e-5), ('task_lr_min', 1e-6), ('temporal_warmup_steps', 1000)]:
        changed = deepcopy(config)
        changed['stages']['joint'][key] = value
        with pytest.raises(ValueError, match='fingerprint mismatch'):
            ckpt.validate_training_checkpoint(source, changed)
    changed = deepcopy(config)
    changed['fusion']['temporal_lowfreq_weight'] *= 2
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        ckpt.validate_training_checkpoint(source, changed)


def test_safi_resume_rejects_changed_validation_or_inconsistent_metadata():
    config, source = v7_case()
    changed = deepcopy(config)
    changed['validation_clips'] += 1
    with pytest.raises(ValueError, match='validation configuration mismatch'):
        ckpt.validate_training_checkpoint(source, changed)
    source['segmentation_enabled'] = True
    with pytest.raises(ValueError, match='segmentation mode is inconsistent'):
        ckpt.validate_training_checkpoint(source, config)
