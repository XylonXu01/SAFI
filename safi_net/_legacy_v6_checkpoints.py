"""Portable, atomic trusted-local checkpoints and inference reconstruction."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

ARCHITECTURE = 'rst_async_v6'
TRAINING_RECIPE = 'semantic_motion_temporal_v6'
V5_ARCHITECTURE = 'rst_async_v5'
V5_TRAINING_RECIPE = 'query_semantic_temporal_v5'
V3_ARCHITECTURE = 'rst_async_v3'
V3_TRAINING_RECIPE = 'target_gated_vis_only_v3'
RESOURCES = {'ir_motion': 'rife/flownet.pkl', 'vis_motion': 'rife/flownet.pkl',
             'semantics': 'dinov3/model.safetensors',
             'person': 'person/pmt_small_coco.safetensors'}


def _loss_fingerprint(config, architecture, recipe):
    """Identify the complete loss recipe; batch sizes and step budgets are mutable."""
    from ._legacy_v6_losses import normalize_training_fusion_config
    from ._legacy_v6_losses import LossComputer
    fusion = normalize_training_fusion_config(config.get('fusion'))
    ablation = config.get('model', {}).get('ablation')
    objectives = {stage: LossComputer(stage, fusion, ablation=ablation) for stage in 'abc'}
    if architecture == V5_ARCHITECTURE:
        for objective in objectives.values():
            objective.weights.pop('applied_luma', None)
    specification = {'recipe': recipe, 'architecture': architecture, 'fusion': fusion,
                     'model': config.get('model', {}), 'assets': config.get('asset_fingerprints', {}),
                     'stages': {stage: {'weights': objective.weights,
                                        'temporal_weight': objective.temporal_weight}
                                for stage, objective in objectives.items()}}
    encoded = json.dumps(specification, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def loss_fingerprint(config):
    return _loss_fingerprint(config, ARCHITECTURE, TRAINING_RECIPE)


def validate_training_checkpoint(checkpoint, config, *, initialization=False):
    """V6 resume is exact; explicit weights-only forks may change segmentation mode."""
    if (checkpoint.get('architecture') != ARCHITECTURE
            or checkpoint.get('training_recipe') != TRAINING_RECIPE):
        raise ValueError('Architecture/training recipe mismatch; v6 requires a v6 checkpoint, '
                         'or explicit --init-from-v5 for baseline migration')
    saved_config = checkpoint.get('config', {})
    if 'model_config' in checkpoint and checkpoint['model_config'] != saved_config.get('model'):
        raise ValueError('Checkpoint model architecture configuration is inconsistent with its saved config')
    if ('segmentation_enabled' in checkpoint and checkpoint['segmentation_enabled'] !=
            saved_config.get('model', {}).get('target_gate', {}).get('segmentation_enabled', True)):
        raise ValueError('Checkpoint segmentation mode is inconsistent with its saved config')
    stored = checkpoint.get('loss_fingerprint')
    if not stored or stored != loss_fingerprint(saved_config):
        raise ValueError('Checkpoint loss fingerprint is missing or inconsistent with its saved config')
    if checkpoint.get('asset_fingerprints', {}) != saved_config.get('asset_fingerprints', {}):
        raise ValueError('Checkpoint asset fingerprint is inconsistent with its saved config')
    target = deepcopy(config)
    if initialization:
        target.setdefault('model', {}).setdefault('target_gate', {})['segmentation_enabled'] = (
            saved_config.get('model', {}).get('target_gate', {}).get('segmentation_enabled', True))
    if stored != loss_fingerprint(target):
        raise ValueError('v6 loss/model/semantic asset fingerprint mismatch; resume requires the saved recipe; '
                         'only --init-from in a fresh run can change segmentation mode')
    if not initialization:
        def validation_settings(value):
            # Match the runtime defaults when older portable configs omit them.
            return {'validation_clips': value.get('validation_clips', 150),
                    'seed': value.get('seed', 3407),
                    'crop_size': value.get('data', {}).get('crop_size', 256),
                    'checkpoint_policy': value.get('checkpoint_policy', 'best')}
        if validation_settings(checkpoint.get('config', {})) != validation_settings(config):
            raise ValueError('Resume validation configuration mismatch (validation_clips, seed, crop_size or checkpoint_policy); '
                             'use --init-from in a fresh run so best_score is reset')


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': {'format': 'current_device_v1',
                     'state': torch.cuda.get_rng_state(torch.cuda.current_device()).cpu()}
            if torch.cuda.is_initialized() else None}


def restore_rng(state):
    cuda = state['cuda']
    if cuda is not None and (not isinstance(cuda, dict) or cuda.get('format') != 'current_device_v1'):
        raise ValueError('Unsupported CUDA RNG checkpoint format; use --init-from for a weights-only fork')
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if cuda is not None:
        torch.cuda.set_rng_state(cuda['state'].cpu(), torch.cuda.current_device())


@contextmanager
def preserve_rng():
    state = capture_rng()
    try:
        yield
    finally:
        restore_rng(state)


def atomic_save(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, architecture=ARCHITECTURE, manifest_identity=None, *, initialization=False):
    checkpoint = torch.load(Path(path), map_location='cpu', weights_only=False)
    if checkpoint.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format version')
    if checkpoint.get('architecture') != architecture:
        raise ValueError('Checkpoint architecture mismatch; v5 migration requires explicit --init-from-v5')
    if manifest_identity is not None and checkpoint.get('manifest_identity') != manifest_identity:
        raise ValueError('Checkpoint manifest identity mismatch')
    return checkpoint


def _weight_model_config(config):
    result = deepcopy(config)
    result.setdefault('target_gate', {}).pop('segmentation_enabled', None)
    return result


def load_initial_weights(model, checkpoint, model_config):
    """Weights-only v6 stage transition or same-stage fork; never restore counters."""
    if checkpoint.get('architecture') != ARCHITECTURE:
        raise ValueError('Checkpoint architecture mismatch; use explicit --init-from-v5')
    if _weight_model_config(checkpoint['model_config']) != _weight_model_config(model_config):
        raise ValueError('Checkpoint model architecture configuration mismatch')
    model.load_state_dict(checkpoint['model'], strict=True)
    if checkpoint.get('stage') == 'a':
        _reset_fusion_head(model)


def validate_v5_initialization(checkpoint, config, manifest_identity=None):
    """Verify the historical baseline independently of the v6 training recipe."""
    if (checkpoint.get('format_version') != 1 or checkpoint.get('architecture') != V5_ARCHITECTURE
            or checkpoint.get('training_recipe') != V5_TRAINING_RECIPE or checkpoint.get('stage') != 'c'):
        raise ValueError('--init-from-v5 requires a baseline v5 stage-C checkpoint')
    saved = checkpoint.get('config', {})
    if saved.get('model', {}).get('ablation') is not None or checkpoint.get('model_config', {}).get('ablation') is not None:
        raise ValueError('v5 migration does not accept ablation checkpoints')
    if checkpoint.get('resources', RESOURCES) != RESOURCES:
        raise ValueError('v5 migration resource identifiers mismatch')
    if manifest_identity is not None and checkpoint.get('manifest_identity') != manifest_identity:
        raise ValueError('Checkpoint manifest identity mismatch')
    stored = checkpoint.get('loss_fingerprint')
    if not stored or stored != _loss_fingerprint(saved, V5_ARCHITECTURE, V5_TRAINING_RECIPE):
        raise ValueError('v5 checkpoint loss fingerprint is missing or inconsistent')
    if (checkpoint.get('asset_fingerprints', {}) != saved.get('asset_fingerprints', {})
            or checkpoint.get('asset_fingerprints', {}) != config.get('asset_fingerprints', {})):
        raise ValueError('v5 migration pretrained asset fingerprints mismatch')
    if checkpoint.get('model_config') != saved.get('model'):
        raise ValueError('v5 checkpoint model configuration is inconsistent')
    source_model, target_model = deepcopy(checkpoint['model_config']), deepcopy(config['model'])
    source_model.pop('target_gate', None)
    target_model.pop('target_gate', None)
    if source_model != target_model:
        raise ValueError('v5 migration shared model architecture configuration mismatch')
    if checkpoint.get('temporal_reference') is None:
        raise ValueError('v5 migration requires its frozen temporal_reference state')


def load_v5_initialization(path, config, manifest_identity=None):
    checkpoint = load_checkpoint(path, architecture=V5_ARCHITECTURE, manifest_identity=manifest_identity)
    validate_v5_initialization(checkpoint, config, manifest_identity)
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    metadata = {'kind': 'v5_stage_c_weights_and_reference', 'source_path': str(Path(path).resolve()),
                'source_sha256': digest.hexdigest(), 'source_architecture': checkpoint['architecture'],
                'source_training_recipe': checkpoint['training_recipe'], 'source_stage': checkpoint['stage'],
                'source_step': checkpoint['step'], 'source_loss_fingerprint': checkpoint['loss_fingerprint'],
                'reset_optimizer': True, 'reset_steps': True, 'reset_best_and_early_stopping': True,
                'preserve_fused_head': True}
    return checkpoint, metadata


def load_v5_weights(model, checkpoint):
    """Exact learned tensor identity; v6 adds parameter-free routing only."""
    source, target = checkpoint['model'], model.state_dict()
    if set(source) != set(target):
        raise ValueError('v5 migration shared state keys mismatch')
    for name in target:
        if source[name].shape != target[name].shape:
            raise ValueError(f'v5 migration shared state shape mismatch: {name}')
    model.load_state_dict(source, strict=True)


def _reset_fusion_head(model):
    # Zero output starts B exactly at the VIS reference, including A checkpoints
    # whose never-trained fused branch happens to contain arbitrary parameters.
    with torch.no_grad():
        for parameter in model.decoder.fused.parameters():
            parameter.zero_()


def reconstruct_model(checkpoint_or_path, pretrained_root=None, device='cpu', *, test_backbones=None):
    from .models import build_model
    checkpoint = (load_checkpoint(checkpoint_or_path)
                  if isinstance(checkpoint_or_path, (str, Path)) else checkpoint_or_path)
    if checkpoint.get('architecture') != ARCHITECTURE or checkpoint.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint architecture or format')
    if checkpoint.get('resources', RESOURCES) != RESOURCES:
        raise ValueError('Checkpoint resource identifiers do not match this architecture')
    validate_training_checkpoint(checkpoint, checkpoint.get('config', {}))
    if test_backbones is None:
        from .paths import PACKAGE_ROOT
        from .preflight import verify_assets
        root = Path(pretrained_root) if pretrained_root is not None else PACKAGE_ROOT / 'pretrained'
        actual = verify_assets(root)
        if checkpoint.get('asset_fingerprints') != actual:
            raise ValueError('Inference asset fingerprints differ from the checkpoint pretrained assets')
    model = build_model(checkpoint['model_config'], pretrained_root=pretrained_root,
                        test_backbones=test_backbones)
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.to(device).eval()


def reconstruct_initial_model(checkpoint_or_path, config, pretrained_root=None, device='cpu', *,
                              test_backbones=None):
    """Explicit weights-only evaluation using the same migration checks as training.

    This returns a v6 model without restoring optimizer, stage counters or RNG.
    """
    from .models import build_model
    from .runtime import resolve_manifest, validate_config
    target = deepcopy(config)
    validate_config(target)
    manifest = resolve_manifest(target)
    checkpoint = (load_checkpoint(checkpoint_or_path, initialization=True,
                                  manifest_identity=manifest['fingerprint'])
                  if isinstance(checkpoint_or_path, (str, Path)) else checkpoint_or_path)
    if checkpoint.get('format_version') != 1:
        raise ValueError('Unsupported checkpoint format version')
    if checkpoint.get('manifest_identity') != manifest['fingerprint']:
        raise ValueError('Checkpoint manifest identity mismatch')
    if test_backbones is None:
        from .paths import PACKAGE_ROOT
        from .preflight import verify_assets
        root = Path(pretrained_root) if pretrained_root is not None else PACKAGE_ROOT / 'pretrained'
        actual = verify_assets(root)
        if target.get('asset_fingerprints', actual) != actual:
            raise ValueError('Configured inference asset fingerprints differ from local pretrained assets')
        target['asset_fingerprints'] = actual
    validate_training_checkpoint(checkpoint, target, initialization=True)
    model = build_model(target['model'], pretrained_root=pretrained_root, test_backbones=test_backbones)
    load_initial_weights(model, checkpoint, target['model'])
    return model.to(device).eval()
