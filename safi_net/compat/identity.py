"""Validate historical identities before making a SAFI-Net runtime view."""
from copy import deepcopy

OLD_IDENTITIES = {
    'rst_async_v7': ('full_image_joint_temporal_v7', 'safi_net', 'full_image_joint_temporal'),
    'rst_async_v7_ablation_v2': ('full_image_joint_temporal_v7_ablation_v2',
                              'safi_net_ablation_v2', 'full_image_joint_temporal_ablation_v2'),
}


def normalize_config(config):
    result = deepcopy(config)
    if result.get('architecture') in OLD_IDENTITIES:
        # The public config architecture denotes the shared model family.
        result['architecture'] = 'safi_net'
    return result


def normalize_checkpoint(checkpoint):
    identity = OLD_IDENTITIES.get(checkpoint.get('architecture'))
    if identity is None:
        return checkpoint
    from .legacy_checkpoints import validate_training_checkpoint
    validate_training_checkpoint(checkpoint, checkpoint.get('config', {}))
    recipe, architecture, target_recipe = identity
    if checkpoint.get('training_recipe') != recipe:
        raise ValueError('Historical training recipe mismatch')
    result = dict(checkpoint)
    result['config'] = normalize_config(checkpoint['config'])
    result['architecture'] = architecture
    result['training_recipe'] = target_recipe
    from ..checkpoints import loss_fingerprint
    result['loss_fingerprint'] = loss_fingerprint(result['config'])
    result['legacy_identity'] = {
        'architecture': checkpoint['architecture'], 'training_recipe': recipe,
        'loss_fingerprint': checkpoint['loss_fingerprint'], 'file_unchanged': True,
    }
    return result

V5_ARCHITECTURE = "rst_async_v5"
V5_TRAINING_RECIPE = "query_semantic_temporal_v5"
V3_ARCHITECTURE = "rst_async_v3"
V3_TRAINING_RECIPE = "target_gated_vis_only_v3"
V6_ARCHITECTURE = "rst_async_v6"
