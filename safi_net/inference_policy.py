"""Explicit segmentation policy shared by inference entry points and receipts."""


def segmentation_identity(source):
    """Return the effective policy, validating any redundant human-readable label."""
    enabled = source.get('segmentation_enabled', True)
    if type(enabled) is not bool:
        raise ValueError('segmentation_enabled must be a boolean')
    mode = 'on' if enabled else 'off'
    if source.get('segmentation', mode) != mode:
        raise ValueError('Inconsistent segmentation mode in output identity')
    return {'segmentation_enabled': enabled, 'segmentation': mode}


def resolve_segmentation(checkpoint, requested=None):
    if requested not in (None, 'on', 'off'):
        raise ValueError('Segmentation must be on or off')
    config = checkpoint.get('model_config', checkpoint.get('config', {}).get('model', {}))
    saved = config.get('target_gate', {}).get('segmentation_enabled', True)
    return segmentation_identity({'segmentation_enabled': saved if requested is None else requested == 'on'})


def configure_segmentation(model, requested=None, *, enabled=None):
    """Overrides are runtime policy only; checkpoint tensors are loaded unchanged."""
    identity = resolve_segmentation({'model_config': getattr(model, 'config', {})}, requested)
    if enabled is not None:
        identity = segmentation_identity({'segmentation_enabled': enabled})
    setter = getattr(model, 'set_segmentation_enabled', None)
    if setter is not None:
        setter(identity['segmentation_enabled'])
    elif not identity['segmentation_enabled']:
        raise ValueError('This model does not support disabling segmentation')
    return identity


def require_same_segmentation(value, expected):
    if segmentation_identity(value) != segmentation_identity(expected):
        raise ValueError('Segmentation mode differs; use a new output directory')


def add_segmentation_argument(parser):
    parser.add_argument('--segmentation', choices=('off',), default=None,
                        help='SAFI-Net full-image fusion uses segmentation off')
