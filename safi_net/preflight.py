"""Offline-only integrity, strict architecture loading and decoding checks."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path

import torch

from .checkpoints import RESOURCES
from .data import AsyncClipDataset, balanced_sample_key, collate_clips
from .models import build_model
from .inference_policy import segmentation_identity
from .paths import PACKAGE_ROOT, relative_path, resolve_paths
from .runtime import add_runtime_arguments, apply_runtime_arguments, load_config, resolve_manifest, validate_config


def verify_assets(pretrained_root, model_config=None):
    root = Path(pretrained_root)
    checksum_file = root / 'SHA256SUMS'
    if not checksum_file.is_file():
        raise FileNotFoundError(f'Offline asset checksum manifest required: {checksum_file}')
    def read_sums(path):
        result = {}
        for line in path.read_text().splitlines():
            if line.strip():
                digest, name = line.split(maxsplit=1)
                result[name.strip().lstrip('*')] = digest
        return result
    expected = read_sums(PACKAGE_ROOT / 'pretrained/SHA256SUMS')
    supplied = read_sums(checksum_file)
    result = {}
    from .checkpoints import resources_for
    for name in sorted(set(resources_for({'model': model_config or {}}).values())):
        path = relative_path(root, name)
        if not path.is_file():
            raise FileNotFoundError(f'Genuine offline pretrained asset required: {path}')
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        actual = digest.hexdigest()
        if name not in expected or supplied.get(name) != expected[name] or actual != expected[name]:
            raise ValueError(f'Pretrained asset SHA256 mismatch: {path}')
        result[name] = actual
    return result


def preflight(config, device='cuda'):
    validate_config(config)
    roots = resolve_paths(config['data'].get('root'), config.get('pretrained_root'), config.get('run_dir'))
    resolved = {'package_root': str(PACKAGE_ROOT),
                'data_root': str(roots.data_root), 'pretrained_root': str(roots.pretrained_root),
                'run_dir': str(roots.run_dir)}
    print(json.dumps({'preflight': 'resolved_paths', **resolved}), flush=True)
    for name in ('torch', 'numpy', 'PIL', 'yaml', 'transformers', 'safetensors', 'cv2'):
        importlib.import_module(name)
    assets = verify_assets(roots.pretrained_root, model_config=config['model'])
    saved_assets = config.get('asset_fingerprints')
    if saved_assets is not None and saved_assets != assets:
        raise ValueError('Configured checkpoint asset fingerprints differ from local pretrained assets')
    config['asset_fingerprints'] = assets
    if not roots.data_root.is_dir():
        raise FileNotFoundError(f'Dataset root does not exist: {roots.data_root}')
    ancestor = roots.run_dir
    while not ancestor.exists():
        ancestor = ancestor.parent
    import os
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
        raise ValueError(f'Run directory is not writable: {roots.run_dir}')
    manifest = resolve_manifest(config)
    decoded = []
    for split in ('train', 'val', 'test'):
        dataset = AsyncClipDataset(roots.data_root, manifest, split,
                                   crop_size=config['data']['crop_size'], training=False)
        if split in ('train', 'val') and not dataset.groups:
            raise ValueError(f'Manifest {split} split has no usable windows')
        if dataset.groups:
            for index in range(2 * len(dataset.groups)):
                key = balanced_sample_key(dataset, index, config['seed'])
                batch = collate_clips([dataset[key]])
                if batch['ir_targets'].shape[1:3] != (3, 1) or batch['vis_targets'].shape[1:3] != (3, 3):
                    raise ValueError('Decoded sample violates the clip load contract')
                decoded.append({'split': split, 'source': dataset.windows[key.window_index]['source'],
                                'protocol': key.protocol})
    target = torch.device(device)
    gpu = None
    if target.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but no visible CUDA device is available')
        if config['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('Visible GPU does not support bf16; select --precision float32')
        properties = torch.cuda.get_device_properties(target)
        gpu = {'name': properties.name, 'total_memory_bytes': properties.total_memory,
               'visible_devices': torch.cuda.device_count()}
    model = build_model(config['model'], pretrained_root=roots.pretrained_root)
    if {p.data_ptr() for p in model.ir_motion.parameters()} & {p.data_ptr() for p in model.vis_motion.parameters()}:
        raise ValueError('IR/VIS motion storage is shared')
    architecture = model.architecture_id
    del model
    report = {**segmentation_identity(config['model']['target_gate']), 'phase': 'preflight_passed', 'architecture': architecture, **resolved, 'assets': assets,
              'manifest_identity': manifest['fingerprint'], 'decoded': decoded,
              'device': str(target), 'gpu': gpu}
    print(json.dumps(report, allow_nan=False), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    config = apply_runtime_arguments(load_config(args.config), args)
    preflight(config, device=args.device)


if __name__ == '__main__':
    main()
