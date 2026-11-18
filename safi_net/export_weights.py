"""Export a self-contained FP32 inference checkpoint from a SAFI-Net training run."""
import argparse
import hashlib
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from .checkpoints import (ARCHITECTURE, INFERENCE_FORMAT, atomic_save,
                          read_inference_checkpoint, reconstruct_model)
from .compat.identity import normalize_checkpoint


def export(checkpoint_path, output, pretrained_root=None):
    source, output = Path(checkpoint_path), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output file already exists: {output}')

    digest = hashlib.sha256()
    with source.open('rb') as handle:
        before = os.fstat(handle.fileno())
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
        handle.seek(0)
        snapshot = normalize_checkpoint(read_inference_checkpoint(handle))
        after = os.fstat(handle.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('Checkpoint changed during export')
    if snapshot.get('architecture') != ARCHITECTURE:
        raise ValueError('Expected a full SAFI-Net training checkpoint')
    model = reconstruct_model(snapshot, pretrained_root=pretrained_root)
    payload = {
        'format': INFERENCE_FORMAT,
        'architecture': ARCHITECTURE,
        'model_config': snapshot['model_config'],
        'model': model.state_dict(),
        'source_checkpoint_sha256': digest.hexdigest(),
        'asset_fingerprints': snapshot['asset_fingerprints'],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temp:
        staged = Path(temp) / output.name
        atomic_save(payload, staged)
        os.link(staged, output)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Training checkpoint such as best.pt')
    parser.add_argument('--output', required=True)
    parser.add_argument('--pretrained-root')
    args = parser.parse_args(argv)
    print(export(args.checkpoint, args.output, args.pretrained_root))


if __name__ == '__main__':
    main()
