"""Build compact, portable sequence-level splits without changing the dataset."""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


def temporal_index(name):
    groups = re.findall(r'\d+', Path(name).stem)
    if not groups:
        raise ValueError(f'No temporal index in {name}')
    return int(''.join(groups))


def sequence_source(sequence):
    for prefix, source in (('hdo_', 'HDO'), ('m3_', 'M3SVD'), ('vt_', 'VTMOT')):
        if sequence.startswith(prefix):
            return source
    raise ValueError(f'Unknown sequence source: {sequence}')


def _lineage(row, sequence, source):
    # Retain source identity, never old machine paths or original train/test folders.
    def leaf(value):
        path = Path(value)
        if path.name in ('visible', 'infrared'):
            path = path.parent
        return path.name
    return {'source': source,
            'visible': leaf(row['visible_source']) if row.get('visible_source') else sequence,
            'infrared': leaf(row['infrared_source']) if row.get('infrared_source') else sequence}


def manifest_fingerprint(manifest):
    contents = {k: v for k, v in manifest.items() if k != 'fingerprint'}
    return hashlib.sha256(json.dumps(contents, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def build_manifest(root, seed=3407, val_fraction=0.1):
    root = Path(root).resolve()
    if not root.is_dir() or not 0 <= val_fraction < 1:
        raise ValueError('Dataset root must exist and validation fraction must be in [0,1)')
    provenance = {}
    annotation = root / 'annotations' / 'sequences.jsonl'
    if annotation.exists():
        with annotation.open() as handle:
            for line in handle:
                row = json.loads(line)
                provenance[row['split'], row['sequence_id']] = row
    original = {'train': [], 'test': []}
    for split in original:
        folder = root / split
        if not folder.is_dir():
            continue
        for sequence in sorted(p for p in folder.iterdir() if p.is_dir()):
            source = sequence_source(sequence.name)
            modalities = [sequence / 'infrared', sequence / 'visible']
            if any(not p.is_dir() for p in modalities):
                raise ValueError(f'Missing paired modality directory: {sequence.name}')
            names = [{p.name for p in directory.iterdir() if p.is_file() and
                      p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')}
                     for directory in modalities]
            if not names[0] or names[0] != names[1]:
                raise ValueError(f'Invalid paired image filenames: {split}/{sequence.name}')
            frames = sorted(names[0], key=lambda name: (temporal_index(name), name))
            indices = [temporal_index(name) for name in frames]
            if len(set(indices)) != len(indices):
                raise ValueError(f'Duplicate temporal indices: {sequence.name}')
            starts = [i for i in range(len(frames) - 4)
                      if indices[i + 1] > indices[i] and
                      all(indices[j + 1] - indices[j] == indices[i + 1] - indices[i]
                          for j in range(i + 1, i + 4))]
            row = provenance.get((split, sequence.name), {})
            original[split].append({'source': source, 'sequence_id': sequence.name,
                                    'lineage': _lineage(row, sequence.name, source),
                                    'original_split': split,
                                    'ir_dir': modalities[0].relative_to(root).as_posix(),
                                    'vis_dir': modalities[1].relative_to(root).as_posix(),
                                    'frames': frames, 'indices': indices, 'window_starts': starts})
    lineage_key = lambda r: json.dumps(r['lineage'], sort_keys=True)
    test_keys = {lineage_key(row) for row in original['test']}
    excluded = [row for row in original['train'] if lineage_key(row) in test_keys]
    candidates = [row for row in original['train'] if lineage_key(row) not in test_keys]
    no_windows = [row for row in candidates if not row['window_starts']]
    eligible = [row for row in candidates if row['window_starts']]
    # Group lineage aliases together so no originating sequence can leak to validation.
    train, validation = [], []
    rng = random.Random(seed)
    for source in sorted({row['source'] for row in eligible}):
        by_lineage = {}
        for row in eligible:
            if row['source'] == source:
                by_lineage.setdefault(lineage_key(row), []).append(row)
        keys = sorted(by_lineage)
        rng.shuffle(keys)
        count = max(1, round(len(keys) * val_fraction)) if val_fraction and len(keys) > 1 else 0
        val_keys = set(keys[:count])
        for key in sorted(by_lineage):
            (validation if key in val_keys else train).extend(by_lineage[key])
    splits = {'train': train, 'val': validation, 'test': original['test']}
    counts = {split: {'sequences': len(rows), 'frames': sum(len(r['frames']) for r in rows),
                      'windows': sum(len(r['window_starts']) for r in rows),
                      'sources': dict(sorted(Counter(r['source'] for r in rows).items()))}
              for split, rows in splits.items()}
    result = {'version': 1, 'seed': seed, 'val_fraction': val_fraction,
              'time_coordinates': 'synthetic_uniform_0_to_4', 'splits': splits,
              'excluded_overlap': [{k: r[k] for k in ('source', 'sequence_id', 'lineage', 'original_split')}
                                   for r in excluded],
              'excluded_no_windows': [{k: r[k] for k in ('source', 'sequence_id', 'lineage', 'original_split')}
                                      for r in no_windows], 'counts': counts}
    result['fingerprint'] = manifest_fingerprint(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=3407)
    args = parser.parse_args(argv)
    root, output = Path(args.data_root).resolve(), Path(args.output).resolve()
    if output.is_relative_to(root):
        parser.error('Manifest output must be outside the immutable dataset root')
    manifest = build_manifest(root, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        json.dump(manifest, handle, separators=(',', ':'), allow_nan=False)
        handle.write('\n')
    print(json.dumps({'counts': manifest['counts'], 'fingerprint': manifest['fingerprint']}))


if __name__ == '__main__':
    main()
