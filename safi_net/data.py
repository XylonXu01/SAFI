"""Five-frame clips with stateless augmentation and target/input isolation."""
import bisect
import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .paths import relative_path
from .temporal import observation_indices, canonicalize_inputs
from .types import TemporalInputs


@dataclass(frozen=True)
class SampleKey:
    window_index: int
    protocol: str
    seed: int
    # None preserves the full three-query dataset/validation interface.
    query_index: int | None = None


class _Windows(Sequence):
    """Expand paths only for requested windows, not all half-million clips."""

    def __init__(self, records):
        self.records = records
        self.ends = []
        total = 0
        for record in records:
            total += len(record['window_starts'])
            self.ends.append(total)

    def __len__(self):
        return self.ends[-1] if self.ends else 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = bisect.bisect_right(self.ends, index)
        record = self.records[position]
        local = index - (self.ends[position - 1] if position else 0)
        start = record['window_starts'][local]
        names = record['frames'][start:start + 5]
        return {'source': record['source'], 'sequence_id': record['sequence_id'],
                'sample_id': f"{record['sequence_id']}:{names[0]}-{names[-1]}",
                'indices': record['indices'][start:start + 5],
                'ir': [str(Path(record['ir_dir']) / name) for name in names],
                'vis': [str(Path(record['vis_dir']) / name) for name in names]}


def _sample_seed(seed, index):
    return int.from_bytes(hashlib.sha256(f'{seed}:{index}'.encode()).digest()[:8], 'big')


def balanced_sample_key(dataset, global_index, seed=3407):
    if not isinstance(global_index, int) or global_index < 0 or not dataset.groups:
        raise ValueError('Sampling requires a nonnegative integer index and nonempty dataset')
    sources = sorted(dataset.groups)
    block, slot = divmod(global_index, 2 * len(sources))
    choices = [(source, protocol) for source in sources for protocol in ('same', 'vis_fast')]
    random.Random(_sample_seed(seed, block)).shuffle(choices)
    source, protocol = choices[slot]
    sample_seed = _sample_seed(seed, global_index)
    rng = random.Random(sample_seed)
    sequence = rng.choice(sorted(dataset.groups[source]))
    index = rng.choice(dataset.groups[source][sequence])
    return SampleKey(index, protocol, sample_seed)


class AsyncClipDataset(Dataset):
    def __init__(self, root, manifest, split, crop_size=256, training=True):
        self.root = Path(root).resolve()
        if isinstance(manifest, (str, Path)):
            with Path(manifest).open() as handle:
                manifest = json.load(handle)
        if manifest.get('version') != 1 or split not in manifest['splits']:
            raise ValueError('Unsupported manifest version or split')
        if not isinstance(crop_size, int) or crop_size < 1:
            raise ValueError('crop_size must be a positive integer')
        self.manifest, self.split = manifest, split
        self.crop_size, self.training = crop_size, training
        self.windows = _Windows(manifest['splits'][split])
        self.groups = {}
        offset = 0
        for record in self.windows.records:
            count = len(record['window_starts'])
            if count:
                self.groups.setdefault(record['source'], {}).setdefault(record['sequence_id'], []).extend(
                    range(offset, offset + count))
            offset += count

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, key):
        if isinstance(key, int):
            key = SampleKey(key, 'same' if key % 2 == 0 else 'vis_fast', _sample_seed(3407, key))
        if not isinstance(key, SampleKey) or key.protocol not in ('same', 'vis_fast'):
            raise ValueError('Expected integer or SampleKey with same/vis_fast protocol')
        if key.query_index is not None and (type(key.query_index) is not int or not 0 <= key.query_index < 3):
            raise ValueError('Selected query_index must be 0, 1 or 2')
        row = self.windows[key.window_index]
        tensors = []
        for modality, mode in (('ir', 'L'), ('vis', 'RGB')):
            for value in row[modality]:
                with Image.open(relative_path(self.root, value)) as image:
                    array = np.array(image.convert(mode), copy=True)
                tensor = torch.from_numpy(array).float() / 255.
                tensors.append(tensor[None] if tensor.ndim == 2 else tensor.permute(2, 0, 1))
        shapes = {tuple(t.shape[-2:]) for t in tensors}
        if len(shapes) != 1:
            raise ValueError('Paired image dimensions must match throughout the clip')
        rng = random.Random(key.seed)
        h, w = tensors[0].shape[-2:]
        if min(h, w) < self.crop_size:
            scale = self.crop_size / min(h, w)
            h, w = max(self.crop_size, round(h * scale)), max(self.crop_size, round(w * scale))
            tensors = [F.interpolate(t[None], (h, w), mode='bilinear', align_corners=False,
                                     antialias=True)[0] for t in tensors]
        top = rng.randrange(h - self.crop_size + 1) if self.training else (h - self.crop_size) // 2
        left = rng.randrange(w - self.crop_size + 1) if self.training else (w - self.crop_size) // 2
        tensors = [t[:, top:top + self.crop_size, left:left + self.crop_size] for t in tensors]
        horizontal_flip = self.training and rng.random() < .5
        if horizontal_flip:
            tensors = [t.flip(-1) for t in tensors]
        ir, vis = torch.stack(tensors[:5]), torch.stack(tensors[5:])
        if self.training and rng.random() < .5:
            ir, vis = ir.flip(0), vis.flip(0)
        ir_slots = [0, 2, 4]
        vis_slots = ir_slots if key.protocol == 'same' else [0, 1, 2, 3, 4]
        def make_query(islots, vslots, query):
            # Select the final pair once, instead of first copying all candidate
            # frames and then selecting the same pair from that temporary bank.
            ii = [islots[index] for index in observation_indices(islots, query)]
            vi = [vslots[index] for index in observation_indices(vslots, query)]
            inputs = TemporalInputs(
                ir[ii][None], vis[vi][None],
                torch.tensor([ii], dtype=torch.float64), torch.tensor([vi], dtype=torch.float64),
                torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool),
                torch.tensor([query], dtype=torch.float64))
            return canonicalize_inputs(inputs)
        query_times = (1., 2., 3.) if key.query_index is None else (float(key.query_index + 1),)
        queries = [make_query(ir_slots, vis_slots, q) for q in query_times]
        target_slice = slice(1, 4) if key.query_index is None else slice(key.query_index + 1, key.query_index + 2)
        result = {'inputs': queries, 'ir_targets': ir[target_slice], 'vis_targets': vis[target_slice],
                'source': row['source'], 'sequence_id': row['sequence_id'],
                'sample_id': row['sample_id'], 'protocol': key.protocol}
        if key.query_index is not None:
            result['selected_query'] = key.query_index
        return result


def collate_clips(samples):
    if not samples:
        raise ValueError('Cannot collate empty clip list')
    selected = ['selected_query' in sample for sample in samples]
    if any(selected) and not all(selected):
        raise ValueError('Cannot mix selected and full-query samples')
    queries = 1 if all(selected) else 3
    if any(len(sample['inputs']) != queries or sample['ir_targets'].shape[0] != queries
           or sample['vis_targets'].shape[0] != queries for sample in samples):
        raise ValueError('Clip query count does not match selected/full-query format')
    result = {'inputs': [TemporalInputs.cat([s['inputs'][q] for s in samples]) for q in range(queries)],
            'ir_targets': torch.stack([s['ir_targets'] for s in samples]),
            'vis_targets': torch.stack([s['vis_targets'] for s in samples]),
            **{k: [s[k] for s in samples] for k in ('source', 'sequence_id', 'sample_id', 'protocol')}}
    if all(selected):
        result['selected_queries'] = [s['selected_query'] for s in samples]
    if all('teacher_inputs' in s for s in samples):
        if any(len(sample['teacher_inputs']) != queries for sample in samples):
            raise ValueError('Teacher query count must match observation query count')
        result['teacher_inputs'] = [TemporalInputs.cat([s['teacher_inputs'][q] for s in samples])
                                    for q in range(queries)]
    elif any('teacher_inputs' in s for s in samples):
        raise ValueError('Cannot mix samples with and without teacher context')
    return result
