"""Hand-derived contracts: incorrect selection, masks or shared clocks must fail."""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def module(name):
    try:
        return importlib.import_module('safi_net.' + name)
    except ModuleNotFoundError as exc:
        pytest.fail(f'Required foundation module is missing: {exc.name}')


def inputs(ir_times=(0., 4.), vis_times=(1., 3.), query=2.):
    return module('types').TemporalInputs(
        ir=torch.ones(1, 2, 1, 3, 3), vis=torch.ones(1, 2, 3, 3, 3),
        ir_times=torch.tensor([ir_times], dtype=torch.float64),
        vis_times=torch.tensor([vis_times], dtype=torch.float64),
        ir_valid=torch.ones(1, 2, dtype=torch.bool),
        vis_valid=torch.ones(1, 2, dtype=torch.bool),
        query_time=torch.tensor([query], dtype=torch.float64))


def test_shared_normalization_preserves_relative_intervals_and_affine_time():
    temporal = module('temporal')
    x = inputs()
    context = temporal.time_context(x)
    assert context['offsets'].tolist() == [[-.5, -.25, .5, .25]]
    assert context['spans'].tolist() == [[1., .5]]
    assert context['tau'].tolist() == [[.5, .5]]
    assert context['observed'].tolist() == [[False, False]]
    x.ir_times = x.ir_times * 13 + 91
    x.vis_times = x.vis_times * 13 + 91
    x.query_time = x.query_time * 13 + 91
    for field in ('offsets', 'spans', 'tau'):
        torch.testing.assert_close(temporal.time_context(x)[field], context[field])


def test_exact_masks_exclude_neighbor_evidence():
    temporal = module('temporal')
    context = temporal.time_context(inputs((0., 2.), (2., 4.)))
    assert context['observed'].tolist() == [[True, True]]
    assert context['effective_valid'].tolist() == [[False, True, True, False]]
    assert context['offsets'].count_nonzero() == 0
    assert context['spans'].count_nonzero() == 0
    assert context['tau'].count_nonzero() == 0


def test_same_and_fast_selection_keeps_context_but_masks_exact_neighbor():
    t = module('temporal')
    ir = torch.arange(3.).view(3, 1, 1, 1).expand(-1, 1, 3, 3)
    vis = torch.arange(5.).view(5, 1, 1, 1).expand(-1, 3, 3, 3)
    x = t.select_inputs(ir, vis, [0., 2., 4.], [0., 1., 2., 3., 4.], 3.)
    assert x.ir[:, :, 0, 0, 0].tolist() == [[1., 2.]]
    assert x.ir_times.tolist() == [[2., 4.]]
    assert x.vis[:, :, 0, 0, 0].tolist() == [[2., 3.]]
    assert x.vis_valid.tolist() == [[True, True]]
    assert t.time_context(x)['effective_valid'].tolist() == [[True, False, True, True]]
    assert t.time_context(x)['observed'].tolist() == [[False, True]]
    same = t.select_inputs(ir, vis[::2], [0., 2., 4.], [0., 2., 4.], 1.)
    assert same.vis[:, :, 0, 0, 0].tolist() == [[0., 2.]]
    assert same.vis_valid.all()
    torch.testing.assert_close(module('types').TemporalInputs.cat([x, same]).to('cpu').query_time,
                               torch.tensor([3., 1.], dtype=torch.float64))


@pytest.mark.parametrize('bad', [[0., 0.], [4., 0.], [0., float('nan')]])
def test_rejects_duplicate_unsorted_nonfinite_valid_times(bad):
    with pytest.raises(ValueError):
        module('temporal').time_context(inputs(ir_times=bad))


def test_rejects_absent_brackets_bad_shapes_and_wrong_time_dtype():
    t = module('temporal')
    with pytest.raises(ValueError):
        t.time_context(inputs(query=7.))
    x = inputs()
    x.ir_times = x.ir_times.float()
    with pytest.raises(ValueError):
        t.time_context(x)
    x = inputs()
    x.vis = x.vis[:, :, :1]
    with pytest.raises(ValueError):
        t.time_context(x)
    with pytest.raises(ValueError):
        t.select_observations(torch.ones(2, 1, 2, 2), [0., 1.], -.1)


def test_timeline_has_no_extrapolation_or_tail_duplication():
    t = module('temporal')
    assert t.output_timeline([0., 2., 4.], [0., 2., 4.], 5., 5.).tolist() == [0., 1., 2., 3., 4.]
    assert t.output_timeline([0., 2., 4.], [1., 2., 3., 4., 5.], 5., 10.).tolist() == [1., 2., 3., 4.]
    for args in (([0., 1.], [2., 3.], 1., 1.), ([0., 1.], [0., 1.], 1., 3.)):
        with pytest.raises(ValueError):
            t.output_timeline(*args)


def write_sequence(root, split, sequence, values=range(7), lineage=None):
    for modality in ('infrared', 'visible'):
        directory = root / split / sequence / modality
        directory.mkdir(parents=True)
        for i in values:
            pixels = np.arange(36, dtype=np.uint8).reshape(6, 6) + i * 10
            Image.fromarray(pixels).save(directory / f'{i:03}.png')
    return {'split': split, 'sequence_id': sequence,
            'source': {'hdo': 'HDO', 'm3': 'M3SVD', 'vt': 'VTMOT'}[sequence.split('_')[0]],
            'visible_source': f'/old/data/{lineage or sequence}/visible',
            'infrared_source': f'/old/data/{lineage or sequence}/infrared'}


def make_dataset(tmp_path):
    rows = []
    for prefix in ('hdo', 'm3', 'vt'):
        for i in range(10):
            rows.append(write_sequence(tmp_path, 'train', f'{prefix}_{i}'))
        rows.append(write_sequence(tmp_path, 'test', f'{prefix}_held'))
    rows.append(write_sequence(tmp_path, 'train', 'm3_alias', lineage='m3_held'))
    ann = tmp_path / 'annotations'
    ann.mkdir()
    (ann / 'sequences.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return module('split_manifest').build_manifest(tmp_path)


def test_splits_remove_lineages_and_have_stratified_deterministic_validation(tmp_path):
    manifest = make_dataset(tmp_path)
    assert manifest == module('split_manifest').build_manifest(tmp_path)
    assert len(manifest['splits']['test']) == 3
    assert len(manifest['splits']['val']) == 3
    assert len(manifest['splits']['train']) == 27
    assert {r['source'] for r in manifest['splits']['val']} == {'HDO', 'M3SVD', 'VTMOT'}
    assert manifest['excluded_overlap'][0]['sequence_id'] == 'm3_alias'
    assert '/old/data' not in json.dumps(manifest)
    assert len(manifest['fingerprint']) == 64
    assert manifest['counts']['train']['windows'] == 81


def test_strict_windows_and_missing_pair_rejected(tmp_path):
    write_sequence(tmp_path, 'train', 'hdo_a', values=(0, 1, 2, 4, 6, 8, 10))
    m = module('split_manifest').build_manifest(tmp_path)
    all_records = m['splits']['train'] + m['splits']['val']
    assert all_records[0]['window_starts'] == [2]
    (tmp_path / 'train/hdo_a/infrared/004.png').unlink()
    with pytest.raises(ValueError, match='pair'):
        module('split_manifest').build_manifest(tmp_path)


def test_unusable_train_sequences_are_excluded_before_split_but_test_membership_kept(tmp_path):
    write_sequence(tmp_path, 'train', 'hdo_short', values=(0, 1, 2, 4))
    write_sequence(tmp_path, 'train', 'hdo_valid')
    write_sequence(tmp_path, 'test', 'hdo_short_test', values=(0, 1))
    m = module('split_manifest').build_manifest(tmp_path)
    assert [r['sequence_id'] for r in m['splits']['train']] == ['hdo_valid']
    assert m['splits']['val'] == []
    assert [r['sequence_id'] for r in m['excluded_no_windows']] == ['hdo_short']
    assert [r['sequence_id'] for r in m['splits']['test']] == ['hdo_short_test']


def test_one_exact_branch_preserves_other_span_and_rejects_empty_masks():
    t = module('temporal')
    context = t.time_context(inputs((0., 2.), (1., 3.)))
    assert context['effective_valid'].tolist() == [[False, True, True, True]]
    assert context['offsets'].tolist() == [[0., -.5, 0., .5]]
    assert context['spans'].tolist() == [[0., 1.]]
    assert context['tau'].tolist() == [[0., .5]]
    x = inputs()
    x.ir_valid[:] = False
    with pytest.raises(ValueError):
        t.time_context(x)


def test_hidden_targets_never_change_inputs_and_fast_vis_is_observed(tmp_path):
    manifest = make_dataset(tmp_path)
    data = module('data')
    dataset = data.AsyncClipDataset(tmp_path, manifest, 'train', crop_size=4, training=False)
    sample = dataset[data.SampleKey(0, 'vis_fast', 123)]
    assert sample['ir_targets'].shape == (3, 1, 4, 4)
    assert [x.query_time.item() for x in sample['inputs']] == [1., 2., 3.]
    assert [module('temporal').time_context(x)['observed'].tolist()[0] for x in sample['inputs']] == [
        [False, True], [True, True], [False, True]]
    first = dataset.windows[0]
    for index in (1, 3):
        path = tmp_path / first['ir'][index]
        Image.fromarray(np.full((6, 6), 255, dtype=np.uint8)).save(path)
    changed = dataset[data.SampleKey(0, 'vis_fast', 123)]
    for left, right in zip(sample['inputs'], changed['inputs']):
        torch.testing.assert_close(left.ir, right.ir)
        torch.testing.assert_close(left.vis, right.vis)
        torch.testing.assert_close(left.ir_times, right.ir_times)
        torch.testing.assert_close(left.vis_times, right.vis_times)
        assert left.motion_context is right.motion_context is None
    assert not torch.equal(sample['ir_targets'], changed['ir_targets'])
    batch = data.collate_clips([sample, changed])
    assert batch['ir_targets'].shape == (2, 3, 1, 4, 4)
    assert batch['inputs'][0].ir.shape == (2, 2, 1, 4, 4)


def test_balanced_sampling_is_stateless_and_augmentations_repeat(tmp_path):
    data = module('data')
    dataset = data.AsyncClipDataset(tmp_path, make_dataset(tmp_path), 'train', crop_size=4)
    keys = [data.balanced_sample_key(dataset, i) for i in range(60)]
    assert keys == [data.balanced_sample_key(dataset, i) for i in range(60)]
    for start in range(0, 60, 6):
        pairs = {(dataset.windows[k.window_index]['source'], k.protocol) for k in keys[start:start+6]}
        assert pairs == {(s, p) for s in ('HDO', 'M3SVD', 'VTMOT') for p in ('same', 'vis_fast')}
    for key in keys[:5]:
        a, b = dataset[key], dataset[key]
        torch.testing.assert_close(a['ir_targets'], b['ir_targets'])
        torch.testing.assert_close(a['vis_targets'], b['vis_targets'])
        # All frames share crop/flip, with a constant per-frame intensity increment.
        torch.testing.assert_close(a['vis_targets'][:, :1], a['ir_targets'])
        for first, second in zip(a['inputs'], b['inputs']):
            for name in ('ir', 'vis', 'ir_valid', 'vis_valid', 'ir_times', 'vis_times'):
                torch.testing.assert_close(getattr(first, name), getattr(second, name))



def test_portable_paths_do_not_depend_on_cwd_and_reject_escape(tmp_path, monkeypatch):
    paths = module('paths')
    monkeypatch.chdir(tmp_path)
    roots = paths.resolve_paths(tmp_path)
    assert roots.pretrained_root == paths.PACKAGE_ROOT / 'pretrained'
    assert roots.run_dir == tmp_path / 'runs'
    assert paths.relative_path(tmp_path, 'train/a.png') == tmp_path / 'train/a.png'
    for invalid in ('../escape', '/absolute/path'):
        with pytest.raises(ValueError):
            paths.relative_path(tmp_path, invalid)


def test_manifest_cli_writes_relative_fingerprinted_output_only_outside_dataset(tmp_path):
    root = tmp_path / 'data'
    expected = make_dataset(root)
    output = tmp_path / 'manifest.json'
    before = sorted(str(p.relative_to(root)) for p in root.rglob('*'))
    command = [sys.executable, '-m', 'safi_net.split_manifest', '--data-root', str(root), '--output']
    result = subprocess.run(command + [str(output)], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert output.is_file(), 'CLI must emit the requested manifest'
    assert json.loads(output.read_text()) == expected
    assert json.loads(result.stdout)['fingerprint'] == expected['fingerprint']
    rejected = subprocess.run(command + [str(root / 'forbidden.json')], capture_output=True)
    assert rejected.returncode != 0
    assert before == sorted(str(p.relative_to(root)) for p in root.rglob('*'))
