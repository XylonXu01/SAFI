"""Real clocks, lazy image selection and actual FFmpeg output; no GPU use."""
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image
import pytest
import torch


def api():
    try:
        return importlib.import_module('safi_net.infer_video')
    except ModuleNotFoundError:
        pytest.fail('Offline inference implementation is missing')


def frames(root, count, size=(17, 19)):
    root.mkdir()
    for i in range(count):
        Image.fromarray(np.full((size[1], size[0], 3), i * 20, np.uint8)).save(root / f'f{i}.png')
    return root


@pytest.fixture
def video_tools():
    missing = [tool for tool in ('ffmpeg', 'ffprobe') if shutil.which(tool) is None]
    if missing:
        pytest.skip('Video integration requires external tools: ' + ', '.join(missing))


def test_folder_lazy_brackets_and_protocol_tail(tmp_path):
    m = api()
    ir = frames(tmp_path / 'ir', 5)
    vis = frames(tmp_path / 'vis', 9)
    with m.FrameSource(ir, fps=24, grayscale=True) as a, m.FrameSource(vis, fps=48) as b:
        assert len(a) == 5 and len(b) == 9
        assert a.cache_size == 0
        x, audit = m.query_inputs(a, b, 1 / 48)
        assert x.ir.shape == (1, 2, 1, 32, 32)
        assert x.vis.shape == (1, 2, 3, 32, 32)
        assert audit['ir']['indices'] == [0, 1]
        assert audit['vis']['indices'] == [0, 1]
        assert x.vis_valid.tolist() == [[True, True]]
        assert torch.equal(x.vis[0, 1], torch.full_like(x.vis[0, 1], 20 / 255))
        assert a.cache_size <= 4 and b.cache_size <= 4
        q = m.output_timeline(a.times, b.times, a.fps, b.fps)
        assert len(q) == 9 and float(q[-1]) == pytest.approx(1 / 6)
    clock = tmp_path / 'clock.json'
    clock.write_text(json.dumps([(i + 1) / 48 for i in range(9)]))
    with m.FrameSource(ir, fps=24, grayscale=True) as a, m.FrameSource(vis, fps=48, timestamps=clock) as b:
        q = m.output_timeline(a.times, b.times, a.fps, b.fps)
        assert len(q) == 8 and float(q[0]) == pytest.approx(1 / 48)


@pytest.mark.parametrize('clock', [[0, 0, 1], [0, 2, 1], [0, float('nan'), 1], [0, 1]])
def test_reject_invalid_directory_clock(tmp_path, clock):
    root = frames(tmp_path / 'images', 3)
    path = tmp_path / 'times.json'
    path.write_text(json.dumps(clock))
    with pytest.raises(ValueError):
        with api().FrameSource(root, fps=24, timestamps=path):
            pass


def test_same_rate_schedule_and_invalid_rates(tmp_path):
    m = api()
    root = frames(tmp_path / 'images', 5)
    with m.FrameSource(root, fps=24) as a:
        assert m.output_timeline(a.times, a.times, 24, 24).tolist() == pytest.approx([i / 48 for i in range(9)])
        for rate in (0, -1, 36, float('inf')):
            with pytest.raises(ValueError):
                m.output_timeline(a.times, a.times, 24, rate)
    with pytest.raises(ValueError, match='FPS|fps'):
        with m.FrameSource(root):
            pass


@pytest.mark.parametrize('missing', ['ffmpeg', 'ffprobe'])
def test_missing_video_tools_fail_before_model_or_output_creation(tmp_path, monkeypatch, missing):
    m = api()
    monkeypatch.setattr(m.shutil, 'which', lambda name: None if name == missing else '/test/bin/' + name)
    output = tmp_path / 'not-created'
    args = m.parser().parse_args([
        '--ir-input', str(tmp_path / 'ir'), '--vis-input', str(tmp_path / 'vis'),
        '--checkpoint', 'must-not-load.pt', '--output-dir', str(output),
        '--device', 'cpu', '--precision', 'float32', '--fps-ir', '12', '--fps-vis', '12'])
    def forbidden_loader(*args, **kwargs):
        raise AssertionError('Model must not load when video dependencies are missing')
    with pytest.raises(RuntimeError, match='Missing video executable.*' + missing):
        m.run_inference(args, model_loader=forbidden_loader)
    assert not output.exists()


def test_explicit_json_clock_without_fps_and_bounded_cache(tmp_path):
    m = api()
    root = frames(tmp_path / 'images', 10)
    clock = tmp_path / 'times.json'
    clock.write_text(json.dumps({'timestamps': [7 + i / 25 for i in range(10)]}))
    with m.FrameSource(root, timestamps=clock) as source:
        assert source.fps == pytest.approx(25)
        assert float(source.times[0]) == 7
        for i in range(10):
            source.read(i)
            assert source.cache_size <= 4


def test_dangling_output_symlinks_never_create_their_targets(tmp_path):
    m = api()
    source = frames(tmp_path / 'images', 2)
    target = tmp_path / 'must not be created.mp4'
    link = tmp_path / 'link.mp4'
    link.symlink_to(target)
    with pytest.raises(FileExistsError):
        m.encode_video(sorted(source.glob('*.png')), [0, .1], link)
    assert not target.exists()
    directory_target = tmp_path / 'must not be created'
    directory_link = tmp_path / 'linked output'
    directory_link.symlink_to(directory_target)
    args = m.parser().parse_args(['--ir-input', str(source), '--vis-input', str(source),
        '--checkpoint', 'unneeded.pt', '--output-dir', str(directory_link), '--device', 'cpu',
        '--precision', 'float32', '--fps-ir', '24', '--fps-vis', '24'])
    with pytest.raises(FileExistsError):
        m.run_inference(args)
    assert not directory_target.exists()


def test_actual_vfr_mp4_preserves_pts_odd_dimensions_and_source_mtime(tmp_path, video_tools):
    m = api()
    root = frames(tmp_path / 'images', 5)
    originals = [p.stat().st_mtime_ns for p in sorted(root.glob('*.png'))]
    video = tmp_path / 'vfr.mp4'
    result = m.encode_video(sorted(root.glob('*.png')), [2, 2.05, 2.13, 2.17, 2.3], video)
    # Capture the tool's actual successful-encode diagnostics for the audit.
    version = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True).stdout
    if version.startswith('ffmpeg version 4.'):
        assert 'vsync 0' in result['encoder_stderr']
    assert result['decoded_frames'] == 5
    assert result['width'] == 17 and result['height'] == 19
    assert result['pts'] == pytest.approx([0, .05, .13, .17, .3], abs=1.1e-6)
    assert [p.stat().st_mtime_ns for p in sorted(root.glob('*.png'))] == originals
    with m.FrameSource(video, fps=24) as source:
        assert len(source) == 5
        assert source.times.tolist() == pytest.approx([0, .05, .13, .17, .3], abs=1.1e-6)
        assert source.read(4).shape == (3, 19, 17)
        temp = source.staging_root
    assert not temp.exists()
    with pytest.raises(FileExistsError):
        m.encode_video(sorted(root.glob('*.png')), [0, .05, .13, .17, .3], video)
    with pytest.raises(ValueError, match='microsecond|quantiz'):
        m.encode_video(sorted(root.glob('*.png')), [0, .00000001, .13, .17, .3], tmp_path / 'collision.mp4')


@pytest.mark.parametrize('times', [[0], [0, .1], [0, .5, 1], [0, .001, .002]])
def test_video_encoder_accepts_short_clips_at_arbitrary_positive_rates(tmp_path, times, video_tools):
    root = frames(tmp_path / 'images', len(times))
    result = api().encode_video(sorted(root.glob('*.png')), times, tmp_path / 'short.mp4')
    assert result['decoded_frames'] == len(times)
    assert result['pts'] == pytest.approx(times, abs=1.1e-6)


def test_inference_real_task_layers_once_unpads_audit_and_refuses_overwrite(tmp_path, video_tools):
    from test_model import TestMotion, TestSemantic
    from safi_net.models import build_model
    m = api()
    ir = frames(tmp_path / 'ir', 3)
    vis = frames(tmp_path / 'vis', 3)
    output = tmp_path / 'out'
    calls = []

    def loader(checkpoint, pretrained_root=None, device='cpu'):
        calls.append(checkpoint)
        return build_model({'base_channels': 4, 'semantic_dim': 8, 'transport': {'radius': 1}},
                           test_backbones=(TestMotion(.1), TestMotion(-.2), TestSemantic())).eval()

    checkpoint = tmp_path / 'explicit-test-loader.pt'
    checkpoint.write_bytes(b'explicitly injected test loader')
    args = m.parser().parse_args(['--ir-input', str(ir), '--vis-input', str(vis),
        '--checkpoint', str(checkpoint), '--output-dir', str(output),
        '--fps-ir', '24', '--fps-vis', '24', '--device', 'cpu', '--precision', 'float32',
        '--save-aux', '--save-references'])
    summary = m.run_inference(args, model_loader=loader)
    assert len(calls) == 1
    assert summary['output_frames'] == 5
    assert summary['mp4']['decoded_frames'] == 5
    assert summary['mp4']['r_frame_rate'] == '48/1'
    assert summary['mp4']['pts'] == pytest.approx([i / 48 for i in range(5)], abs=1.1e-6)
    assert Image.open(output / 'fused/00000004.png').size == (17, 19)
    assert len(list((output / 'ir_aux').glob('*.png'))) == 5
    records = [json.loads(line) for line in (output / 'audit.jsonl').read_text().splitlines()]
    assert records[0]['observed'] == [True, True]
    assert records[1]['observed'] == [False, False]
    assert records[-1]['ir']['indices'] == [1, 2]
    assert json.loads((output / 'summary.json').read_text())['source_frames'] == {'ir': 3, 'vis': 3}
    with pytest.raises(FileExistsError):
        m.run_inference(args, model_loader=loader)
    assert len(calls) == 1
