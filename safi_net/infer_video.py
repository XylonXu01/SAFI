"""Offline fusion on independent real clocks; FFmpeg preserves source frame PTS."""
import argparse
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import replace
from fractions import Fraction
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .checkpoints import load_inference_model
from .inference_policy import add_segmentation_argument, configure_segmentation
from .motion_context import build_motion_context
from .temporal import _clock, observation_indices, output_timeline
from .types import TemporalInputs


def require_video_tools():
    """Fail before model inference if MP4 encoding/verification is unavailable."""
    missing = [name for name in ('ffmpeg', 'ffprobe') if shutil.which(name) is None]
    if missing:
        raise RuntimeError(
            'Missing video executable(s): ' + ', '.join(missing)
            + '. Install ffmpeg (including ffprobe) in the active environment and make both '
              'available on PATH before inference. If fused/*.png and audit.jsonl already '
              'exist, preserve them: video encoding can be retried without model inference.')


def _run(command, *, with_stderr=False):
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'{command[0]} failed: {result.stderr.strip()}')
    return (result.stdout, result.stderr) if with_stderr else result.stdout


def probe_video(path):
    info = json.loads(_run(['ffprobe', '-v', 'error', '-threads', '1', '-select_streams', 'v:0',
        '-show_streams', '-show_frames', '-show_entries',
        'stream=width,height,time_base,r_frame_rate,avg_frame_rate:frame=pts,pkt_pts,best_effort_timestamp',
        '-of', 'json', str(path)]))
    if not info.get('streams'):
        raise ValueError(f'Video has no video stream: {path}')
    stream = info['streams'][0]
    base = Fraction(stream['time_base'])
    times = []
    for frame in info.get('frames', []):
        tick = next((frame[k] for k in ('pts', 'pkt_pts', 'best_effort_timestamp') if k in frame), None)
        if tick is None:
            raise ValueError('Each video frame requires an actual container PTS')
        times.append(float(int(tick) * base))
    _clock(times)
    try:
        rate = float(Fraction(stream.get('r_frame_rate', '0/1')))
    except (ValueError, ZeroDivisionError):
        rate = 0.
    return {'times': times, 'fps': rate, 'width': stream['width'], 'height': stream['height'],
            'time_base': str(base), 'r_frame_rate': stream.get('r_frame_rate'),
            'avg_frame_rate': stream.get('avg_frame_rate')}


def _natural(path):
    return tuple((1, int(s)) if s.isdigit() else (0, s.lower()) for s in re.split(r'(\d+)', path.name))


class FrameSource:
    """A bounded image cache; video frames are decoded once into owned temp files."""
    def __init__(self, path, fps=None, timestamps=None, grayscale=False):
        self.path = Path(path).resolve()
        self.fps = fps
        self.timestamps = timestamps
        self.grayscale = grayscale
        self._cache = OrderedDict()
        self._temp = None
        self.staging_root = None
        self.metadata = None

    def __enter__(self):
        try:
            if self.path.is_dir():
                self.paths = sorted((p for p in self.path.iterdir()
                    if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}), key=_natural)
                if not self.paths:
                    raise ValueError('Image directory is empty')
                if self.timestamps:
                    payload = json.loads(Path(self.timestamps).read_text())
                    values = payload['timestamps'] if isinstance(payload, dict) else payload
                    self.times = _clock(values)
                    if self.fps is None and len(self.times) > 1:
                        self.fps = 1 / float((self.times[1:] - self.times[:-1]).median())
                else:
                    if self.fps is None or not math.isfinite(self.fps) or self.fps <= 0:
                        raise ValueError('Directory inputs require positive FPS or explicit timestamps JSON')
                    self.times = torch.arange(len(self.paths), dtype=torch.float64) / self.fps
            else:
                if self.timestamps:
                    raise ValueError('Video timestamps come from container PTS; JSON overrides are directory-only')
                self.metadata = probe_video(self.path)
                self.times = _clock(self.metadata['times'])
                if self.fps is None:
                    self.fps = self.metadata['fps']
                self._temp = tempfile.TemporaryDirectory(prefix='safi-decode-')
                self.staging_root = Path(self._temp.name)
                _run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-filter_threads', '1',
                      '-threads', '1', '-i', str(self.path),
                      '-map', '0:v:0', '-vsync', '0', '-threads', '1',
                      str(self.staging_root / '%08d.png')])
                self.paths = sorted(self.staging_root.glob('*.png'))
            if len(self.times) != len(self.paths):
                raise ValueError('Decoded source frame count and timestamp count differ')
            if self.fps is None or not math.isfinite(self.fps) or self.fps <= 0:
                raise ValueError('Positive nominal FPS required; declare --fps-ir/--fps-vis for ambiguous video metadata')
            _clock(self.times)
            with Image.open(self.paths[0]) as image:
                self.size = image.size
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        self._cache.clear()
        if self._temp is not None:
            self._temp.cleanup()

    def __len__(self):
        return len(self.paths)

    @property
    def cache_size(self):
        return len(self._cache)

    def read(self, index):
        if index not in self._cache:
            with Image.open(self.paths[index]) as image:
                if image.size != self.size:
                    raise ValueError('Source frame spatial dimensions changed')
                values = np.asarray(image.convert('L' if self.grayscale else 'RGB')).copy()
            if self.grayscale:
                values = values[..., None]
            self._cache[index] = torch.from_numpy(values).permute(2, 0, 1).float() / 255
            if len(self._cache) > 4:
                self._cache.popitem(last=False)
        self._cache.move_to_end(index)
        return self._cache[index]


def _observations(source, query):
    if query < source.times[0] or query > source.times[-1]:
        raise ValueError('Query lies outside source support')
    indices = observation_indices(source.times, query)
    first = source.read(indices[0])
    images = torch.stack([first, torch.zeros_like(first) if indices[1] is None else source.read(indices[1])])
    h, w = images.shape[-2:]
    images = F.pad(images, (0, (-w) % 32, 0, (-h) % 32), mode='replicate')
    times = [float(source.times[i]) if i is not None else 0. for i in indices]
    valid = [i is not None for i in indices]
    audit = {'indices': indices, 'times': times, 'valid': valid,
             'files': [source.paths[i].name if i is not None else None for i in indices]}
    return images[None], torch.tensor([times], dtype=torch.float64), torch.tensor([valid]), audit


def query_inputs(ir, vis, query):
    if ir.size != vis.size:
        raise ValueError('IR/VIS must have identical registered spatial dimensions')
    a, at, av, aa = _observations(ir, query)
    b, bt, bv, ba = _observations(vis, query)
    return TemporalInputs(a, b, at, bt, av, bv, torch.tensor([query], dtype=torch.float64)), {'ir': aa, 'vis': ba}


def encode_video(paths, times, output, fps=None):
    """Encode staged copies with nanosecond image2 clocks, then verify full decode."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Refusing to overwrite: {output}')
    require_video_tools()
    times = _clock(times).tolist()
    if len(times) != len(paths):
        raise ValueError('Output PNG and timestamp count differ')
    normalized = [t - times[0] for t in times]
    ticks = [round(t * 1_000_000) for t in normalized]
    if any(b <= a for a, b in zip(ticks, ticks[1:])):
        raise ValueError('Output timestamps collide after microsecond quantization')
    # Give the final packet a duration while preserving the source PTS.
    packet_rate = fps if fps is not None else (1 / (times[-1] - times[-2]) if len(times) > 1 else 25.)
    if not math.isfinite(packet_rate) or packet_rate <= 0:
        raise ValueError('Output nominal FPS must be positive and finite')
    setts_options = _run(['ffmpeg', '-hide_banner', '-h', 'bsf=setts'])
    if re.search(r'^\s+-duration\s', setts_options, re.MULTILINE):
        duration_options = ['-bsf:v', f'setts=pts=PTS:dts=DTS:duration=1/({packet_rate}*TB)']
    else:
        # FFmpeg 4.4 derives the last duration from the output rate in passthrough mode.
        duration_options = ['-r', str(packet_rate)]
    with tempfile.TemporaryDirectory(prefix='safi-encode-') as directory:
        directory = Path(directory)
        for i, path in enumerate(paths):
            target = directory / f'{i:08d}.png'
            shutil.copyfile(path, target)
            ns = 1_000_000_000 + round(normalized[i] * 1_000_000_000)
            os.utime(target, ns=(ns, ns))
            if target.stat().st_mtime_ns != ns:
                raise ValueError('Temporary filesystem does not retain nanosecond timestamps')
        _, encoder_stderr = _run(['ffmpeg', '-v', 'warning', '-nostdin', '-n', '-filter_threads', '1',
              '-threads', '1', '-f', 'image2',
              '-framerate', str(packet_rate), '-ts_from_file', '2',
              '-i', str(directory / '%08d.png'), '-vf', 'setpts=PTS-STARTPTS', '-vsync', '0',
              '-c:v', 'libx264', '-threads', '1', '-x264-params',
              f'fps={Fraction(str(packet_rate)).limit_denominator(1000000)}',
              '-pix_fmt', 'yuv444p', '-enc_time_base', '1:1000000',
              *duration_options,
              '-video_track_timescale', '1000000', str(output)], with_stderr=True)
    with FrameSource(output, fps=1) as decoded:
        if len(decoded) != len(paths):
            raise ValueError('Encoded MP4 frame count differs from requested count')
        with Image.open(paths[0]) as first:
            expected_size = first.size
        for index in range(len(decoded)):
            if tuple(reversed(decoded.read(index).shape[-2:])) != expected_size:
                raise ValueError('Encoded MP4 dimensions differ')
        errors = [abs(a - b) for a, b in zip(decoded.times.tolist(), normalized)]
        error = max(errors)
        if error > 1.1e-6:
            raise ValueError(f'Encoded MP4 PTS mismatch: maximum error {error}')
        return {'decoded_frames': len(decoded), 'pts': decoded.times.tolist(),
                'width': decoded.size[0], 'height': decoded.size[1],
                'zero_origin_offset': times[0], 'max_pts_error_seconds': error,
                'encoder_stderr': encoder_stderr.strip(),
                'r_frame_rate': decoded.metadata['r_frame_rate'],
                'avg_frame_rate': decoded.metadata['avg_frame_rate']}


def _save_image(tensor, path, size):
    w, h = size
    pixels = tensor[0, :, :h, :w].detach().float().clamp(0, 1)
    array = (pixels.permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    Image.fromarray(array).save(path)
    with Image.open(path) as image:
        image.load()
        if image.size != size:
            raise ValueError('Written PNG dimensions differ')


def apply_fusion_region(prediction, region):
    """Inference-only composition override; keep the original tensors intact."""
    if region == 'target':
        return prediction
    if region != 'all':
        raise ValueError('fusion_region must be target or all')
    if getattr(prediction, 'fusion_representation', 'thermal_delta') == 'direct_rgb':
        fused = prediction.fused
        residual = prediction.fusion_delta
        if (not torch.isfinite(fused).all() or (fused < 0).any() or (fused > 1).any()
                or residual.shape != fused.shape
                or not torch.allclose(residual, fused - prediction.vis_reference, atol=1e-6, rtol=1e-5)):
            raise ValueError('Invalid direct RGB decoder output')
        return prediction
    reference = getattr(prediction, 'vis_reference', None)
    delta = getattr(prediction, 'pre_gate_delta', None)
    gate = getattr(prediction, 'target_gate', None)
    if not all(isinstance(value, torch.Tensor) for value in (reference, delta, gate)):
        raise ValueError('Full-image fusion requires vis_reference, pre_gate_delta and target_gate tensors')
    if (reference.ndim != 4 or reference.shape[1] != 3
            or delta.shape != (reference.shape[0], 1, *reference.shape[2:])
            or gate.shape != delta.shape):
        raise ValueError('Full-image fusion requires RGB reference and matching single-channel delta/gate shapes')
    if any(value.device != reference.device or not value.is_floating_point() for value in (delta, gate)):
        raise ValueError('Full-image fusion tensors must be floating point on the same device')
    if not reference.is_floating_point():
        raise ValueError('Full-image fusion reference must be floating point')
    if not all(torch.isfinite(value).all() for value in (reference, delta, gate)):
        raise ValueError('Full-image fusion tensors must be finite')
    # The decoder has already bounded this learned increment. Do not replace it
    # with a hand-designed IR blend or apply the model's target gate again.
    fused = reference + delta
    if (delta < 0).any() or (delta > .25).any() or (fused < 0).any() or (fused > 1).any():
        raise ValueError('Full-image fusion requires a bounded nonnegative pre_gate_delta and valid RGB range')
    return replace(prediction, fused=fused, target_gate=torch.ones_like(gate), fusion_delta=delta)


def _checkpoint_identity(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
        after = os.fstat(handle.fileno())
    signature = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if signature(before) != signature(after) or signature(after) != signature(path.stat()):
        raise RuntimeError('Checkpoint changed while being identified; use a fixed checkpoint copy')
    return {'path': str(path.resolve()), 'sha256': digest.hexdigest(), 'bytes': after.st_size}, signature(after)


def _tensor_statistics(value):
    value = value.detach().float()
    return {'mean': float(value.mean()), 'min': float(value.min()), 'max': float(value.max()),
            'nonzero_fraction': float((value != 0).float().mean())}


def run_inference(args, *, model_loader=None):
    requested_output = Path(args.output_dir).absolute()
    if requested_output.exists() or requested_output.is_symlink():
        raise FileExistsError(f'Output directory must not exist: {requested_output}')
    output = requested_output.resolve()
    if output.exists():
        raise FileExistsError(f'Output directory must not exist: {output}')
    require_video_tools()
    fusion_region = getattr(args, 'fusion_region', 'all')
    if fusion_region not in ('target', 'all'):
        raise ValueError('fusion_region must be target or all')
    fast = getattr(args, 'fast', False)
    precision = args.precision or ('float32' if fast else 'bf16')
    if fast and precision != 'float32':
        raise ValueError('Fast inference requires float32')
    if model_loader is None:
        if fast:
            from .fast import load_fast_model
            model_loader = partial(load_fast_model, return_aux=args.save_aux)
        else:
            model_loader = load_inference_model
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; explicit --device cpu --precision float32 is supported')
        if precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('Visible CUDA device does not support bf16; select --precision float32')
    elif device.type != 'cpu' or precision != 'float32':
        raise ValueError('CPU execution requires explicit --device cpu --precision float32')
    with ExitStack() as stack:
        ir = stack.enter_context(FrameSource(args.ir_input, args.fps_ir, args.timestamps_ir, grayscale=True))
        vis = stack.enter_context(FrameSource(args.vis_input, args.fps_vis, args.timestamps_vis))
        if ir.size != vis.size:
            raise ValueError('IR/VIS spatial dimensions differ; inputs must already be registered')
        try:
            timeline = output_timeline(ir.times, vis.times, ir.fps, vis.fps).tolist()
        except ValueError as exc:
            raise ValueError(f'{exc}; video VFR nominal protocol may need --fps-ir and --fps-vis declarations') from exc
        if not timeline:
            raise ValueError('Common support contains no output query')
        # Load only once, after validating clocks, before creating output artifacts.
        checkpoint_identity, checkpoint_signature = _checkpoint_identity(args.checkpoint)
        model = model_loader(args.checkpoint, pretrained_root=args.pretrained_root, device=device).eval()
        loaded_stat = Path(args.checkpoint).stat()
        if (loaded_stat.st_dev, loaded_stat.st_ino, loaded_stat.st_size, loaded_stat.st_mtime_ns) != checkpoint_signature:
            raise RuntimeError('Checkpoint changed during model loading; use a fixed checkpoint copy')
        segmentation = configure_segmentation(model, getattr(args, 'segmentation', None))
        output.mkdir(parents=True, exist_ok=False)
        fields = ['fused']
        if args.save_aux:
            fields += ['ir_aux', 'vis_aux']
        if args.save_references:
            fields += ['ir_base', 'vis_base']
        mask_fields = ('person_mask', 'vehicle_mask', 'semantic_mask', 'ir_motion_mask', 'vis_motion_mask',
                       'motion_mask', 'target_gate', 'model_target_gate')
        diagnostic_fields = (*mask_fields, 'fusion_delta', 'masked_fused', 'vis_reference', 'pre_gate_delta')
        save_diagnostics = getattr(args, 'save_diagnostics', False)
        if save_diagnostics:
            fields += list(diagnostic_fields)
        for field in fields:
            (output / field).mkdir()
        # This cache belongs to one input stream only; it is never model state.
        gate_cache = {'max_alignment_pairs': 8}
        with (output / 'audit.jsonl').open('x') as audit_file, torch.inference_mode():
            for index, query in enumerate(timeline):
                inputs, audit = query_inputs(ir, vis, query)
                w, h = ir.size
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=device.type == 'cuda' and precision == 'bf16'):
                    model_prediction = model(inputs.to(device), gate_cache=gate_cache)
                if fusion_region != 'all':
                    raise ValueError('SAFI-Net only supports trained full-image fusion')
                prediction = model_prediction
                exported = {field: getattr(prediction, field) for field in fields
                            if field not in ('model_target_gate', 'masked_fused')}
                if save_diagnostics:
                    exported.update(model_target_gate=model_prediction.target_gate, masked_fused=model_prediction.fused)
                for field in fields:
                    tensor = exported[field]
                    if tensor is None:
                        raise ValueError(f'Model does not provide requested diagnostic: {field}')
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f'Nonfinite model output: {field}, frame {index}')
                    _save_image(tensor, output / field / f'{index:08d}.png', ir.size)
                record = {**segmentation, 'fusion_region': fusion_region,
                          'checkpoint_sha256': checkpoint_identity['sha256'],
                          'index': index, 'query_time': query, **audit,
                          'observed': prediction.observed[0].cpu().tolist(),
                          'fused': f'fused/{index:08d}.png'}
                if save_diagnostics:
                    from .runtime import json_ready
                    w, h = ir.size
                    gate = prediction.target_gate[..., :h, :w]
                    delta = prediction.fused[..., :h, :w] - prediction.vis_reference[..., :h, :w]
                    outside_pixels = gate <= 0
                    outside = outside_pixels.expand_as(delta)
                    record['diagnostics'] = {
                        'files': {field: f'{field}/{index:08d}.png' for field in diagnostic_fields},
                        'coverage': {field: float(exported[field][..., :h, :w].float().mean())
                                     for field in mask_fields},
                        'statistics': {field: _tensor_statistics(exported[field][..., :h, :w])
                                       for field in ('fusion_delta', 'masked_fused', 'vis_reference', 'pre_gate_delta')},
                        'outside_gate_pixels': int(outside_pixels.sum()),
                        'outside_gate_max_abs': float(delta.abs()[outside].max()) if outside.any() else 0.,
                        'gate_diagnostics_scope': 'model_target_gate',
                        'gate': json_ready(prediction.gate_diagnostics),
                    }
                audit_file.write(json.dumps(record, allow_nan=False) + '\n')
        mp4 = encode_video(sorted((output / 'fused').glob('*.png')), timeline, output / 'fused.mp4', fps=2 * ir.fps)
        summary = {'architecture': getattr(model, 'architecture_id', 'safi_net'), **segmentation,
                   'fusion_region': fusion_region, 'checkpoint': checkpoint_identity,
                   'source_paths': {'ir': str(ir.path), 'vis': str(vis.path)},
                   'source_frames': {'ir': len(ir), 'vis': len(vis)},
                   'nominal_fps': {'ir': ir.fps, 'vis': vis.fps}, 'output_frames': len(timeline),
                   'output_times': timeline, 'source_times': {'ir': ir.times.tolist(), 'vis': vis.times.tolist()},
                   'width': ir.size[0], 'height': ir.size[1], 'mp4': mp4,
                   'device': str(device), 'precision': precision, 'fast': fast,
                   'saved_diagnostics': save_diagnostics,
                   'source_video_metadata': {'ir': ir.metadata, 'vis': vis.metadata}}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('ir-input', 'vis-input', 'checkpoint', 'output-dir'):
        result.add_argument('--' + name, required=True)
    result.add_argument('--pretrained-root')
    for modality in ('ir', 'vis'):
        result.add_argument('--fps-' + modality, type=float)
        result.add_argument('--timestamps-' + modality)
    result.add_argument('--device', default='cuda:0')
    add_segmentation_argument(result)
    result.add_argument('--fusion-region', choices=('all',), default='all',
                        help='use the trained full-image fusion output')
    result.add_argument('--fast', action='store_true', help='use the FP32 inference runtime')
    result.add_argument('--precision', choices=('bf16', 'float32'),
                        help='default: float32 with --fast, bf16 otherwise')
    result.add_argument('--save-aux', action='store_true')
    result.add_argument('--save-references', action='store_true')
    result.add_argument('--save-diagnostics', action='store_true',
                        help='Save semantic and motion masks, final gate and applied brightness PNGs with audit statistics')
    return result


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    args.precision = args.precision or ('float32' if args.fast else 'bf16')
    if args.fast and args.precision != 'float32':
        cli.error('--fast requires --precision float32')
    print(json.dumps(run_inference(args), allow_nan=False))


if __name__ == '__main__':
    main()
