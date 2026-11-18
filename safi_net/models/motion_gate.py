"""Deterministic full-field independent motion, with separate IR/VIS confidence.

This module owns no flow weights and never consumes a supervised target.  The
caller supplies frozen flow backbones and observation-only, letterboxed context.
"""
from collections import OrderedDict
from collections.abc import Mapping
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..geometry import flow_warp
from ..motion_context import ensure_motion_context, project_context_map


DEFAULT_MOTION_CONFIG = {
    'longest_side': 512, 'flow_microbatch': 8,
    'photometric_threshold': 0.10, 'texture_threshold': 0.01,
    'sample_stride': 8, 'min_points': 64, 'min_inlier_ratio': 0.70,
    'min_spatial_cells': 8, 'min_spatial_span': 0.60,
    'ransac_threshold': 1.0, 'ransac_iterations': 1000,
    'ransac_confidence': 0.99, 'ransac_refine_iterations': 10,
    'residual_floor': 1.5, 'min_component_area': 16,
    'max_motion_fraction': 0.35,
}


def motion_configuration(config=None):
    if config is not None and not isinstance(config, Mapping):
        raise ValueError('Motion configuration must be a mapping')
    config = {} if config is None else dict(config)
    unknown = set(config) - set(DEFAULT_MOTION_CONFIG)
    if unknown:
        raise ValueError(f'Unknown motion configuration: {sorted(unknown)}')
    result = {**DEFAULT_MOTION_CONFIG, **config}
    for name, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Motion {name} must be a finite number')
        if isinstance(DEFAULT_MOTION_CONFIG[name], int):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f'Motion {name} must be a positive integer')
        elif value <= 0:
            raise ValueError(f'Motion {name} must be positive')
    for name in ('min_inlier_ratio', 'min_spatial_span', 'ransac_confidence', 'max_motion_fraction'):
        if result[name] > 1:
            raise ValueError(f'Motion {name} cannot exceed one')
    if result['ransac_confidence'] >= 1:
        raise ValueError('Motion ransac_confidence must be strictly below one')
    if result['min_spatial_cells'] > 16:
        raise ValueError('Motion min_spatial_cells cannot exceed 16')
    if result['longest_side'] != 512:
        raise ValueError('v6 motion context uses a fixed 512-pixel letterbox')
    return result


def _fb_valid(first, reverse):
    warped, inside = flow_warp(reverse, first)
    error = (first + warped).square().sum(1, keepdim=True)
    scale = first.square().sum(1, keepdim=True) + warped.square().sum(1, keepdim=True)
    return inside * (error <= 0.01 * scale + 0.5)


def _texture(image):
    weights = image.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]
    gray = (image * weights).sum(1, keepdim=True)
    mean = F.avg_pool2d(gray, 7, stride=1, padding=3, count_include_pad=False)
    square = F.avg_pool2d(gray.square(), 7, stride=1, padding=3, count_include_pad=False)
    return (square - mean.square()).clamp_min(0).sqrt()


def _report(reason='no_context'):
    return {'camera_ok': False, 'reason': reason, 'sample_points': 0,
            'inlier_ratio': 0., 'motion_fraction': 0., 'support_fraction': 0.}


class MotionEvidence(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = motion_configuration(config)
        # Coordinate templates are ephemeral, have no weights, and never enter
        # state_dict. A small bound also covers CPU fallback / device changes.
        self._coordinate_cache = OrderedDict()

    def _coordinates(self, value):
        h, w = value.shape[-2:]
        key = (h, w, str(value.device))
        if key not in self._coordinate_cache:
            y, x = torch.meshgrid(torch.arange(h, device=value.device),
                                  torch.arange(w, device=value.device), indexing='ij')
            self._coordinate_cache[key] = torch.stack((x, y, torch.ones_like(x)), 0).float()
            while len(self._coordinate_cache) > 4:
                self._coordinate_cache.popitem(last=False)
        self._coordinate_cache.move_to_end(key)
        return self._coordinate_cache[key]

    def _camera_scores_batch(self, forward, backward, reliable0, reliable1,
                             spatial_valid, time_scales, finite):
        """Keep the original per-pair OpenCV decisions, batch device transfers.

        One download supplies RANSAC samples, a second compact download supplies
        validity/bounds/statistics, and a third supplies all connected-component
        bitmaps. No scalar CUDA reads occur inside either CPU per-pair loop.
        """
        import cv2

        cfg = self.config
        batch = len(forward)
        h, w = forward.shape[-2:]
        stride = cfg['sample_stride']
        yy, xx = np.mgrid[0:h:stride, 0:w:stride]
        points = np.stack((xx, yy), -1).reshape(-1, 2).astype(np.float32)
        sampled = torch.cat((reliable0[..., ::stride, ::stride],
                             forward[..., ::stride, ::stride]), 1).cpu().numpy()
        inside = spatial_valid[:, 0] > .5
        occupied_x, occupied_y = inside.any(1), inside.any(2)
        xs, ys = torch.arange(w, device=forward.device), torch.arange(h, device=forward.device)
        bounds = torch.stack((torch.where(occupied_x, xs, w).amin(1),
                              torch.where(occupied_y, ys, h).amin(1),
                              torch.where(occupied_x, xs, -1).amax(1),
                              torch.where(occupied_y, ys, -1).amax(1)), 1).float()
        area = spatial_valid.flatten(1).sum(1).clamp_min(1)
        support_fraction = torch.maximum(reliable0, reliable1).flatten(1).sum(1) / area
        metadata = torch.cat((finite[:, None].float(), bounds, area[:, None],
                              support_fraction[:, None]), 1).cpu().numpy()
        transforms = np.zeros((batch, 2, 2, 3), dtype=np.float64)
        thresholds = np.full(batch, cfg['residual_floor'], dtype=np.float64)
        reports, accepted = [], []
        for row in range(batch):
            report = _report('insufficient_points' if metadata[row, 0] else 'nonfinite_flow')
            reports.append(report)
            if not metadata[row, 0]:
                continue
            selected = sampled[row, 0].reshape(-1) > 0
            vectors = sampled[row, 1:].transpose(1, 2, 0).reshape(-1, 2)
            selected &= np.isfinite(vectors).all(1)
            source, target = points[selected], (points + vectors)[selected]
            report['sample_points'] = len(source)
            if len(source) < cfg['min_points']:
                continue
            # Reset for EVERY pair; neither batch size nor earlier fit failures
            # may alter OpenCV's sample sequence.
            cv2.setRNGSeed(3407)
            matrix, inliers = cv2.estimateAffine2D(
                source, target, method=cv2.RANSAC,
                ransacReprojThreshold=cfg['ransac_threshold'], maxIters=cfg['ransac_iterations'],
                confidence=cfg['ransac_confidence'], refineIters=cfg['ransac_refine_iterations'])
            report['reason'] = 'camera_fit_failed'
            if matrix is None or inliers is None or not np.isfinite(matrix).all():
                continue
            inliers = inliers.reshape(-1).astype(bool)
            report['inlier_ratio'] = float(inliers.mean())
            if report['inlier_ratio'] < cfg['min_inlier_ratio'] or abs(np.linalg.det(matrix[:, :2])) < 1e-8:
                continue
            lower = metadata[row, 1:3].astype(np.int64)
            upper = metadata[row, 3:5].astype(np.int64)
            if np.any(upper < lower):
                report['reason'] = 'empty_spatial_context'
                continue
            extent = np.maximum(upper - lower, 1)
            support = source[inliers]
            cells = np.clip(((support - lower) / (extent + 1) * 4).astype(int), 0, 3)
            coverage = len(np.unique(cells[:, 0] + cells[:, 1] * 4))
            span = np.ptp(support, axis=0) / extent
            report.update(reason='insufficient_camera_coverage', spatial_cells=coverage,
                          spatial_span=span.tolist())
            if coverage < cfg['min_spatial_cells'] or np.any(span < cfg['min_spatial_span']):
                continue
            sample_error = np.linalg.norm(source @ matrix[:, :2].T + matrix[:, 2] - target, axis=1)[inliers]
            sample_error *= time_scales[row]
            median = float(np.median(sample_error))
            mad = float(np.median(np.abs(sample_error - median)))
            thresholds[row] = max(cfg['residual_floor'], median + 3 * 1.4826 * mad)
            transforms[row, 0], transforms[row, 1] = matrix, cv2.invertAffineTransform(matrix)
            report.update(camera_ok=True, reason='ok', residual_threshold=thresholds[row].item(),
                          time_scale=time_scales[row])
            accepted.append(row)
        scores = forward.new_zeros(batch, 2, h, w)
        if not accepted:
            return scores, reports
        coordinates = self._coordinates(forward)
        matrices = forward.new_tensor(transforms[accepted])
        camera = torch.einsum('bdij,jhw->bdihw', matrices, coordinates) - coordinates[None, None, :2]
        fields = torch.stack((forward[accepted], backward[accepted]), 1)
        residual = (fields - camera).square().sum(2).sqrt()
        residual *= forward.new_tensor([time_scales[row] for row in accepted])[:, None, None, None]
        threshold = forward.new_tensor(thresholds[accepted])[:, None, None, None]
        reliable = torch.cat((reliable0[accepted], reliable1[accepted]), 1)
        values = ((residual - threshold) / threshold).clamp(0, 1) * reliable
        binaries = (values > 0).to(torch.uint8).cpu().numpy()
        kept = np.zeros_like(binaries, dtype=np.uint8)
        for offset, row in enumerate(accepted):
            fractions = []
            for slot in (0, 1):
                binary = binaries[offset, slot]
                if binary.any():
                    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                    keep = np.zeros(count, dtype=bool)
                    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= cfg['min_component_area']
                    kept[offset, slot] = keep[labels]
                # Match the old FP32 area division, including its rounding.
                fractions.append(float(np.float32(np.count_nonzero(kept[offset, slot])) / metadata[row, 5]))
            report = reports[row]
            fraction = max(fractions)
            report['motion_fraction'] = fraction
            if fraction > cfg['max_motion_fraction']:
                report.update(camera_ok=False, reason='excessive_motion_area')
                kept[offset] = 0
            else:
                report['support_fraction'] = float(metadata[row, 6])
        scores[accepted] = values * torch.from_numpy(kept).to(values.device)
        return scores, reports

    def _camera_scores(self, forward, backward, reliable0, reliable1, spatial_valid, time_scale):
        """Single-pair compatibility wrapper around the batched implementation."""
        scores, reports = self._camera_scores_batch(forward, backward, reliable0, reliable1,
            spatial_valid, [time_scale], torch.ones(1, device=forward.device, dtype=torch.bool))
        return scores[:, :1], scores[:, 1:], reports[0]

    def _pair_batch(self, first, second, spatial, scales, motion):
        zero = first.new_zeros(len(first))
        fields0, fields1 = motion(first, second, zero), motion(first, second, zero + 1)
        expected = (len(first), 4, *first.shape[-2:])
        if fields0.shape != expected or fields1.shape != expected:
            raise ValueError('Endpoint RIFE flow must provide [B,4,512,512] in context pixels')
        forward, backward = fields0[:, 2:].float(), fields1[:, :2].float()
        finite = torch.isfinite(forward).flatten(1).all(1) & torch.isfinite(backward).flatten(1).all(1)
        forward = torch.nan_to_num(forward, nan=1e6, posinf=1e6, neginf=-1e6)
        backward = torch.nan_to_num(backward, nan=1e6, posinf=1e6, neginf=-1e6)
        warped1, inside0 = flow_warp(second, forward)
        warped0, inside1 = flow_warp(first, backward)
        spatial1, _ = flow_warp(spatial, forward)
        spatial0, _ = flow_warp(spatial, backward)
        reliable0 = _fb_valid(forward, backward) * inside0 * spatial * (spatial1 > .999)
        reliable1 = _fb_valid(backward, forward) * inside1 * spatial * (spatial0 > .999)
        reliable0 *= (first - warped1).abs().mean(1, keepdim=True) <= self.config['photometric_threshold']
        reliable1 *= (second - warped0).abs().mean(1, keepdim=True) <= self.config['photometric_threshold']
        reliable0 *= _texture(first) >= self.config['texture_threshold']
        reliable1 *= _texture(second) >= self.config['texture_threshold']
        scores, reports = self._camera_scores_batch(forward, backward, reliable0, reliable1,
                                                    spatial, scales, finite)
        ok = first.new_tensor([report['camera_ok'] for report in reports])[:, None, None, None]
        supports = torch.cat((reliable0, reliable1), 1) * ok
        return [{'motion0': scores[row:row+1, :1], 'motion1': scores[row:row+1, 1:],
                 'support0': supports[row:row+1, :1], 'support1': supports[row:row+1, 1:],
                 'report': report} for row, report in enumerate(reports)]

    def _modality(self, inputs, context, name, motion, cache):
        frames = getattr(context, name).detach().float()
        def rgb(value):
            return value.repeat(1, 3, 1, 1) if value.shape[1] == 1 else value
        times, valid = getattr(inputs, name + '_times'), getattr(inputs, name + '_valid')
        duration = (times[:, 1] - times[:, 0]).abs()
        # The temporal numbers are tiny. Transfer once rather than synchronizing
        # Python with CUDA separately for every row, timestamp, slot and tau.
        metadata = torch.cat((times, valid.to(times.dtype), inputs.query_time[:, None],
            (context.frame_step / duration)[:, None],
            ((inputs.query_time - times[:, 0]) / (times[:, 1] - times[:, 0]))[:, None]), 1).cpu().tolist()
        persist = cache.get('cache_motion_pairs', True)
        if not isinstance(persist, bool):
            raise ValueError('cache_motion_pairs must be a boolean')
        pairs = cache.setdefault('motion_pairs', OrderedDict()) if persist else OrderedDict()
        limit = cache.get('max_motion_pairs', 4 * len(frames))
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError('max_motion_pairs must be a positive integer')
        versions = tuple(parameter._version for parameter in motion.parameters())
        evidence, pending, candidates = {}, [], []
        for row, (t0, t1, v0, v1, _, scale, _) in enumerate(metadata):
            if not (v0 and v1 and t0 != t1 and math.isfinite(t1 - t0)):
                continue
            key = (name, row, (t0, t1), scale, tuple(frames.shape[2:]),
                   str(frames.device), id(motion), versions, tuple(self.config.items()))
            saved = pairs.get(key)
            if saved is not None:
                candidates.append((row, key, scale, saved))
            else:
                pending.append((row, key, scale))
        if candidates:
            matches = torch.stack([
                (saved['first'] == frames[row:row+1, 0]).all()
                & (saved['second'] == frames[row:row+1, 1]).all()
                & (saved['spatial'] == context.spatial_valid[row:row+1]).all()
                for row, _, _, saved in candidates]).cpu().tolist()
            for (row, key, scale, saved), matches_input in zip(candidates, matches):
                if matches_input:
                    evidence[row] = saved
                    pairs.move_to_end(key)
                else:
                    pending.append((row, key, scale))
        # Keep microbatch order independent of which rows happened to hit cache.
        pending.sort(key=lambda item: item[0])
        for start in range(0, len(pending), self.config['flow_microbatch']):
            part = pending[start:start + self.config['flow_microbatch']]
            rows = [item[0] for item in part]
            results = self._pair_batch(rgb(frames[rows, 0]), rgb(frames[rows, 1]),
                context.spatial_valid[rows].float(), [item[2] for item in part], motion)
            for (row, key, _), item in zip(part, results):
                evidence[row] = item
                if persist:
                    # IR snapshots remain grayscale; RGB is expanded only for
                    # the active flow microbatch, not the entire clip batch.
                    item.update(first=frames[row:row+1, 0].clone(),
                                second=frames[row:row+1, 1].clone(),
                                spatial=context.spatial_valid[row:row+1].clone())
                    pairs[key] = item
                    pairs.move_to_end(key)
                    while len(pairs) > limit:
                        pairs.popitem(last=False)
        masks = torch.zeros_like(context.spatial_valid, dtype=torch.float32)
        supports = torch.zeros_like(masks)
        reports = []
        missing = []
        for row in range(len(frames)):
            item = evidence.get(row)
            report = dict(item['report']) if item else _report()
            reports.append(report)
            if item is None or not report['camera_ok']:
                continue
            t0, t1, v0, v1, query, _, tau = metadata[row]
            exact0, exact1 = bool(v0 and t0 == query), bool(v1 and t1 == query)
            if exact0 or exact1:
                slot = 0 if exact0 else 1
                masks[row] = item[f'motion{slot}'][0]
                supports[row] = item[f'support{slot}'][0]
                report['query_source'] = f'endpoint{slot}'
            else:
                if 0 < tau < 1:
                    missing.append((row, tau))
                else:
                    report.update(reason='query_outside_context', camera_ok=False)
        for start in range(0, len(missing), self.config['flow_microbatch']):
            part = missing[start:start + self.config['flow_microbatch']]
            rows = [item[0] for item in part]
            tau = frames.new_tensor([item[1] for item in part])
            fields = motion(rgb(frames[rows, 0]), rgb(frames[rows, 1]), tau).float()
            if fields.shape != (len(rows), 4, *frames.shape[-2:]):
                raise ValueError('Query RIFE flow must provide [B,4,512,512] in context pixels')
            finite = torch.isfinite(fields).flatten(1).all(1)
            finite_rows = finite.cpu().tolist()
            # Warp the two endpoint masks AND supports together. Invalid rows
            # get a safe zero grid and are zeroed before writing their evidence.
            safe_fields = torch.where(finite[:, None, None, None], fields, 0.)
            values0 = torch.cat((torch.cat([evidence[row]['motion0'] for row in rows]),
                                 torch.cat([evidence[row]['support0'] for row in rows])), 1)
            values1 = torch.cat((torch.cat([evidence[row]['motion1'] for row in rows]),
                                 torch.cat([evidence[row]['support1'] for row in rows])), 1)
            warped, inside = flow_warp(torch.cat((values0, values1)),
                                      torch.cat((safe_fields[:, :2], safe_fields[:, 2:])))
            warped *= inside
            aligned = torch.maximum(warped[:len(rows)], warped[len(rows):])
            aligned *= finite[:, None, None, None]
            masks[rows], supports[rows] = aligned[:, :1], aligned[:, 1:]
            for row, is_finite in zip(rows, finite_rows):
                if not is_finite:
                    reports[row].update(reason='nonfinite_query_flow', camera_ok=False)
                else:
                    reports[row]['query_source'] = 'aligned_endpoint_union'
        masks *= context.spatial_valid
        supports *= context.spatial_valid
        size = inputs.vis.shape[-2:]
        projected = project_context_map(torch.cat((masks, supports), 1), context, size).clamp(0, 1)
        return projected[:, :1], projected[:, 1:], reports

    @torch.no_grad()
    def forward(self, inputs, ir_motion, vis_motion, cache=None):
        cache = {} if cache is None else cache
        context = ensure_motion_context(inputs)
        with torch.autocast(device_type=inputs.vis.device.type, enabled=False):
            ir_mask, ir_support, ir_reports = self._modality(inputs, context, 'ir', ir_motion, cache)
            vis_mask, vis_support, vis_reports = self._modality(inputs, context, 'vis', vis_motion, cache)
        return {'ir_motion_mask': ir_mask, 'vis_motion_mask': vis_mask,
                'motion_mask': torch.maximum(ir_mask, vis_mask),
                'ir_motion_support': ir_support, 'vis_motion_support': vis_support,
                'diagnostics': {'ir': ir_reports, 'vis': vis_reports}}
