"""Loss-only target supervision. No teacher is attached to the inference model."""
import copy
import math
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from .geometry import flow_warp, image_gradients
from .temporal import time_context
from .types import SOURCE_KEYS


def luminance(rgb):
    return (rgb.float()*rgb.new_tensor([0.299, 0.587, 0.114]).float().view(1, 3, 1, 1)).sum(1, keepdim=True)


def masked_mean(error, mask):
    error = error.float()
    mask = mask.to(error).expand_as(error)
    return (error*mask).sum()/mask.sum().clamp_min(1e-6)


def image_masked_mean(error, mask):
    mask = mask.to(error).expand_as(error)
    return (error * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp_min(1e-6)


def image_gradient_l1(prediction, target, mask=None):
    px, py = image_gradients(prediction.float())
    tx, ty = image_gradients(target.float())
    if mask is None:
        return (px-tx).abs().flatten(1).mean(1) + (py-ty).abs().flatten(1).mean(1)
    mx, my = mask
    return image_masked_mean((px-tx).abs(), mx) + image_masked_mean((py-ty).abs(), my)


def sum_statistics(rows):
    return {name: (sum(row[name][0] for row in rows), sum(row[name][1] for row in rows))
            for name in rows[0]}


def normalize_statistics(statistics, weights, denominators=None):
    """Normalize additive per-image sums using local or effective-update counts."""
    denominators = denominators if denominators is not None else {k: v[1] for k, v in statistics.items()}
    means = {name: numerator / denominators[name].clamp_min(1)
             for name, (numerator, _) in statistics.items()}
    terms = {name: means[name] for name in weights if name not in ('gradient', 'motion')}
    for name in ('gradient', 'motion'):
        available = sum((denominators[name + '_' + modality] > 0).to(means[name + '_' + modality])
                        for modality in ('ir', 'vis')).clamp_min(1)
        terms[name] = (means[name + '_ir'] + means[name + '_vis']) / available
    terms['total'] = sum(weights[name] * value for name, value in terms.items())
    return terms


def gradient_l1(prediction, target, mask=None):
    px, py = image_gradients(prediction.float())
    tx, ty = image_gradients(target.float())
    if mask is None:
        return (px-tx).abs().mean() + (py-ty).abs().mean()
    if isinstance(mask, tuple):
        mx, my = mask
    else:
        mx = torch.maximum(mask[..., :, 1:], mask[..., :, :-1])
        my = torch.maximum(mask[..., 1:, :], mask[..., :-1, :])
    return masked_mean((px-tx).abs(), mx) + masked_mean((py-ty).abs(), my)


def hot_region(ir):
    """Original fusion ROI: pixels at or above the per-image 80th percentile."""
    threshold = torch.quantile(ir.float().flatten(1), 0.8, dim=1).view(-1, 1, 1, 1)
    return (ir >= threshold).float()


def fusion_terms(prediction, ir_gt, vis_gt):
    pred_y, vis_y = luminance(prediction), luminance(vis_gt)
    gx, gy = image_gradients(vis_y)
    return {'fusion_vis': gradient_l1(pred_y, vis_y, (gx.abs() > 0.03, gy.abs() > 0.03)),
            'fusion_ir': masked_mean((pred_y-ir_gt).abs(), hot_region(ir_gt)),
            'color': F.l1_loss(prediction-pred_y, vis_gt-vis_y)}


def validate_target_mask(mask, target):
    if (mask is None or mask.shape != target.shape or mask.device != target.device
            or not torch.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any()):
        raise ValueError('Independent GT target mask must be finite [B,1,H,W] in [0,1]')
    return mask.detach().float()


@torch.no_grad()
def build_target_supervision(model, inputs, vis_gt, reference=None, teacher_inputs=None, cache=None):
    """Independent GT teacher; unavailable evidence remains unknown.

    `teacher_inputs` contains the true neighboring GT query frames and full-FOV
    context. It must never be passed to model.forward. Absence of this context
    disables motion supervision rather than deriving labels from predictions.
    """
    zero = vis_gt.new_zeros(vis_gt.shape[0], 1, *vis_gt.shape[-2:], dtype=torch.float32)
    enabled = getattr(model, 'segmentation_enabled',
                      getattr(model, 'config', {}).get('target_gate', {}).get('segmentation_enabled', True))
    semantic = validate_target_mask(model.reference_target_masks(vis_gt), zero) if enabled else zero
    # PMT positives are evidence; missing detections cannot rule out motion targets.
    support = (semantic > 0).float()
    evidence = {'ir_motion_mask': zero, 'vis_motion_mask': zero, 'motion_mask': zero,
                'ir_motion_support': zero, 'vis_motion_support': zero,
                'diagnostics': {modality: [{'camera_ok': False, 'reason': 'teacher_context_unavailable',
                                            'motion_fraction': 0., 'support_fraction': 0.}
                                           for _ in range(vis_gt.shape[0])] for modality in ('ir', 'vis')}}
    if reference is not None and teacher_inputs is not None:
        from .models.motion_gate import MotionEvidence
        config = getattr(model, 'config', {}).get('target_gate', {}).get('motion', {})
        engine = MotionEvidence(config).to(vis_gt.device).eval()
        with torch.autocast(device_type=vis_gt.device.type, enabled=False):
            evidence = engine(teacher_inputs, reference.ir_motion, reference.vis_motion, cache=cache)
        support = torch.maximum(support, torch.maximum(evidence['ir_motion_support'],
                                                       evidence['vis_motion_support']))
    mask = torch.maximum(semantic, evidence['motion_mask']) * support
    return {'mask': validate_target_mask(mask, zero), 'semantic_mask': semantic,
            'ir_motion_mask': evidence['ir_motion_mask'], 'vis_motion_mask': evidence['vis_motion_mask'],
            'support': support.detach().float(), 'diagnostics': evidence['diagnostics'],
            'segmentation_enabled': bool(enabled)}


def target_supervision_statistics(supervision):
    """Additive diagnostics use their own denominators and never objective weights."""
    result = {}
    for name, key in (('teacher_evidence_coverage', 'support'), ('teacher_semantic_coverage', 'semantic_mask'),
                      ('teacher_ir_motion_coverage', 'ir_motion_mask'),
                      ('teacher_vis_motion_coverage', 'vis_motion_mask')):
        values = supervision[key].float().flatten(1).mean(1)
        result[name] = (values.sum(), values.new_tensor(values.numel()))
    for modality in ('ir', 'vis'):
        reports = supervision['diagnostics'].get(modality, [])
        values = supervision['mask'].new_tensor([bool(row.get('camera_ok', False)) for row in reports])
        result['teacher_' + modality + '_camera_success'] = (values.sum(), values.new_tensor(values.numel()))
    return result


@torch.no_grad()
def thermal_delta_target(ir_gt, vis_gt, fusion_config=None):
    config = normalize_training_fusion_config(fusion_config)
    delta = config['target_gain'] * (ir_gt.float()-luminance(vis_gt)).relu()
    return torch.minimum(delta, (1-vis_gt.float().amax(1, keepdim=True)).clamp_min(0)).clamp(
        0, config['residual_scale'])


V6_FUSION_DEFAULTS = {
    'mode': 'target_gated', 'residual_scale': .25, 'target_gain': .5,
    'target_luma_weight': 1., 'target_detail_weight': .5,
    'interp_vis_weight': 3., 'temporal_weight': .1,
    'reference_l1_weight': 1., 'reference_gradient_weight': .1,
}


def normalize_training_fusion_config(config=None):
    result = dict(V6_FUSION_DEFAULTS)
    if config is not None:
        if not isinstance(config, dict):
            raise ValueError('fusion must be a mapping')
        unknown = set(config) - set(result)
        if unknown:
            raise ValueError(f'Unknown v6 fusion settings: {sorted(unknown)}')
        result.update(config)
    if result['mode'] != 'target_gated':
        raise ValueError('v6 only supports target_gated fusion')
    for name, value in result.items():
        if name == 'mode':
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f'fusion.{name} must be finite and nonnegative')
        result[name] = float(value)
    if result['residual_scale'] != .25:
        raise ValueError('v6 architecture requires fusion.residual_scale=.25')
    if not 0 < result['target_gain'] <= 1:
        raise ValueError('target_gain must be in (0, 1]')
    return result


class LossComputer(nn.Module):
    def __init__(self, stage, fusion_config=None, *, ablation=None):
        super().__init__()
        from .models.ablation import normalize_ablation
        if stage not in ('a', 'b', 'c'):
            raise ValueError('Stage must be a, b or c')
        self.stage = stage
        self.fusion_config = normalize_training_fusion_config(fusion_config)
        self.ablation = normalize_ablation(ablation)
        self.ablation_enabled = ablation is not None
        self.weights = {'interp_ir': 1., 'interp_vis': 1., 'gradient': .1, 'motion': .2}
        if stage == 'a':
            self.weights.update(reference_l1=self.fusion_config['reference_l1_weight'],
                                reference_gradient=self.fusion_config['reference_gradient_weight'])
        if stage != 'a':
            self.weights.update(match=.05, allocation=.05, reliability=.1,
                                interp_vis=self.fusion_config['interp_vis_weight'],
                                target_luma=self.fusion_config['target_luma_weight'],
                                target_detail=self.fusion_config['target_detail_weight'], applied_luma=1.)
            if not self.ablation['semantic_features']:
                self.weights.pop('match')
            if self.ablation['aggregation'] == 'standard':
                for name in ('match', 'allocation', 'reliability'):
                    self.weights.pop(name, None)
            if self.ablation_enabled:
                self.weights['background'] = self.ablation['background_weight']
        self.temporal_weight = self.fusion_config['temporal_weight'] if stage == 'c' else 0.

    @property
    def validation_weights(self):
        # Applied error measures the actual output, including unreachable zero gates.
        return dict(self.weights, **({'applied_luma': 1.} if self.stage != 'a' else {}))

    def forward(self, output, inputs, ir_gt, vis_gt, *, return_stats=False, reference=None,
                target_mask=None, target_support=None):
        with torch.autocast(device_type=output.fused.device.type, enabled=False):
            statistics = self._compute(output, inputs, ir_gt.float(), vis_gt.float(), reference, target_mask, target_support)
            if return_stats:
                return statistics
            terms = normalize_statistics(statistics, self.weights)
            return terms['total'], terms

    def _compute(self, output, inputs, ir_gt, vis_gt, reference=None, target_mask=None, target_support=None):
        if self.stage != 'a':
            for name in ('target_gate', 'vis_reference', 'pre_gate_delta'):
                value = getattr(output, name, None)
                channels = 3 if name == 'vis_reference' else 1
                expected = (ir_gt.shape[0], channels, *ir_gt.shape[-2:])
                if value is None or value.shape != expected or value.device != ir_gt.device:
                    raise ValueError(f'v6 fusion loss requires output.{name} of shape {expected}')
            target_mask = validate_target_mask(target_mask, ir_gt)
            target_support = (torch.ones_like(target_mask) if target_support is None
                              else validate_target_mask(target_support, ir_gt))
            target_mask = target_mask * target_support
        context = time_context(inputs)
        missing = ~context['observed']
        statistics = {}
        def record(name, values, selected=None):
            if selected is None:
                statistics[name] = (values.sum(), values.new_tensor(values.shape[0]))
            else:
                statistics[name] = ((values * selected).sum(), selected.sum().to(values))
        for i, (name, prediction, target) in enumerate((('ir', output.ir_aux, ir_gt),
                                                       ('vis', output.vis_aux, vis_gt))):
            selected = missing[:, i]
            record('interp_' + name, (prediction.float()-target).abs().flatten(1).mean(1), selected)
            record('gradient_' + name, image_gradient_l1(prediction, target), selected)
        # The actual RGB reference, rather than the auxiliary decoder output, is
        # the background used by fusion. Exact VIS samples never dilute this loss.
        record('reference_l1', (output.vis_reference.float()-vis_gt).abs().flatten(1).mean(1), missing[:, 1])
        record('reference_gradient', image_gradient_l1(output.vis_reference, vis_gt), missing[:, 1])
        images = (inputs.ir[:, 0], inputs.vis[:, 0], inputs.ir[:, 1], inputs.vis[:, 1])
        errors, validities, motion_values, motion_masks = [], [], [], []
        for i, (key, source) in enumerate(zip(SOURCE_KEYS, images)):
            target = ir_gt if i % 2 == 0 else vis_gt
            active = context['effective_valid'][:, i, None, None, None]
            # Ignore padded sources at pixels as well as in the loss mask.
            source = torch.where(active, source, torch.zeros_like(source))
            warped, valid = flow_warp(source.float(), output.flows[key])
            valid = valid*active
            error = (warped-target).abs().mean(1, keepdim=True)
            errors.append(error)
            validities.append(valid)
            motion_values.append(error)
            motion_masks.append(valid*missing[:, i % 2, None, None, None])
        for modality, name in enumerate(('ir', 'vis')):
            values = image_masked_mean(torch.cat(motion_values[modality::2], 1),
                                       torch.cat(motion_masks[modality::2], 1))
            record('motion_' + name, values, missing[:, modality])
        if self.stage != 'a':
            valid = torch.cat(validities, 1)
            reliability_error = torch.cat(errors, 1).detach()
            if self.ablation['cascade']:
                # Cascade encodes query images copied into the original active
                # slots, not warped endpoints. Supervise the evidence actually
                # seen by fusion, while retaining endpoint errors for motion.
                query_errors = ((output.ir_base.detach().float()-ir_gt).abs().mean(1, keepdim=True),
                                (output.vis_reference.detach().float()-vis_gt).abs().mean(1, keepdim=True))
                reliability_error = torch.cat(query_errors * 2, 1)
                valid = context['effective_valid'][:, :, None, None].to(reliability_error).expand_as(reliability_error)
            target_reliability = torch.exp(-reliability_error/0.1)*valid
            probability = F.interpolate(output.reliability.float(), ir_gt.shape[-2:], mode='bilinear', align_corners=False).clamp(1e-5, 1-1e-5)
            visibility = F.interpolate(output.visibility.float(), ir_gt.shape[-2:], mode='bilinear', align_corners=False).clamp(1e-5, 1-1e-5)
            active = context['effective_valid'][:, :, None, None]
            reliability = image_masked_mean(F.binary_cross_entropy(probability, target_reliability, reduction='none'), active)
            reliability += 0.5*image_masked_mean(F.binary_cross_entropy(visibility, (target_reliability > 0.2).float(), reduction='none'), active)
            record('reliability', reliability)
            small = F.interpolate(target_reliability, output.reliability.shape[-2:], mode='area')
            mass = small/small.sum(1, keepdim=True).clamp_min(1e-6)
            if self.ablation['semantic_features']:
                need = 1-small.max(1, keepdim=True).values
                allocation = torch.cat((mass*(1-need), need), 1).detach()
                allocation_error = F.kl_div(output.source_weights.float().clamp_min(1e-6).log(), allocation,
                                           reduction='none').sum(1)
                record('allocation', allocation_error.flatten(1).mean(1))
            else:
                # S=0 has no fifth semantic destination. Only actual sources
                # receive mass, and pixels with no valid source do not enter KL.
                supported = F.interpolate(valid, small.shape[-2:], mode='area').sum(1, keepdim=True) > 0
                mass = small/small.sum(1, keepdim=True).clamp_min(torch.finfo(small.dtype).tiny)
                allocation = torch.cat((mass, torch.zeros_like(mass[:, :1])), 1).detach()
                allocation_error = F.kl_div(output.source_weights.float().clamp_min(1e-6).log(), allocation,
                                           reduction='none').sum(1, keepdim=True)
                record('allocation', image_masked_mean(allocation_error, supported))
            details = output.transport_details
            match = output.fused.float().flatten(1).sum(1) * 0
            if details is not None and 'match' in self.weights:
                error = -(details['target']*details['weights'].clamp_min(1e-12).log()).sum(-1)
                values = image_masked_mean(error, details['reliable']*details['valid'].any(-1))
                match = match.index_add(0, details['batch_indices'], values)
            record('match', match)
            prediction = output.fused.float()
            pred_y, vis_y = luminance(prediction), luminance(vis_gt)
            gate = output.target_gate.detach().float().clamp(0, 1)
            with torch.no_grad():
                target_delta = thermal_delta_target(ir_gt, vis_gt, self.fusion_config)
                desired_delta = target_mask * target_delta
            actual_delta = pred_y - luminance(output.vis_reference.detach())
            if self.ablation_enabled:
                # Every ablation, including Full, uses the same GT-teacher
                # background rather than its own predicted gate. Empty images
                # contribute zero under the scheduled per-query denominator.
                background = (1-target_mask)*target_support
                record('background', image_masked_mean(
                    (prediction-output.vis_reference.detach().float()).abs(), background))
                record('background_support', (background.flatten(1).sum(1) > 1e-6).float())
            record('target_luma', image_masked_mean((output.pre_gate_delta.float()-target_delta).abs(), target_mask))
            record('applied_luma', image_masked_mean((actual_delta-desired_delta).abs(), target_mask))
            record('gate_miss', image_masked_mean((gate <= 1e-6).float(), target_mask))
            record('target_support', (target_mask.flatten(1).sum(1) > 1e-6).float())
            record('target_coverage', target_mask.flatten(1).mean(1))
            edges = (torch.minimum(target_mask[..., :, 1:], target_mask[..., :, :-1]),
                     torch.minimum(target_mask[..., 1:, :], target_mask[..., :-1, :]))
            record('target_detail', image_gradient_l1(pred_y, vis_y + target_delta, edges))
            record('target_detail_support', ((edges[0].flatten(1).sum(1)+edges[1].flatten(1).sum(1)) > 1e-6).float())
        return statistics


@torch.no_grad()
def missing_metrics(output, ir_gt, vis_gt):
    """Per-image means over missing samples only; zero/count=0 means unavailable."""
    result = {}
    for i, name, prediction, target in ((0, 'ir', output.ir_aux, ir_gt),
                                       (1, 'vis', output.vis_aux, vis_gt),
                                       (1, 'vis_reference', output.vis_reference, vis_gt)):
        select = ~output.observed[:, i]
        result[name+'_count'] = int(select.sum())
        if not select.any():
            result[name+'_psnr'] = prediction.new_zeros((), dtype=torch.float32)
            result[name+'_ssim'] = prediction.new_zeros((), dtype=torch.float32)
            continue
        p, t = prediction[select].float(), target[select].float()
        mse = (p-t).square().flatten(1).mean(1)
        result[name+'_psnr'] = (-10*mse.clamp_min(1e-12).log10()).mean()
        # Local SSIM with an 11x11 Gaussian window, clipped to small image sizes.
        size = min(11, p.shape[-2], p.shape[-1])
        coords = torch.arange(size, device=p.device, dtype=torch.float32)-(size-1)/2
        kernel = (-coords.square()/(2*1.5**2)).exp()
        kernel = kernel/kernel.sum()
        window = (kernel[:, None]*kernel[None, :])[None, None].expand(p.shape[1], 1, -1, -1)
        def average(x):
            return F.conv2d(x, window, groups=p.shape[1])
        mp, mt = average(p), average(t)
        vp, vt = (average(p*p)-mp*mp).clamp_min(0), (average(t*t)-mt*mt).clamp_min(0)
        cov = average(p*t)-mp*mt
        ssim = ((2*mp*mt+0.01**2)*(2*cov+0.03**2))/((mp*mp+mt*mt+0.01**2)*(vp+vt+0.03**2))
        result[name+'_ssim'] = ssim.flatten(1).mean(1).mean()
    return result


class TemporalReference(nn.Module):
    """Frozen, separate training helper. Save state under checkpoint['temporal_reference']."""
    def __init__(self, ir_motion, vis_motion, fusion_config=None, *, ablation=None):
        super().__init__()
        from .models.ablation import normalize_ablation
        self.fusion_config = normalize_training_fusion_config(fusion_config)
        self.ablation = normalize_ablation(ablation)
        self.ir_motion = copy.deepcopy(ir_motion)
        self.vis_motion = copy.deepcopy(vis_motion)
        for module in (self.ir_motion, self.vis_motion):
            if hasattr(module, 'set_trainable'):
                module.set_trainable(False)
        self.requires_grad_(False).eval()

    @classmethod
    def from_best_a(cls, model, snapshot, fusion_config=None, *, ablation=None):
        if isinstance(snapshot, (str, Path)):
            snapshot = torch.load(snapshot, map_location='cpu', weights_only=False)
        state = snapshot.get('model', snapshot)
        ir_state = {k.removeprefix('ir_motion.'): v for k, v in state.items() if k.startswith('ir_motion.')}
        vis_state = {k.removeprefix('vis_motion.'): v for k, v in state.items() if k.startswith('vis_motion.')}
        if not ir_state or not vis_state:
            raise ValueError('bestA snapshot must carry both IR and VIS motion states')
        reference = cls(model.ir_motion, model.vis_motion, fusion_config=fusion_config, ablation=ablation)
        reference.ir_motion.load_state_dict(ir_state, strict=True)
        reference.vis_motion.load_state_dict(vis_state, strict=True)
        return reference

    def train(self, mode=True):
        super().train(False)
        return self

    @staticmethod
    @torch.no_grad()
    def _flow_and_mask(motion, a, b):
        tau = torch.ones(a.shape[0], device=a.device, dtype=torch.float32)
        flow = motion(a.float(), b.float(), tau)[:, :2]
        reverse = motion(a.float(), b.float(), torch.zeros_like(tau))[:, 2:]
        reverse_warp, valid = flow_warp(reverse, flow)
        error = (flow+reverse_warp).square().sum(1, keepdim=True)
        scale = flow.square().sum(1, keepdim=True)+reverse_warp.square().sum(1, keepdim=True)
        return flow, valid*(error <= 0.01*scale+0.5)

    def forward(self, predictions, ir_targets, vis_targets=None, return_per_sample=False, *,
                target_masks=None, observed_vis=None, return_statistics=False):
        """Independent teacher-union supervision; observed anchors cannot be dimmed.

        Frozen best-A IR flow aligns actual applied deltas. The mask combines
        independent GT semantics, forward/backward visibility and GT VIS
        photometric confidence. Same-rate pairs train observed -> missing only;
        fully observed VIS pairs train both directions. Thermal target changes
        compensate real brightness changes instead of enforcing a constant lift.
        """
        if predictions.ndim != 5 or predictions.shape[1] < 2 or predictions.shape[2] != 1:
            raise ValueError('v6 temporal supervision requires [B,T>=2,1,H,W] luminance deltas')
        if ir_targets.shape != predictions.shape or target_masks is None or target_masks.shape != predictions.shape:
            raise ValueError('IR targets and target masks must match the luminance deltas')
        batch, frames, _, height, width = predictions.shape
        if vis_targets is None or vis_targets.shape != (batch, frames, 3, height, width):
            raise ValueError('Temporal supervision requires independent GT VIS [B,T,3,H,W]')
        if observed_vis is None:
            observed_vis = torch.ones(batch, frames, device=predictions.device, dtype=torch.bool)
        if observed_vis.shape != (batch, frames) or observed_vis.dtype != torch.bool:
            raise ValueError('observed_vis must be boolean [B,T]')
        masks = validate_target_mask(target_masks.flatten(0, 1), ir_targets.flatten(0, 1)).reshape_as(target_masks)
        if self.ablation['temporal'] == 'standard':
            return self._standard(predictions, ir_targets, vis_targets, masks,
                                  return_per_sample=return_per_sample, return_statistics=return_statistics)
        with torch.autocast(device_type=predictions.device.type, enabled=False):
            desired = (thermal_delta_target(ir_targets.flatten(0, 1), vis_targets.flatten(0, 1),
                                             self.fusion_config).reshape_as(predictions) * masks)
            zero = predictions.float().flatten(1).sum(1) * 0
            numerator, valid_pairs = zero, torch.zeros_like(zero)
            eligible_pairs, support_mass, visible_mass = zero.detach().sum(), zero.detach().sum(), zero.detach().sum()
            for i in range(frames-1):
                for source, target in ((i, i+1), (i+1, i)):
                    active = observed_vis[:, source]
                    if not active.any():
                        continue
                    with torch.no_grad():
                        a, b = ir_targets[:, source].float(), ir_targets[:, target].float()
                        flow, consistent = self._flow_and_mask(
                            self.ir_motion, a.repeat(1, 3, 1, 1), b.repeat(1, 3, 1, 1))
                        old_mask, inside = flow_warp(masks[:, source], flow)
                        union = torch.maximum(old_mask, masks[:, target]).clamp(0, 1)
                        warped_vis, image_valid = flow_warp(vis_targets[:, source].float(), flow)
                        photo = (warped_vis-vis_targets[:, target].float()).abs().mean(1, keepdim=True)
                        confidence = torch.exp(-photo / .1)
                        geometry = inside * image_valid * consistent * active[:, None, None, None]
                        visible = geometry * confidence
                        weight = union * visible
                        anchor, anchor_valid = flow_warp(predictions[:, source].detach().float(), flow)
                        old_desired, _ = flow_warp(desired[:, source], flow)
                        # No current-output gate enters either the target or mask.
                        target_delta = anchor + desired[:, target] - old_desired
                        headroom = (1-vis_targets[:, target].float().amax(1, keepdim=True)).clamp_min(0)
                        target_delta = torch.minimum(target_delta.clamp(0, .25), headroom)
                        weight = weight * anchor_valid
                        normalization_mask = union * geometry * anchor_valid
                        valid = weight.flatten(1).sum(1) > 1e-6
                        eligible_pairs = eligible_pairs + active.sum()
                        support_mass = support_mass + weight.sum() / (height*width)
                        visible_mass = visible_mass + visible.sum() / (height*width)
                    error = (predictions[:, target].float()-target_delta).abs()
                    # Confidence must attenuate an entirely unreliable ROI, not
                    # cancel from numerator and denominator of a weighted mean.
                    numerator = numerator + image_masked_mean(error * confidence, normalization_mask) * valid
                    valid_pairs = valid_pairs + valid
            value = numerator / valid_pairs.clamp_min(1)
            if return_statistics:
                count = value.new_tensor(batch)
                return {'temporal': (value.sum(), count),
                        'temporal_support': ((valid_pairs > 0).float().sum(), count),
                        'temporal_pair_support': (valid_pairs.sum(), eligible_pairs),
                        'temporal_coverage': (support_mass, eligible_pairs),
                        'temporal_visibility': (visible_mass, eligible_pairs)}
            return value if return_per_sample else value.mean()

    def _standard(self, predictions, ir_targets, vis_targets, masks, *,
                  return_per_sample=False, return_statistics=False):
        """Symmetric flow consistency, with identical geometry and confidence.

        Every adjacent pair contributes both directions regardless of observation
        flags. Both predictions receive gradients; no thermal correction or
        target clipping is applied to this conventional consistency baseline.
        """
        batch, frames, _, height, width = predictions.shape
        with torch.autocast(device_type=predictions.device.type, enabled=False):
            zero = predictions.float().flatten(1).sum(1) * 0
            numerator, valid_pairs = zero, torch.zeros_like(zero)
            eligible_pairs, support_mass, visible_mass = zero.detach().sum(), zero.detach().sum(), zero.detach().sum()
            for i in range(frames-1):
                for source, target in ((i, i+1), (i+1, i)):
                    with torch.no_grad():
                        a, b = ir_targets[:, source].float(), ir_targets[:, target].float()
                        flow, consistent = self._flow_and_mask(
                            self.ir_motion, a.repeat(1, 3, 1, 1), b.repeat(1, 3, 1, 1))
                        old_mask, inside = flow_warp(masks[:, source], flow)
                        union = torch.maximum(old_mask, masks[:, target]).clamp(0, 1)
                        warped_vis, image_valid = flow_warp(vis_targets[:, source].float(), flow)
                        photo = (warped_vis-vis_targets[:, target].float()).abs().mean(1, keepdim=True)
                        confidence = torch.exp(-photo / .1)
                        geometry = inside * image_valid * consistent
                        visible = geometry * confidence
                    # The warp must be outside no_grad: unlike the proposed
                    # observed anchor, this source prediction remains trainable.
                    warped_prediction, prediction_valid = flow_warp(predictions[:, source].float(), flow)
                    with torch.no_grad():
                        weight = union * visible * prediction_valid
                        normalization_mask = union * geometry * prediction_valid
                        valid = weight.flatten(1).sum(1) > 1e-6
                        eligible_pairs = eligible_pairs + batch
                        support_mass = support_mass + weight.sum() / (height*width)
                        visible_mass = visible_mass + visible.sum() / (height*width)
                    error = (predictions[:, target].float()-warped_prediction).abs()
                    numerator = numerator + image_masked_mean(error * confidence, normalization_mask) * valid
                    valid_pairs = valid_pairs + valid
            value = numerator / valid_pairs.clamp_min(1)
            if return_statistics:
                count = value.new_tensor(batch)
                return {'temporal': (value.sum(), count),
                        'temporal_support': ((valid_pairs > 0).float().sum(), count),
                        'temporal_pair_support': (valid_pairs.sum(), eligible_pairs),
                        'temporal_coverage': (support_mass, eligible_pairs),
                        'temporal_visibility': (visible_mass, eligible_pairs)}
            return value if return_per_sample else value.mean()
