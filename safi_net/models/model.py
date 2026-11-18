"""SAFI-Net full-image fusion with semantic features and canonical observations."""
from collections import OrderedDict
from copy import deepcopy
import torch
from torch import nn

from ..geometry import flow_warp
from ..temporal import canonicalize_inputs, time_context
from ..types import AsyncOutput, SOURCE_KEYS, TemporalInputs
from .components import (ConvBlock, ModalityEncoder, SemanticAdapter, PyramidDecoder,
                         normalized_pair, resize)
from .transport import FlowSparseTransport
from .person import TARGET_CLASSES, normalize_target_classes
from .ablation import normalize_ablation, masked_source_softmax


class SAFINet(nn.Module):
    architecture_id = 'safi_net'

    def __init__(self, config, ir_motion, vis_motion, dino):
        super().__init__()
        self.config = deepcopy(config)
        config = self.config
        self.ablation = normalize_ablation(config.get('ablation'))
        if config.get('target_gate', {}).get('segmentation_enabled', False):
            raise ValueError('SAFI-Net uses full-image fusion without segmentation')
        if config.get('fusion_region', 'all') != 'all':
            raise ValueError('SAFI-Net requires model.fusion_region=all')
        normalize_target_classes(config.get('perception', {}).get('target_classes'))
        self.ir_motion, self.vis_motion, self.dino = ir_motion, vis_motion, dino
        self.vis_motion.requires_grad_(False).eval()
        if self.dino is not None:
            self.dino.requires_grad_(False).eval()
        base, dim = int(config.get('base_channels', 32)), int(config.get('semantic_dim', 128))
        self.ir_encoder, self.vis_encoder = ModalityEncoder(1, base), ModalityEncoder(3, base)
        self.mode = self.ablation['mode']
        if self.mode != 'full':
            self.architecture_id = 'safi_net_ablation_v2'
        self.semantic_adapter = SemanticAdapter(dino.output_dim, dim) if self.ablation['semantic_features'] else None
        channels, encoded = [2*base, 4*base, 4*base, dim], [2*base, 4*base, 6*base, 8*base]
        self.ir_projection = nn.ModuleList(nn.Conv2d(a, b, 1) for a, b in zip(encoded, channels))
        self.vis_projection = nn.ModuleList(nn.Conv2d(a, b, 1) for a, b in zip(encoded, channels))
        self.semantic_projection = nn.ModuleList(nn.Conv2d(dim, c, 1) for c in channels) if self.ablation['semantic_features'] else None
        # Four validity maps, confidence, four ages, two spans and two observation flags.
        self.reliability_heads = nn.ModuleList(nn.Sequential(ConvBlock(4*c+13, c), nn.Conv2d(c, 13 if self.ablation['semantic_features'] else 12, 1))
                                               for c in channels)
        self.transport = FlowSparseTransport(dim, **config.get('transport', {})) if self.ablation['semantic_features'] else None
        self.decoder = PyramidDecoder(channels, base, plain=self.mode == 'a3_plain_decoder')
        if self.mode == 'a2_separate_weights':
            self.interpolation_weight_heads = nn.ModuleList(nn.Conv2d(c, 4, 1) for c in channels)
            self.initialize_interpolation_heads()
        self.config['target_gate'] = {'policy': 'full_image', 'segmentation_enabled': False}
        self.config['fusion_region'] = 'all'
        self.perception_microbatch = int(config.get('perception', {}).get('microbatch', 4))
        if self.perception_microbatch < 1:
            raise ValueError('perception.microbatch must be positive')

    def train(self, mode=True):
        super().train(mode)
        if self.dino is not None:
            self.dino.eval()
        self.vis_motion.train(mode and getattr(self, 'stage', 'b') == 'a')
        self.ir_motion.train(mode and getattr(self, 'stage', 'b') == 'a')
        return self

    @torch.no_grad()
    def initialize_interpolation_heads(self):
        for interpolation, shared in zip(self.interpolation_weight_heads, self.reliability_heads):
            interpolation.weight.copy_(shared[-1].weight[8:12])
            interpolation.bias.copy_(shared[-1].bias[8:12])

    @property
    def segmentation_enabled(self):
        return False

    def set_segmentation_enabled(self, enabled):
        """Keep the inference policy interface while rejecting a different model."""
        if type(enabled) is not bool or enabled:
            raise ValueError('SAFI-Net has no segmentation head; segmentation must remain off')
        return self

    def _motion(self, images, context, *, include_ir_reference=False):
        flows = {}
        # Exact observations have already been masked down to their exact slot.
        vis_reference = (images['vis0'] + images['vis1']).detach().clone()
        ir_reference = (images['ir0'] + images['ir1']).detach().clone() if include_ir_reference else None
        for modality, name, motion in ((0, 'ir', self.ir_motion), (1, 'vis', self.vis_motion)):
            missing = ~context['observed'][:, modality]
            first, second = images[name+'0'], images[name+'1']
            value = first.new_zeros(first.shape[0], 4, *first.shape[-2:], dtype=torch.float32)
            if missing.any():
                a, b = first[missing], second[missing]
                if modality == 0:
                    a, b = a.repeat(1, 3, 1, 1), b.repeat(1, 3, 1, 1)
                tau = context['tau'][missing, modality].float()
                if modality == 1:
                    with torch.set_grad_enabled(torch.is_grad_enabled() and getattr(self, 'stage', 'b') == 'a'):
                        if not hasattr(motion, 'interpolate_with_flow'):
                            if not self.config.get('_test_backbones'):
                                raise TypeError('VIS RIFE must expose interpolate_with_flow')
                            field = motion(a, b, tau)
                            reference = a * (1-tau[:, None, None, None]) + b * tau[:, None, None, None]
                        else:
                            field, reference = motion.interpolate_with_flow(a, b, tau)
                    vis_reference = vis_reference.index_copy(0, missing.nonzero().flatten(), reference.float())
                else:
                    if include_ir_reference:
                        if hasattr(motion, 'interpolate_with_flow'):
                            field, reference = motion.interpolate_with_flow(a, b, tau)
                        elif self.config.get('_test_backbones'):
                            field = motion(a, b, tau)
                            reference = a * (1-tau[:, None, None, None]) + b * tau[:, None, None, None]
                        else:
                            raise TypeError('Cascade IR RIFE must expose interpolate_with_flow')
                        # RIFE receives replicated grayscale; average its RGB
                        # reconstruction to obtain one synchronized IR channel.
                        ir_reference = ir_reference.index_copy(
                            0, missing.nonzero().flatten(), reference.float().mean(1, keepdim=True))
                    else:
                        field = motion(a, b, tau)
                value = value.index_copy(0, missing.nonzero().flatten(), field)
            flows[name+'0'], flows[name+'1'] = value[:, :2], value[:, 2:]
        return (flows, vis_reference, ir_reference) if include_ir_reference else (flows, vis_reference)

    @torch.no_grad()
    def _observations(self, inputs, context, cache):
        """Cache frozen outputs, never trainable adapters or missing-time targets.

        Cache lifetime is one objective evaluation (training), or one video.
        Input equality additionally rejects a reused timestamp after augmentation.
        Only exact frames / real interpolation endpoints are encoded.
        """
        entries = cache.setdefault('perception', OrderedDict())
        active = context['effective_valid'][:, (1, 3)]
        requested, pending, local = [], [], {}
        for row, slot in active.nonzero().tolist():
            frame = inputs.vis[row, slot]
            key = (row, float(inputs.vis_times[row, slot]), tuple(frame.shape),
                   str(frame.device), str(frame.dtype), self.segmentation_enabled)
            requested.append((row, slot, key))
            saved = entries.get(key)
            if saved is not None and torch.equal(saved[0], frame):
                entries.move_to_end(key)
                local[key] = saved[1:]
            else:
                pending.append((key, frame))
        for start in range(0, len(pending), self.perception_microbatch):
            chunk = pending[start:start+self.perception_microbatch]
            images = torch.stack([frame for _, frame in chunk])
            if not self.segmentation_enabled:
                if hasattr(self.dino, 'encode_features'):
                    levels = self.dino.encode_features(images)
                elif self.config.get('_test_backbones'):
                    levels = self.dino(images)
                else:
                    raise TypeError('Segmentation disabled requires DINO.encode_features without PMT')
                persons = images.new_zeros(len(chunk), len(TARGET_CLASSES), *images.shape[-2:])
            elif hasattr(self.dino, 'encode_observation'):
                levels, persons = self.dino.encode_observation(images)
            elif self.config.get('_test_backbones'):
                levels = self.dino(images)
                persons = images.new_zeros(len(chunk), len(TARGET_CLASSES), *images.shape[-2:])
            else:
                raise TypeError('V4 requires DINO with the pretrained PMT semantic head')
            if persons.shape != (len(chunk), len(TARGET_CLASSES), *images.shape[-2:]):
                raise ValueError('PMT perception must return six semantic mask channels')
            for i, (key, frame) in enumerate(chunk):
                value = (tuple(level[i:i+1].detach() for level in levels), persons[i:i+1].detach())
                local[key] = value
                entries[key] = (frame.detach().clone(), *value)
                entries.move_to_end(key)
        # Batch rows are distinct clips; a video has one row and four cached frames.
        while len(entries) > 4 * inputs.vis.shape[0]:
            entries.popitem(last=False)
        encoded = {(row, slot): local[key][0] for row, slot, key in requested}
        persons = inputs.vis.new_zeros(inputs.vis.shape[0], 2, len(TARGET_CLASSES), *inputs.vis.shape[-2:])
        for row, slot, key in requested:
            persons[row, slot] = local[key][1][0]
        return encoded, persons

    @torch.no_grad()
    def _reference_class_masks(self, images):
        """Run the same frozen PMT pipeline for generated references and loss targets."""
        if (images.ndim != 4 or images.shape[0] < 1 or images.shape[1] != 3
                or min(images.shape[-2:]) < 1 or not images.is_floating_point()
                or not torch.isfinite(images).all() or (images < 0).any() or (images > 1).any()):
            raise ValueError('Reference images must be finite [B,3,H,W] values in [0,1]')
        if not self.segmentation_enabled:
            return images.new_zeros(len(images), len(TARGET_CLASSES), *images.shape[-2:], dtype=torch.float32)
        masks = []
        for chunk in images.detach().split(self.perception_microbatch):
            if hasattr(self.dino, 'encode_masks'):
                classes = self.dino.encode_masks(chunk)
            elif hasattr(self.dino, 'encode_observation'):
                _, classes = self.dino.encode_observation(chunk)
            elif self.config.get('_test_backbones'):
                classes = chunk.new_zeros(chunk.shape[0], len(TARGET_CLASSES), *chunk.shape[-2:])
            else:
                raise TypeError('V5 requires DINO with the pretrained PMT semantic head')
            if (classes.shape != (chunk.shape[0], len(TARGET_CLASSES), *chunk.shape[-2:])
                    or classes.device != chunk.device or not torch.isfinite(classes).all()
                    or (classes < 0).any() or (classes > 1).any()):
                raise ValueError('PMT reference perception must return finite [B,6,H,W] masks in [0,1]')
            masks.append(classes.detach().float())
        return torch.cat(masks)

    @torch.no_grad()
    def reference_target_masks(self, vis_gt):
        """Loss-only semantic teacher; no targets enter forward or its caches.

        Return the six-class union without feathering, so supervision includes
        target boundaries that the output gate intentionally smooths inward.
        """
        return self._reference_class_masks(vis_gt).amax(1, keepdim=True).detach()

    @torch.no_grad()
    def _query_reference_masks(self, inputs, context, observed_masks, reference, cache):
        batch, _, height, width = reference.shape
        if (reference.shape != (inputs.vis.shape[0], 3, *inputs.vis.shape[-2:])
                or not torch.isfinite(reference).all()):
            raise ValueError('VIS reference must be finite [B,3,H,W]')
        result = reference.new_zeros(batch, len(TARGET_CLASSES), height, width, dtype=torch.float32)
        observed = context['observed'][:, 1]
        metadata = torch.cat((inputs.vis_times,
            context['effective_valid'][:, (1, 3)].to(inputs.vis_times.dtype),
            observed[:, None].to(inputs.vis_times.dtype), inputs.query_time[:, None]), 1).cpu().tolist()
        seen_rows = [row for row, values in enumerate(metadata) if values[4]]
        if seen_rows:
            slots = [0 if metadata[row][2] else 1 for row in seen_rows]
            result[seen_rows] = observed_masks[seen_rows, slots].detach().float()
        # Separate from endpoint features: generated masks must not evict real
        # observations. Only detached RGB and masks are kept, never DINO features.
        entries = cache.setdefault('reference_perception', OrderedDict())
        pending, candidates = [], []
        for row, (t0, t1, _, _, seen, query) in enumerate(metadata):
            if seen:
                continue
            frame = reference[row].detach()
            key = (row, query, (t0, t1),
                   tuple(frame.shape), str(frame.device), str(frame.dtype), self.segmentation_enabled)
            saved = entries.get(key)
            if saved is None:
                pending.append((row, key, frame))
            else:
                candidates.append((row, key, frame, saved))
        if candidates:
            matches = torch.stack([(saved[0] == frame).all()
                for _, _, frame, saved in candidates]).cpu().tolist()
            for (row, key, frame, saved), matches_input in zip(candidates, matches):
                if matches_input:
                    entries.move_to_end(key)
                    result[row] = saved[1]
                else:
                    pending.append((row, key, frame))
        pending.sort(key=lambda item: item[0])
        for start in range(0, len(pending), self.perception_microbatch):
            chunk = pending[start:start+self.perception_microbatch]
            classes = self._reference_class_masks(torch.stack([frame for _, _, frame in chunk]))
            for (row, key, frame), masks in zip(chunk, classes):
                result[row] = masks
                entries[key] = (frame.detach().clone(), masks.detach().clone())
                entries.move_to_end(key)
        while len(entries) > 2 * batch:
            entries.popitem(last=False)
        return result

    def _semantics(self, images, flows, context, encoded):
        seen = context['observed'][:, 1]
        batch = seen.shape[0]
        semantic = confidence = details = None
        # Only real exact observations are passed to DINO; no padded neighbour tokens.
        if seen.any():
            selected = [(i, 0 if context['effective_valid'][i, 1] else 1)
                        for i in seen.nonzero().flatten().tolist()]
            levels = tuple(torch.cat([encoded[key][level] for key in selected]) for level in range(4))
            exact = self.semantic_adapter(levels)
            semantic = exact.new_zeros(batch, *exact.shape[1:]).index_copy(0, seen.nonzero().flatten(), exact)
            confidence = exact.new_zeros(batch, 1, *exact.shape[-2:])
            confidence[seen] = 1
        missing = ~seen
        if missing.any():
            a, b = images['vis0'][missing], images['vis1'][missing]
            rows = missing.nonzero().flatten().tolist()
            s0, s1 = (self.semantic_adapter(tuple(torch.cat([encoded[(row, slot)][level]
                                                             for row in rows]) for level in range(4)))
                      for slot in (0, 1))
            tau = context['tau'][missing, 1].float()
            if self.ablation['aggregation'] == 'standard':
                first, v0 = flow_warp(s0, flows['vis0'][missing])
                second, v1 = flow_warp(s1, flows['vis1'][missing])
                tau = tau[:, None, None, None]
                value = normalized_pair(first, second, (1-tau)*v0, tau*v1)
                conf = ((1-tau)*v0 + tau*v1).detach()
            else:
                with torch.no_grad():
                    endpoint = self.vis_motion(a, b, torch.zeros_like(tau))[:, 2:]
                    reverse = self.vis_motion(a, b, torch.ones_like(tau))[:, :2]
                    warped_reverse, valid = flow_warp(reverse, endpoint)
                    error = (endpoint + warped_reverse).square().sum(1, keepdim=True)
                    scale = endpoint.square().sum(1, keepdim=True) + warped_reverse.square().sum(1, keepdim=True)
                    match_valid = valid * (error <= 0.01*scale+0.5)
                matched, conf, details = self.transport(s0, s1, endpoint)
                details['reliable'] = resize(match_valid, s0.shape[-2:]).flatten(1).detach()
                details['batch_indices'] = missing.nonzero().flatten()
                first, v0 = flow_warp((s0+matched)*0.5, flows['vis0'][missing])
                second, v1 = flow_warp(s1, flows['vis1'][missing])
                tau = tau[:, None, None, None]
                value = normalized_pair(first, second, (1-tau)*v0, tau*v1)
                conf, conf_valid = flow_warp(conf, flows['vis0'][missing])
                conf = conf * conf_valid
            if semantic is None:
                semantic = value.new_zeros(batch, *value.shape[1:])
                confidence = conf.new_zeros(batch, *conf.shape[1:])
            semantic = semantic.index_copy(0, missing.nonzero().flatten(), value)
            confidence = confidence.index_copy(0, missing.nonzero().flatten(), conf)
        return semantic, confidence, details

    @torch.no_grad()
    def _cascade_perception(self, inputs, reference, cache):
        """Encode only the synchronized query image, never endpoint features."""
        entries = cache.setdefault('cascade_perception', OrderedDict())
        result = []
        for row, frame in enumerate(reference.detach()):
            key = (row, float(inputs.query_time[row]), tuple(frame.shape), str(frame.device),
                   str(frame.dtype), self.segmentation_enabled)
            saved = entries.get(key)
            if saved is not None and torch.equal(saved[0], frame):
                levels, masks = saved[1:]
                entries.move_to_end(key)
            else:
                image = frame[None]
                if not self.segmentation_enabled:
                    if hasattr(self.dino, 'encode_features'):
                        levels = self.dino.encode_features(image)
                    elif self.config.get('_test_backbones'):
                        levels = self.dino(image)
                    else:
                        raise TypeError('Segmentation disabled requires DINO.encode_features without PMT')
                    masks = image.new_zeros(1, len(TARGET_CLASSES), *image.shape[-2:])
                elif hasattr(self.dino, 'encode_observation'):
                    levels, masks = self.dino.encode_observation(image)
                elif self.config.get('_test_backbones'):
                    levels = self.dino(image)
                    masks = image.new_zeros(1, len(TARGET_CLASSES), *image.shape[-2:])
                else:
                    raise TypeError('Cascade requires the frozen DINO/PMT observation encoder')
                levels, masks = tuple(value.detach() for value in levels), masks.detach()
                entries[key] = (frame.clone(), levels, masks)
            result.append((levels, masks))
        while len(entries) > 2 * len(reference):
            entries.popitem(last=False)
        return (tuple(torch.cat([item[0][level] for item in result]) for level in range(4)),
                torch.cat([item[1] for item in result]))

    def _perception_for_query(self, inputs, images, flows, context, reference, cache):
        """Extract DINO features without semantic masks or motion selection."""
        use_semantic = self.ablation['semantic_features']
        use_gate = self.segmentation_enabled
        masks = reference.new_zeros(len(reference), len(TARGET_CLASSES), *reference.shape[-2:])
        confidence = reference.new_zeros(len(reference), 1, *reference.shape[-2:])
        semantic, details = None, None
        if self.ablation['cascade']:
            if use_semantic or use_gate:
                levels, query_masks = self._cascade_perception(inputs, reference, cache)
                if use_semantic:
                    semantic = self.semantic_adapter(levels)
                    confidence = torch.ones_like(semantic[:, :1])
                if use_gate:
                    masks = query_masks
        elif use_semantic:
            encoded, observed_masks = self._observations(inputs, context, cache)
            semantic, confidence, details = self._semantics(images, flows, context, encoded)
            if use_gate:
                masks = self._query_reference_masks(inputs, context, observed_masks, reference.detach(), cache)
        elif use_gate:
            # No endpoint DINO features are needed without the semantic path.
            # Reuse the query cache also for observed references in this mode.
            query_context = {**context, 'observed': torch.zeros_like(context['observed'])}
            observed_masks = masks.new_zeros(len(reference), 2, *masks.shape[1:])
            masks = self._query_reference_masks(inputs, query_context, observed_masks, reference.detach(), cache)
        zero = reference.new_zeros(len(reference), 1, *reference.shape[-2:])
        evidence = {
            'class_masks': masks, 'person_mask': zero, 'vehicle_mask': zero,
            'semantic_mask': zero, 'motion_mask': zero,
            'ir_motion_mask': zero, 'vis_motion_mask': zero,
            'target_gate': torch.ones_like(zero),
            'motion_diagnostics': {'ir': [], 'vis': []},
            'diagnostics': {'pairs': [], 'motion': {'ir': [], 'vis': []},
                            'segmentation_enabled': False, 'fusion_region': 'all'},
        }
        return semantic, confidence, details, evidence

    def forward(self, inputs: TemporalInputs, *, gate_cache=None) -> AsyncOutput:
        inputs = canonicalize_inputs(inputs)
        context = time_context(inputs)
        cache = {} if gate_cache is None else gate_cache
        raw = (inputs.ir[:, 0], inputs.vis[:, 0], inputs.ir[:, 1], inputs.vis[:, 1])
        active = context['effective_valid']
        images = {key: torch.where(active[:, i, None, None, None], x, torch.zeros_like(x))
                  for i, (key, x) in enumerate(zip(SOURCE_KEYS, raw))}
        if self.ablation['cascade']:
            flows, vis_reference, ir_reference = self._motion(images, context, include_ir_reference=True)
            references = {'ir': ir_reference, 'vis': vis_reference}
            fusion_images = {
                key: torch.where(active[:, i, None, None, None], references[key[:-1]],
                                 torch.zeros_like(references[key[:-1]]))
                for i, key in enumerate(SOURCE_KEYS)}
            # Retain slot identities for auxiliary reliability supervision,
            # but both endpoints now carry only the same interpolated image.
            fusion_active = active
            fusion_flows = {key: torch.zeros_like(value) for key, value in flows.items()}
            # Encoding starts after image interpolation. No real endpoint
            # feature reaches the synchronized fusion branch.
            query_features = {'ir': self.ir_encoder(ir_reference), 'vis': self.vis_encoder(vis_reference)}
            features = {key: query_features[key[:-1]] for key in SOURCE_KEYS}
        else:
            fusion_images, fusion_active = images, active
            features = {key: (self.ir_encoder if key.startswith('ir') else self.vis_encoder)(x)
                        for key, x in images.items()}
            flows, vis_reference = self._motion(images, context)
            fusion_flows = flows
        semantic, confidence, details, evidence = self._perception_for_query(
            inputs, images, flows, context, vis_reference, cache)
        condition = torch.cat((context['offsets'], context['spans'], context['observed']), 1).float()
        pyramid, ir_private, vis_private = [], [], []
        interpolation_weights = []
        for level, head in enumerate(self.reliability_heads):
            warped, masks = [], []
            for i, key in enumerate(SOURCE_KEYS):
                project = self.ir_projection if key.startswith('ir') else self.vis_projection
                value, valid = flow_warp(project[level](features[key][level]), fusion_flows[key])
                valid = valid * fusion_active[:, i, None, None, None]
                warped.append(value * valid)
                masks.append(valid)
            size = warped[0].shape[-2:]
            mask = torch.cat(masks, 1)
            head_input = torch.cat((*warped, mask, resize(confidence, size),
                                    condition[:, :, None, None].expand(-1, -1, *size)), 1)
            if self.mode == 'a2_separate_weights':
                shared_evidence = head[0](head_input)
                logits = head[1](shared_evidence)
            else:
                logits = head(head_input)
            visibility = logits[:, :4].float().sigmoid()*mask
            reliability = logits[:, 4:8].float().sigmoid()*visibility
            source_logits = logits[:, 8:12].float()
            if self.ablation['aggregation'] == 'proposed':
                source_logits = source_logits + reliability.clamp_min(1e-6).log()
            source_logits = source_logits.masked_fill(mask == 0, -torch.inf)
            if self.ablation['semantic_features']:
                weights = torch.cat((source_logits, logits[:, 12:13].float()), 1).softmax(1)
                prior = self.semantic_projection[level](resize(semantic, size))
                fused_features = sum(weights[:, i:i+1]*v for i, v in enumerate(warped)) + weights[:, 4:5]*prior
            else:
                source_weights = masked_source_softmax(source_logits, mask > 0)
                weights = source_weights if self.mode == 'a1_no_dino' else torch.cat((source_weights, torch.zeros_like(source_weights[:, :1])), 1)
                fused_features = sum(weights[:, i:i+1]*v for i, v in enumerate(warped))
            pyramid.append(fused_features)
            reconstruction = weights
            if self.mode == 'a2_separate_weights':
                beta_logits = self.interpolation_weight_heads[level](shared_evidence).float()
                beta_logits = beta_logits + reliability.clamp_min(1e-6).log()
                beta_ir = masked_source_softmax(beta_logits[:, [0, 2]], mask[:, [0, 2]] > 0)
                beta_vis = masked_source_softmax(beta_logits[:, [1, 3]], mask[:, [1, 3]] > 0)
                reconstruction = torch.stack((beta_ir[:, 0], beta_vis[:, 0], beta_ir[:, 1], beta_vis[:, 1]), 1)
                interpolation_weights.append(reconstruction)
            ir_private.append(normalized_pair(warped[0], warped[2], reconstruction[:, 0:1], reconstruction[:, 2:3]))
            vis_private.append(normalized_pair(warped[1], warped[3], reconstruction[:, 1:2], reconstruction[:, 3:4]))
            if level == 0:
                shallow = (weights, reliability, visibility)
        weights, reliability, visibility = shallow
        pixel_weights = resize(interpolation_weights[0] if interpolation_weights else weights[:, :4], inputs.ir.shape[-2:])
        pixels, masks = [], []
        for i, key in enumerate(SOURCE_KEYS):
            value, valid = flow_warp(fusion_images[key], fusion_flows[key])
            valid = valid * fusion_active[:, i, None, None, None]
            pixels.append(value * valid)
            masks.append(valid)
        pixel_weights = pixel_weights * torch.cat(masks, 1)
        ir_base = normalized_pair(pixels[0], pixels[2], pixel_weights[:, :1], pixel_weights[:, 2:3])
        vis_base = normalized_pair(pixels[1], pixels[3], pixel_weights[:, 1:2], pixel_weights[:, 3:4])
        ir_exact, vis_exact = images['ir0']+images['ir1'], images['vis0']+images['vis1']
        seen_ir, seen_vis = (context['observed'][:, i, None, None, None] for i in (0, 1))
        ir_base, vis_base = torch.where(seen_ir, ir_exact, ir_base), torch.where(seen_vis, vis_exact, vis_base)
        luminance_residual, ir_aux, vis_aux = self.decoder(pyramid, ir_private, vis_private, ir_base, vis_base)
        # Enforced after the learned heads, so training cannot break exact identity.
        ir_aux, vis_aux = torch.where(seen_ir, ir_exact, ir_aux), torch.where(seen_vis, vis_exact, vis_aux)
        # decoder supplies .5*tanh(logit); bound the positive lift to .25.
        # A common RGB headroom preserves chroma without channel-wise clipping.
        gate = evidence['target_gate'].detach()
        if self.mode == 'a3_plain_decoder':
            from ..losses import luminance
            fused = luminance_residual
            fusion_delta = fused - vis_reference
            delta = luminance(fused) - luminance(vis_reference)
        else:
            delta = torch.minimum((0.5*luminance_residual).clamp_min(0),
                                  (1-vis_reference.amax(1, keepdim=True)).clamp_min(0))
            fusion_delta = gate * delta
            fused = vis_reference + fusion_delta
        return AsyncOutput(fused, ir_aux, vis_aux, ir_base, vis_base, flows,
                           reliability, visibility, weights, context['observed'], details,
                           vis_reference, evidence['person_mask'], evidence['motion_mask'],
                           gate, fusion_delta, evidence.get('diagnostics'),
                           vehicle_mask=evidence['vehicle_mask'], semantic_mask=evidence['semantic_mask'],
                           class_masks=evidence['class_masks'], pre_gate_delta=delta,
                           ir_motion_mask=evidence['ir_motion_mask'],
                           vis_motion_mask=evidence['vis_motion_mask'],
                           motion_diagnostics=evidence['motion_diagnostics'],
                           fusion_representation='direct_rgb' if self.mode == 'a3_plain_decoder' else 'thermal_delta',
                           interpolation_weights=tuple(interpolation_weights))
