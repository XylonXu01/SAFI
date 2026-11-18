"""Inference kernels for full-resolution, FP32 SAFI-Net."""
from collections import OrderedDict

import torch
import torch.nn.functional as F

from ..geometry import flow_warp
from ..models.backbones import DINOv3Backbone, RIFEBackbone
from ..models.components import normalized_pair, resize
from ..models.model import SAFINet
from ..models.person import letterbox_image, restore_patch_features
from ..temporal import canonicalize_inputs, time_context
from ..third_party.rife.warplayer import warp
from ..types import AsyncOutput, SOURCE_KEYS


class FastRIFE(RIFEBackbone):
    def forward(self, x0, x1, tau):
        height, width = x0.shape[-2:]
        with torch.autocast(device_type=x0.device.type, enabled=False):
            images = F.pad(torch.cat((x0.float(), x1.float()), 1),
                           (0, (-width) % 32, 0, (-height) % 32), mode='replicate')
            a, b = images[:, :3], images[:, 3:6]
            t = torch.ones_like(images[:, :1]) * tau.float().reshape(-1, 1, 1, 1)
            flow, mask = self.flownet.block0(torch.cat((a, b, t), 1), None, 4)
            for block, scale in ((self.flownet.block1, 2), (self.flownet.block2, 1)):
                wa, wb = warp(a, flow[:, :2]), warp(b, flow[:, 2:])
                delta_flow, delta_mask = block(torch.cat((a, b, t, wa, wb, mask), 1), flow, scale)
                flow, mask = flow + delta_flow, mask + delta_mask
            return flow[..., :height, :width]


class FastDINO(DINOv3Backbone):
    def encode_features(self, image):
        value, geometry = letterbox_image(image, size=640)
        hidden = self.model((value - self.mean) / self.std, output_hidden_states=True).hidden_states
        return tuple(restore_patch_features(self.model.norm(hidden[i])[:, 5:].transpose(1, 2)
                                            .reshape(len(image), 384, 40, 40), geometry)
                     for i in self.layers)


class FastModel(SAFINet):
    """Frozen runtime with a cache scoped to one video stream."""

    def train(self, mode=True):
        if mode:
            raise RuntimeError('Fast SAFI-Net is inference-only')
        return super().train(False)

    def _encode(self, images, inputs, active, cache):
        projected = {}
        for i, key in enumerate(SOURCE_KEYS):
            if not active[:, i].any():
                continue
            mod, slot = key[:-1], int(key[-1])
            image = images[key]
            encoder, projection = getattr(self, mod + '_encoder'), getattr(self, mod + '_projection')
            entries = cache.setdefault('encoder_' + mod, OrderedDict())
            signature = (tuple(getattr(inputs, mod + '_times')[:, slot].tolist()),
                         tuple(image.shape), image.device, image.dtype)
            saved = entries.get(signature)
            if saved is not None and torch.equal(saved[0], image):
                values = saved[1]
            else:
                values = tuple(p(f) for p, f in zip(projection, encoder(image)))
                entries[signature] = (image.clone(), values)
            entries.move_to_end(signature)
            while len(entries) > 4:
                entries.popitem(last=False)
            projected[key] = values
        template = next(iter(projected.values()))
        for key in SOURCE_KEYS:
            if key not in projected:
                projected[key] = tuple(torch.zeros_like(value) for value in template)
        return projected

    def _adapt(self, levels, cache):
        entries = cache.setdefault('adapted', OrderedDict())
        key = tuple(value.data_ptr() for value in levels)
        saved = entries.get(key)
        if saved is None:
            # Retain the input tensors so their storage addresses remain unique.
            saved = (levels, self.semantic_adapter(levels))
            entries[key] = saved
        entries.move_to_end(key)
        while len(entries) > 8:
            entries.popitem(last=False)
        return saved[1]

    def _cached_semantics(self, images, flows, context, encoded, cache):
        values, confidences = [], []
        for row in range(len(context['observed'])):
            first = self._adapt(encoded[(row, 0)], cache)
            if context['observed'][row, 1]:
                values.append(first)
                confidences.append(torch.ones_like(first[:, :1]))
                continue

            a, b = images['vis0'][row:row+1], images['vis1'][row:row+1]
            second = self._adapt(encoded[(row, 1)], cache)
            tau = context['tau'][row:row+1, 1].float()
            endpoint = self.vis_motion(a, b, tau * 0)[:, 2:]
            matched, confidence, _ = self.transport(first, second, endpoint)
            vis0, vis1 = flows['vis0'][row:row+1], flows['vis1'][row:row+1]
            first, valid0 = flow_warp((first + matched) * .5, vis0)
            second, valid1 = flow_warp(second, vis1)
            t = tau[:, None, None, None]
            values.append(normalized_pair(first, second, (1 - t) * valid0, t * valid1))
            confidence, valid = flow_warp(confidence, vis0)
            confidences.append(confidence * valid)
        return torch.cat(values), torch.cat(confidences)

    @torch.inference_mode()
    def forward(self, inputs, *, gate_cache=None):
        cache = {} if gate_cache is None else gate_cache
        if cache.get('_fast_model') != id(self):
            cache.clear()
            cache['_fast_model'] = id(self)
        inputs = canonicalize_inputs(inputs)
        context = time_context(inputs)
        active = context['effective_valid']
        raw = (inputs.ir[:, 0], inputs.vis[:, 0], inputs.ir[:, 1], inputs.vis[:, 1])
        images = {key: torch.where(active[:, i, None, None, None], image, torch.zeros_like(image))
                  for i, (key, image) in enumerate(zip(SOURCE_KEYS, raw))}
        features = self._encode(images, inputs, active, cache)
        flows, vis_reference = self._motion(images, context)
        encoded, _ = self._observations(inputs, context, cache)
        semantic, confidence = self._cached_semantics(images, flows, context, encoded, cache)
        condition = torch.cat((context['offsets'], context['spans'], context['observed']), 1).float()
        pyramid, ir_private, vis_private = [], [], []
        for level, head in enumerate(self.reliability_heads):
            warped, masks = [], []
            for i, key in enumerate(SOURCE_KEYS):
                value = features[key][level]
                if active[:, i].any():
                    value, valid = flow_warp(value, flows[key])
                else:
                    valid = value.new_zeros(len(value), 1, *value.shape[-2:])
                valid = valid * active[:, i, None, None, None]
                warped.append(value * valid)
                masks.append(valid)
            size = warped[0].shape[-2:]
            mask = torch.cat(masks, 1)
            inputs_to_head = (*warped, mask, resize(confidence, size),
                              condition[:, :, None, None].expand(-1, -1, *size))
            logits = head(torch.cat(inputs_to_head, 1))
            visibility = logits[:, :4].float().sigmoid() * mask
            reliability = logits[:, 4:8].float().sigmoid() * visibility
            source = (logits[:, 8:12].float() + reliability.clamp_min(1e-6).log())
            source = source.masked_fill(mask == 0, -torch.inf)
            weights = torch.cat((source, logits[:, 12:13].float()), 1).softmax(1)
            prior = self.semantic_projection[level](resize(semantic, size))
            pyramid.append(sum(weights[:, i:i+1] * v for i, v in enumerate(warped))
                           + weights[:, 4:5] * prior)
            ir_private.append(normalized_pair(warped[0], warped[2], weights[:, :1], weights[:, 2:3]))
            vis_private.append(normalized_pair(warped[1], warped[3], weights[:, 1:2], weights[:, 3:4]))
            if level == 0:
                shallow = weights, reliability, visibility
        weights, reliability, visibility = shallow

        pixel_weights = resize(weights[:, :4], inputs.ir.shape[-2:])
        pixels, masks = [], []
        for i, key in enumerate(SOURCE_KEYS):
            value, valid = flow_warp(images[key], flows[key])
            valid = valid * active[:, i, None, None, None]
            pixels.append(value * valid)
            masks.append(valid)
        pixel_weights = pixel_weights * torch.cat(masks, 1)
        ir_base = normalized_pair(pixels[0], pixels[2], pixel_weights[:, :1], pixel_weights[:, 2:3])
        vis_base = normalized_pair(pixels[1], pixels[3], pixel_weights[:, 1:2], pixel_weights[:, 3:4])
        ir_exact, vis_exact = images['ir0'] + images['ir1'], images['vis0'] + images['vis1']
        ir_seen, vis_seen = (context['observed'][:, i, None, None, None] for i in (0, 1))
        ir_base = torch.where(ir_seen, ir_exact, ir_base)
        vis_base = torch.where(vis_seen, vis_exact, vis_base)
        x = None
        for i in (3, 2, 1, 0):
            values = [pyramid[i], ir_private[i], vis_private[i]]
            if x is not None:
                values.append(resize(x, pyramid[i].shape[-2:]))
            x = self.decoder.blocks[i](torch.cat(values, 1))
        x = self.decoder.full(torch.cat((resize(x, ir_base.shape[-2:]), ir_base, vis_base), 1))
        ir_aux = vis_aux = None
        if self.return_aux:
            ir_aux = torch.where(ir_seen, ir_exact, (ir_base + .2 * self.decoder.ir(x).tanh()).clamp(0, 1))
            vis_aux = torch.where(vis_seen, vis_exact, (vis_base + .2 * self.decoder.vis(x).tanh()).clamp(0, 1))
        raw_delta = (.25 * self.decoder.fused(x).tanh()).clamp_min(0)
        delta = torch.minimum(raw_delta, (1 - vis_reference.amax(1, keepdim=True)).clamp_min(0))
        zero = torch.zeros_like(delta)
        motion_diagnostics = {'ir': [], 'vis': []}
        gate_diagnostics = {'pairs': [], 'motion': motion_diagnostics,
                            'segmentation_enabled': False, 'fusion_region': 'all'}
        return AsyncOutput((vis_reference + delta).float(), ir_aux, vis_aux, ir_base, vis_base,
                           flows, reliability, visibility, weights, context['observed'],
                           vis_reference=vis_reference, fusion_delta=delta, pre_gate_delta=delta,
                           target_gate=torch.ones_like(delta), gate_diagnostics=gate_diagnostics,
                           motion_diagnostics=motion_diagnostics, person_mask=zero, motion_mask=zero,
                           vehicle_mask=zero, semantic_mask=zero, ir_motion_mask=zero, vis_motion_mask=zero)
