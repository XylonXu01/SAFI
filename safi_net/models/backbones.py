"""Offline-only adapters for the original official DINOv3 and RIFEm assets."""
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from ..third_party.rife.IFNet_m import IFNet_m
from .person import FrozenPMTPersonHead, letterbox_image, restore_patch_features


class RIFEBackbone(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.flownet = IFNet_m()
        if checkpoint is not None:
            checkpoint = Path(checkpoint)
            if not checkpoint.is_file():
                raise FileNotFoundError(f'Official RIFEm checkpoint required: {checkpoint}')
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
            state = {key.removeprefix('module.'): value for key, value in state.items()}
            self.flownet.load_state_dict(state, strict=True)
        self.set_trainable(False)

    def set_trainable(self, enabled):
        self.requires_grad_(False)
        for block in (self.flownet.block0, self.flownet.block1, self.flownet.block2):
            block.requires_grad_(enabled)
        self.trainable_flow = bool(enabled)

    def train(self, mode=True):
        super().train(mode)
        if not self.trainable_flow:
            self.flownet.eval()
        return self

    def forward(self, x0, x1, tau):
        return self._interpolate(x0, x1, tau, refined=False)

    def interpolate_with_flow(self, x0, x1, tau):
        """Return final bidirectional flow and the author's refined RGB frame."""
        return self._interpolate(x0, x1, tau, refined=True)

    def _interpolate(self, x0, x1, tau, *, refined):
        if x0.shape != x1.shape or x0.shape[1] != 3:
            raise ValueError('RIFE expects two equally shaped RGB images')
        h, w = x0.shape[-2:]
        with torch.autocast(device_type=x0.device.type, enabled=False):
            images = torch.cat((x0.float(), x1.float()), 1)
            images = F.pad(images, (0, (-w) % 32, 0, (-h) % 32), mode='replicate')
            # Tensor tau broadcasts per sample. Preserve the caller's no_grad context.
            output = self.flownet(images, timestep=tau.float().reshape(-1, 1, 1, 1),
                                  returnflow=not refined)
        if refined:
            flows, _, merged, *_ = output
            return flows[-1][..., :h, :w], merged[2][..., :h, :w]
        return output[..., :h, :w]


class DINOv3Backbone(nn.Module):
    output_dim = 384

    def __init__(self, checkpoint, layers=(3, 6, 9, 12), *, person_checkpoint=None,
                 person_config=None):
        super().__init__()
        if not layers or any(i < 1 or i > 12 for i in layers):
            raise ValueError('DINO layers must be between 1 and 12')
        from transformers import DINOv3ViTConfig, DINOv3ViTModel
        config = DINOv3ViTConfig(hidden_size=384, num_hidden_layers=12,
                               num_attention_heads=6, intermediate_size=1536,
                               patch_size=16, num_register_tokens=4)
        self.model = DINOv3ViTModel(config)
        if checkpoint is not None:
            from safetensors.torch import load_file
            checkpoint = Path(checkpoint)
            if checkpoint.suffix != '.safetensors' or not checkpoint.is_file():
                raise FileNotFoundError(f'DINOv3 safetensors checkpoint required: {checkpoint}')
            state = load_file(str(checkpoint), device='cpu')
            state = {f'model.{key}' if key.startswith('layer.') else key: value for key, value in state.items()}
            self.model.load_state_dict(state, strict=True)
        self.layers = tuple(layers)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        perception = dict(person_config or {})
        self.perception_microbatch = int(perception.get('microbatch', 4))
        if self.perception_microbatch < 1:
            raise ValueError('Person perception microbatch must be positive')
        if person_checkpoint is not None:
            if self.layers != (3, 6, 9, 12):
                raise ValueError('Pretrained PMT-S requires DINO layers 3/6/9/12')
            if perception.get('kind', 'pmt_small_coco') != 'pmt_small_coco':
                raise ValueError('Only the pinned pretrained PMT-S COCO head is supported')
            if int(perception.get('input_size', 640)) != 640:
                raise ValueError('Pretrained PMT-S uses the fixed 640 letterbox input')
            self.person_head = FrozenPMTPersonHead(person_checkpoint,
                score_threshold=perception.get('person_score_threshold', .8),
                mask_threshold=perception.get('mask_threshold', .5),
                overlap_threshold=perception.get('overlap_mask_area_threshold', .8),
                expected_sha256=perception.get('expected_sha256'),
                target_classes=perception.get('target_classes'))
        else:
            self.person_head = None
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, image):
        h, w = (max(16, d // 16 * 16) for d in image.shape[-2:])
        value = F.interpolate(image, size=(h, w), mode='bilinear', align_corners=False)
        hidden = self.model((value-self.mean)/self.std, output_hidden_states=True).hidden_states
        return tuple(self.model.norm(hidden[i])[:, 5:].transpose(1, 2)
                     .reshape(image.shape[0], 384, h//16, w//16).contiguous() for i in self.layers)

    @torch.no_grad()
    def encode_observation(self, image):
        """One shared frozen encoder pass per microbatch for fusion and six target classes.

        PMT consumes full normalized tokens and the same encoder's RoPE. Fusion
        receives those normalized patch tokens before PMT's task-specific BN,
        resampled from the letterbox to the native observation patch grid.
        """
        return self._encode_perception(image, need_features=True, need_masks=True)

    @torch.no_grad()
    def encode_features(self, image):
        """Fusion features with exactly the shared encoder's preprocessing.

        This differs deliberately from ``forward``: the PMT-compatible 640
        letterbox and restored patch geometry are retained. It also supports
        measuring an independent feature encoder without changing numerics.
        """
        return self._encode_perception(image, need_features=True, need_masks=False)[0]

    @torch.no_grad()
    def encode_masks(self, image):
        """Segmentation-only inference, without unused feature resampling."""
        return self._encode_perception(image, need_features=False, need_masks=True)[1]

    def _encode_perception(self, image, *, need_features, need_masks):
        if need_masks and self.person_head is None:
            raise RuntimeError('V4 observation perception requires the pretrained PMT-S head')
        if image.ndim != 4 or image.shape[0] < 1:
            raise ValueError('Observation batch cannot be empty and must be [B,C,H,W]')
        chunks, masks = [], []
        for images in image.split(self.perception_microbatch):
            value, geometry = letterbox_image(images, size=640)
            value = (value-self.mean)/self.std
            hidden = self.model(value, output_hidden_states=True).hidden_states
            levels = tuple(self.model.norm(hidden[i]) for i in self.layers)
            if need_masks:
                # This deterministic positional calculation does not run the encoder.
                rope = self.model.rope_embeddings(value)
                masks.append(self.person_head(levels, rope, geometry))
            if need_features:
                patches = tuple(level[:, 5:].transpose(1, 2).reshape(images.shape[0], 384, 40, 40)
                                for level in levels)
                chunks.append(tuple(restore_patch_features(patch, geometry) for patch in patches))
        return (tuple(torch.cat([chunk[level] for chunk in chunks]) for level in range(len(self.layers)))
                if need_features else None, torch.cat(masks) if need_masks else None)
