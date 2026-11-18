"""Frozen pretrained PMT-S person and vehicle perception; no GT enters this module.

Panoptic postprocessing follows tue-mps/pmt's training/lightning_module.py.
Source attribution and MIT/Apache-2.0 notices live in third_party/pmt/.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..third_party.pmt import PMTHead

PMT_REPOSITORY = 'tue-mps/coco_panoptic_pmt_small_640_dinov3'
PMT_REVISION = 'd6dbeb2f4e7bdd9c2d801f6e5c534e318759011c'
PERSON_CLASS = 0
NUM_CLASSES = 133
TARGET_CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck')
TARGET_CLASS_IDS = (0, 1, 2, 3, 5, 7)


def normalize_target_classes(value=None):
    """Keep the six output channels stable across configs and checkpoints."""
    if value is None:
        return TARGET_CLASSES
    if not isinstance(value, (list, tuple)) or tuple(value) != TARGET_CLASSES:
        raise ValueError(f'v4 perception.target_classes must be {list(TARGET_CLASSES)} in this order')
    return tuple(value)


@dataclass(frozen=True)
class Letterbox:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    size: int = 640


def letterbox_image(image, size=640):
    """Resize preserving aspect ratio; pad right/bottom with RGB black."""
    if image.ndim != 4 or image.shape[1] != 3 or min(image.shape) < 1:
        raise ValueError('Person perception expects nonempty [B,3,H,W] RGB images')
    height, width = image.shape[-2:]
    scale = min(size/height, size/width)
    resized_h, resized_w = max(1, round(height*scale)), max(1, round(width*scale))
    value = F.interpolate(image.float(), (resized_h, resized_w), mode='bilinear',
                          align_corners=False, antialias=True)
    value = F.pad(value, (0, size-resized_w, 0, size-resized_h), value=0.)
    return value, Letterbox(height, width, resized_h, resized_w, size)


def restore_mask_logits(mask_logits, geometry):
    """Undo padding before resampling to the caller's image coordinates."""
    value = F.interpolate(mask_logits.float(), (geometry.size, geometry.size),
                          mode='bilinear', align_corners=False)
    value = value[..., :geometry.resized_height, :geometry.resized_width]
    return F.interpolate(value, (geometry.original_height, geometry.original_width),
                          mode='bilinear', align_corners=False)


def restore_patch_features(features, geometry):
    """Sample valid fractional patch coordinates; never round a token crop."""
    height = max(1, geometry.original_height//16)
    width = max(1, geometry.original_width//16)
    y = ((torch.arange(height, device=features.device, dtype=torch.float32)+.5)
         * geometry.resized_height/height/geometry.size*2-1)
    x = ((torch.arange(width, device=features.device, dtype=torch.float32)+.5)
         * geometry.resized_width/width/geometry.size*2-1)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    grid = torch.stack((xx, yy), -1)[None].expand(features.shape[0], -1, -1, -1)
    # Float32 supports CPU and avoids BF16 grid coordinate quantization.
    value = F.grid_sample(features.float(), grid, mode='bilinear', padding_mode='border',
                           align_corners=False)
    return value.to(features.dtype)


@torch.no_grad()
def semantic_masks_from_logits(mask_logits, class_logits, *, target_class_ids=TARGET_CLASS_IDS,
                               score_threshold=.8, mask_threshold=.5, overlap_threshold=.8):
    """All panoptic classes compete before accepting the selected semantic classes.

    The output has one channel per selected class. No-object queries never enter
    pixel competition; unselected classes still retain their own pixels.
    """
    if mask_logits.ndim != 4 or class_logits.ndim != 3:
        raise ValueError('Expected masks [B,Q,H,W] and classes [B,Q,K+1]')
    if mask_logits.shape[:2] != class_logits.shape[:2]:
        raise ValueError('Mask and class query dimensions differ')
    if not torch.isfinite(mask_logits).all() or not torch.isfinite(class_logits).all():
        raise ValueError('PMT logits must be finite')
    no_object = class_logits.shape[-1]-1
    ids = tuple(target_class_ids)
    if not ids or len(set(ids)) != len(ids) or any(
            isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < no_object for i in ids):
        raise ValueError('Target class IDs must be unique valid foreground class indices')
    for threshold in (score_threshold, mask_threshold, overlap_threshold):
        if not 0 <= threshold <= 1:
            raise ValueError('Semantic thresholds must be in [0,1]')
    probability = class_logits.float().softmax(-1)
    scores, classes = probability.max(-1)
    result = mask_logits.new_zeros(mask_logits.shape[0], len(ids), *mask_logits.shape[-2:],
                                  dtype=torch.float32)
    # All retained panoptic queries still compete. Mapping a winning query to
    # its semantic channel afterwards removes the per-query Python/CUDA syncs.
    channels = torch.full((no_object+1,), -1, device=classes.device, dtype=torch.long)
    channels[list(ids)] = torch.arange(len(ids), device=classes.device)
    output_channels = torch.arange(len(ids), device=classes.device)[:, None, None]
    for batch in range(mask_logits.shape[0]):
        keep = (classes[batch] != no_object) & (scores[batch] > score_threshold)
        if not keep.any():
            continue
        labels, confidence = classes[batch, keep], scores[batch, keep]
        masks = mask_logits[batch, keep].float().sigmoid()
        winners = (masks*confidence[:, None, None]).argmax(0)
        # Integer scatter sums are exact and independent of summation order.
        area = torch.zeros(len(masks), device=masks.device, dtype=torch.long)
        area.scatter_add_(0, winners.flatten(), torch.ones_like(winners).flatten())
        original_area = (masks >= mask_threshold).flatten(1).sum(1)
        accepted = ((area > 0) & (original_area > 0)
                    & (area.float()/original_area.clamp_min(1).float() >= overlap_threshold))
        winner_channel = channels[labels][winners]
        winner_support = masks.gather(0, winners[None])[0] >= mask_threshold
        valid_pixel = accepted[winners] & winner_support
        result[batch] = ((winner_channel[None] == output_channels) & valid_pixel[None]).float()
    return result


def person_mask_from_logits(mask_logits, class_logits, **kwargs):
    """Compatibility helper for callers that explicitly request person only."""
    return semantic_masks_from_logits(mask_logits, class_logits,
                                      target_class_ids=(PERSON_CLASS,), **kwargs)


def _validate_auxiliary(key, value):
    if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
        raise ValueError(f'Invalid auxiliary PMT checkpoint tensor: {key}')
    if key in ('network.encoder.pixel_mean', 'encoder.pixel_mean'):
        reference = torch.tensor([.485, .456, .406])
    elif key in ('network.encoder.pixel_std', 'encoder.pixel_std'):
        reference = torch.tensor([.229, .224, .225])
    elif key == 'criterion.empty_weight':
        if value.numel() != NUM_CLASSES+1:
            raise ValueError('PMT criterion class count does not match COCO')
        return
    else:
        raise ValueError(f'Unknown PMT checkpoint key: {key}')
    flattened = value.detach().float().cpu().flatten()
    if flattened.numel() != 3:
        raise ValueError(f'PMT image normalization has wrong shape: {key}')
    if flattened.max() > 2:
        flattened = flattened/255.
    if not torch.allclose(flattened, reference, atol=1e-5, rtol=1e-5):
        raise ValueError(f'PMT image normalization differs from ImageNet: {key}')


def normalize_pmt_state(state):
    """Accept only author's two documented key layouts and named auxiliaries."""
    if not isinstance(state, dict) or not state:
        raise ValueError('PMT checkpoint must be a nonempty state dictionary')
    output = {}
    layouts = set()
    for key, value in state.items():
        if key.startswith(('network.bn.', 'network.decoder.')):
            layouts.add('network')
            target = key.removeprefix('network.')
        elif key.startswith(('bn.', 'decoder.')):
            layouts.add('head')
            target = key
        else:
            _validate_auxiliary(key, value)
            continue
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f'Invalid PMT weight tensor: {key}')
        if target in output:
            raise ValueError(f'Duplicate PMT checkpoint key: {target}')
        output[target] = value
    if len(layouts) != 1:
        raise ValueError('PMT checkpoint must use one consistent head key layout')
    return output


class FrozenPMTPersonHead(PMTHead):
    """Requires a real pretrained asset; random fallback is never permitted."""
    def __init__(self, checkpoint, *, score_threshold=.8, mask_threshold=.5,
                 overlap_threshold=.8, expected_sha256=None, target_classes=None):
        self.target_classes = normalize_target_classes(target_classes)
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Pretrained PMT-S head required: {checkpoint}. '
                                    'Run scripts/prepare_person_assets.py first.')
        self.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if expected_sha256 and self.checkpoint_sha256 != expected_sha256:
            raise ValueError('PMT checkpoint SHA256 mismatch')
        super().__init__()
        if checkpoint.suffix == '.safetensors':
            from safetensors.torch import load_file
            state = load_file(str(checkpoint), device='cpu')
        elif checkpoint.suffix in ('.bin', '.pt', '.pth'):
            state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        else:
            raise ValueError('PMT asset must be safetensors or a weights-only PyTorch state')
        self.load_state_dict(normalize_pmt_state(state), strict=True)
        self.score_threshold = float(score_threshold)
        self.mask_threshold = float(mask_threshold)
        self.overlap_threshold = float(overlap_threshold)
        if not all(0 <= value <= 1 for value in (self.score_threshold, self.mask_threshold,
                                                self.overlap_threshold)):
            raise ValueError('PMT thresholds must be in [0,1]')
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, features, rope, geometry):
        masks, classes = super().forward(features, rope)
        masks = restore_mask_logits(masks, geometry)
        return semantic_masks_from_logits(masks, classes, score_threshold=self.score_threshold,
                                       mask_threshold=self.mask_threshold,
                                       overlap_threshold=self.overlap_threshold)
