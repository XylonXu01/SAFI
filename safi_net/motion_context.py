"""Full-view motion inputs, independent of hidden query targets and image crops."""
import torch
import torch.nn.functional as F

from .types import MotionContext

CONTEXT_SIZE = 512


@torch.no_grad()
def letterbox_motion_frames(images, size=CONTEXT_SIZE):
    """Encode any number of full-view frames with exactly the context geometry."""
    height, width = images.shape[-2:]
    ratio = min(1.0, size / max(height, width))
    h, w = max(1, round(height * ratio)), max(1, round(width * ratio))
    value = F.interpolate(images.detach().float(), (h, w), mode='bilinear',
                          align_corners=False, antialias=True)
    return F.pad(value, (0, size - w, 0, size - h))


@torch.no_grad()
def build_motion_context(ir, vis, *, frame_step=1.0, crop=None, flip=False,
                         output_size=None, size=CONTEXT_SIZE):
    """Map output pixel centers into a fixed, right/bottom padded full view.

    ir/vis: [B,2,C,H,W]. crop is (top,left,height,width) in these full-view
    coordinates, before horizontal flipping. All frames share that geometry.
    """
    if (ir.ndim != 5 or vis.ndim != 5 or ir.shape[1:3] != (2, 1)
            or vis.shape[1:3] != (2, 3) or ir.shape[0] != vis.shape[0]
            or ir.shape[-2:] != vis.shape[-2:] or ir.device != vis.device):
        raise ValueError('Motion context needs paired [B,2,1/3,H,W] observations')
    batch, _, _, height, width = ir.shape
    geometry = build_motion_geometry(ir, (height, width), frame_step=frame_step,
                                     crop=crop, flip=flip, output_size=output_size, size=size)
    def encode(value):
        flat = letterbox_motion_frames(value.flatten(0, 1), size)
        return flat.reshape(batch, 2, value.shape[2], size, size)
    return MotionContext(encode(ir), encode(vis), **geometry)


@torch.no_grad()
def build_motion_geometry(reference, image_size, *, frame_step=1.0, crop=None,
                          flip=False, output_size=None, size=CONTEXT_SIZE):
    """Build shared geometry without encoding images that a caller already has.

    ``reference`` supplies batch size, device and tensor allocation; image_size
    describes the original full view, before letterboxing or cropping.
    """
    if not isinstance(size, int) or size < 1:
        raise ValueError('Motion context size must be positive')
    batch = reference.shape[0]
    height, width = image_size
    if min(height, width) < 1:
        raise ValueError('Full-view dimensions must be positive')
    ratio = min(1.0, size / max(height, width))
    h, w = max(1, round(height * ratio)), max(1, round(width * ratio))
    valid = reference.new_zeros(batch, 1, size, size, dtype=torch.float32)
    valid[..., :h, :w] = 1
    if crop is None:
        crop = [0, 0, height, width]
    crop = torch.as_tensor(crop, device=reference.device, dtype=torch.float32)
    if crop.shape == (4,):
        crop = crop[None].expand(batch, -1)
    if crop.shape != (batch, 4) or not torch.isfinite(crop).all():
        raise ValueError('crop must be finite [B,4] top,left,height,width')
    top, left, ch, cw = crop.unbind(1)
    if (top < 0).any() or (left < 0).any() or (ch <= 0).any() or (cw <= 0).any():
        raise ValueError('Invalid crop bounds')
    if (top + ch > height + 1e-4).any() or (left + cw > width + 1e-4).any():
        raise ValueError('Crop must be inside full view')
    if output_size is None:
        output_size = (int(ch[0]), int(cw[0]))
    oh, ow = output_size
    if min(oh, ow) < 1:
        raise ValueError('Output size must be positive')
    flipped = torch.as_tensor(flip, device=reference.device, dtype=torch.bool)
    if flipped.ndim == 0:
        flipped = flipped.expand(batch)
    if flipped.shape != (batch,):
        raise ValueError('flip must be scalar or [B]')
    sx, sy = w / width, h / height
    transform = reference.new_zeros(batch, 2, 3, dtype=torch.float32)
    # align_corners=False pixel-center mapping, including crop and optional flip.
    dx, dy = cw / ow, ch / oh
    transform[:, 0, 0] = torch.where(flipped, -dx * sx, dx * sx)
    transform[:, 0, 2] = torch.where(flipped, (left + cw - .5 * dx) * sx - .5,
                                    (left + .5 * dx) * sx - .5)
    transform[:, 1, 1] = dy * sy
    transform[:, 1, 2] = (top + .5 * dy) * sy - .5
    step = torch.as_tensor(frame_step, device=reference.device, dtype=torch.float64)
    if step.ndim == 0:
        step = step.expand(batch).clone()
    if step.shape != (batch,) or not torch.isfinite(step).all() or (step <= 0).any():
        raise ValueError('frame_step must be finite positive [B]')
    return {'spatial_valid': valid, 'transform': transform, 'frame_step': step}


def infer_frame_step(inputs):
    """The supported output timeline is twice the IR observation frame rate."""
    ir_pair, vis_pair = inputs.ir_valid.all(1), inputs.vis_valid.all(1)
    ir_span = inputs.ir_times[:, 1] - inputs.ir_times[:, 0]
    vis_span = inputs.vis_times[:, 1] - inputs.vis_times[:, 0]
    return torch.where(ir_pair & (ir_span > 0), ir_span / 2,
                       torch.where(vis_pair & (vis_span > 0), vis_span,
                                   torch.ones_like(ir_span)))


@torch.no_grad()
def ensure_motion_context(inputs):
    context = inputs.motion_context
    if context is None:
        return build_motion_context(inputs.ir, inputs.vis, frame_step=infer_frame_step(inputs))
    batch = inputs.ir.shape[0]
    if (context.ir.shape != (batch, 2, 1, CONTEXT_SIZE, CONTEXT_SIZE)
            or context.vis.shape != (batch, 2, 3, CONTEXT_SIZE, CONTEXT_SIZE)
            or context.spatial_valid.shape != (batch, 1, CONTEXT_SIZE, CONTEXT_SIZE)
            or context.transform.shape != (batch, 2, 3)
            or context.frame_step.shape != (batch,)):
        raise ValueError('Invalid full-FOV motion context shape')
    tensors = (context.ir, context.vis, context.spatial_valid, context.transform, context.frame_step)
    if any(t.device != inputs.ir.device for t in tensors):
        raise ValueError('Motion context and observations must share device')
    if any(not torch.isfinite(t).all() for t in tensors):
        raise ValueError('Motion context must be finite')
    if (context.frame_step <= 0).any() or ((context.spatial_valid < 0) | (context.spatial_valid > 1)).any():
        raise ValueError('Invalid motion context step or validity')
    return context


def project_context_map(value, context, output_size):
    """Sample a query-aligned context map at current image/crop pixel centers."""
    if value.ndim != 4 or value.shape[0] != context.transform.shape[0]:
        raise ValueError('Expected [B,C,Hc,Wc] context map')
    height, width = output_size
    ys, xs = torch.meshgrid(torch.arange(height, device=value.device, dtype=torch.float32),
                            torch.arange(width, device=value.device, dtype=torch.float32), indexing='ij')
    positions = torch.stack((xs, ys, torch.ones_like(xs)), 0)
    coordinates = torch.einsum('bij,jhw->bihw', context.transform.float(), positions)
    grid = torch.stack((2 * (coordinates[:, 0] + .5) / value.shape[-1] - 1,
                        2 * (coordinates[:, 1] + .5) / value.shape[-2] - 1), -1)
    with torch.autocast(device_type=value.device.type, enabled=False):
        result = F.grid_sample(value.float(), grid, mode='bilinear', padding_mode='zeros',
                               align_corners=False)
        valid = F.grid_sample(context.spatial_valid.float(), grid, mode='nearest',
                              padding_mode='zeros', align_corners=False)
    return result * valid
