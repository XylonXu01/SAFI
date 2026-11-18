from __future__ import annotations

import torch
import torch.nn.functional as F


def flow_warp(image: torch.Tensor, flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse warp image with target-to-source flow measured in pixels.

    The returned mask is one where the sampling coordinate lies inside the
    source image. ``align_corners=False`` is used consistently everywhere.
    """
    _, _, h, w = image.shape
    original_dtype = image.dtype
    # Pixel grids in BF16 lose single-pixel resolution beyond x/y=256.
    # Keep geometry and grid_sample in FP32 even under mixed-precision training.
    image, flow = image.float(), flow.float()
    if flow.shape[-2:] != (h, w):
        old_h, old_w = flow.shape[-2:]
        flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
        flow = flow * flow.new_tensor([w / old_w, h / old_h]).view(1, 2, 1, 1)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=image.device, dtype=image.dtype),
        torch.arange(w, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    x = xs.unsqueeze(0) + flow[:, 0]
    y = ys.unsqueeze(0) + flow[:, 1]
    valid = ((x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)).unsqueeze(1)
    grid_x = 2 * (x + 0.5) / w - 1
    grid_y = 2 * (y + 0.5) / h - 1
    grid = torch.stack((grid_x, grid_y), dim=-1)
    with torch.autocast(device_type=image.device.type, enabled=False):
        warped = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return warped.to(original_dtype), valid.to(original_dtype)


def image_gradients(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = image[..., :, 1:] - image[..., :, :-1]
    dy = image[..., 1:, :] - image[..., :-1, :]
    return dx, dy


def resize_flow(flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    h, w = flow.shape[-2:]
    new_h, new_w = size
    result = F.interpolate(flow, size=size, mode="bilinear", align_corners=False)
    return result * flow.new_tensor([new_w / w, new_h / h]).view(1, 2, 1, 1)
