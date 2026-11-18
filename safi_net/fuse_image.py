"""Fuse a spatially registered infrared and visible image pair."""
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .checkpoints import load_inference_model
from .types import TemporalInputs


def _read_image(path, mode):
    with Image.open(path) as image:
        return torch.from_numpy(np.array(image.convert(mode)))


def fuse_image(ir_input, vis_input, checkpoint, output, *, device='cuda:0', pretrained_root=None):
    output = Path(output)
    if output.suffix.lower() != '.png':
        raise ValueError('Output must be a PNG file')
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Output file already exists: {output}')

    ir = _read_image(ir_input, 'L')
    vis = _read_image(vis_input, 'RGB')
    height, width = ir.shape
    if vis.shape[:2] != (height, width):
        raise ValueError('IR/VIS spatial dimensions differ; inputs must already be registered')
    device = torch.device(device)
    if device.type not in ('cuda', 'cpu'):
        raise ValueError('Choose a CUDA or CPU device')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; use --device cpu')

    pad = (0, (-width) % 32, 0, (-height) % 32)
    ir = F.pad(ir[None, None].to(device, torch.float32) / 255, pad, mode='replicate')
    vis = F.pad(vis.permute(2, 0, 1)[None].to(device, torch.float32) / 255, pad, mode='replicate')
    # A static pair has one exact observation from each modality.
    times = torch.zeros((1, 2), device=device, dtype=torch.float64)
    valid = torch.tensor([[True, False]], device=device)
    inputs = TemporalInputs(
        ir=torch.stack((ir, torch.zeros_like(ir)), dim=1),
        vis=torch.stack((vis, torch.zeros_like(vis)), dim=1),
        ir_times=times, vis_times=times.clone(), ir_valid=valid, vis_valid=valid.clone(),
        query_time=times[:, 0].clone(),
    )
    model = load_inference_model(checkpoint, pretrained_root=pretrained_root, device=device)
    with torch.inference_mode():
        fused = model(inputs).fused[0, :, :height, :width].float()
        if not torch.isfinite(fused).all():
            raise FloatingPointError('SAFI-Net produced nonfinite output')
        pixels = (fused.clamp(0, 1).permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()

    encoded = BytesIO()
    Image.fromarray(pixels).save(encoded, format='PNG')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as handle:
        handle.write(encoded.getbuffer())
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('ir-input', 'vis-input', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--pretrained-root')
    args = parser.parse_args(argv)
    print(fuse_image(args.ir_input, args.vis_input, args.checkpoint, args.output,
                     device=args.device, pretrained_root=args.pretrained_root))


if __name__ == '__main__':
    main()
