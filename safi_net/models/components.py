"""Original v2 multiscale task network, without diffusion imports or modules."""
import math

import torch
from torch import nn
import torch.nn.functional as F


def resize(x, size):
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


def normalized_pair(a, b, wa, wb):
    mass = wa + wb
    return torch.where(mass > 1e-8, (a*wa + b*wb)/mass.clamp_min(1e-8), torch.zeros_like(a))


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        groups = math.gcd(8, cout)
        self.block = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1), nn.GroupNorm(groups, cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, 1, 1), nn.GroupNorm(groups, cout), nn.SiLU())
        self.skip = nn.Conv2d(cin, cout, 1, stride) if cin != cout or stride != 1 else nn.Identity()

    def forward(self, x):
        return self.block(x) + self.skip(x)


class ModalityEncoder(nn.Module):
    def __init__(self, channels, base):
        super().__init__()
        widths = [channels, 2*base, 4*base, 6*base, 8*base]
        self.stages = nn.ModuleList(ConvBlock(a, b, 2) for a, b in zip(widths, widths[1:]))

    def forward(self, x):
        values = []
        for stage in self.stages:
            x = stage(x)
            values.append(x)
        return values


class SemanticAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.projections = nn.ModuleList(nn.Conv2d(input_dim, output_dim, 1) for _ in range(4))
        self.layer_logits = nn.Parameter(torch.zeros(4))

    def forward(self, features):
        size = features[-1].shape[-2:]
        return sum(w * resize(p(f), size) for w, p, f in
                   zip(self.layer_logits.softmax(0), self.projections, features))


class PyramidDecoder(nn.Module):
    def __init__(self, channels, base, *, plain=False):
        super().__init__()
        widths = [2*base, 2*base, 3*base, 4*base]
        self.blocks = nn.ModuleList(ConvBlock(3*c + (widths[i+1] if i < 3 else 0), widths[i])
                                   for i, c in enumerate(channels))
        self.full = ConvBlock(widths[0]+4, 2*base)
        self.ir = nn.Conv2d(2*base, 1, 3, padding=1)
        self.vis = nn.Conv2d(2*base, 3, 3, padding=1)
        self.plain = plain
        if plain:
            self.rgb_head = nn.Conv2d(2*base, 3, 3, padding=1)
            nn.init.xavier_uniform_(self.rgb_head.weight)
            nn.init.zeros_(self.rgb_head.bias)
        else:
            self.fused = nn.Conv2d(2*base, 1, 3, padding=1)
        for head in (self.ir, self.vis) + (() if plain else (self.fused,)):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, pyramid, ir_private, vis_private, ir_base, vis_base):
        x = None
        for i in (3, 2, 1, 0):
            values = [pyramid[i], ir_private[i], vis_private[i]]
            if x is not None:
                values.append(resize(x, pyramid[i].shape[-2:]))
            x = self.blocks[i](torch.cat(values, 1))
        x = self.full(torch.cat((resize(x, ir_base.shape[-2:]), ir_base, vis_base), 1))
        ir = (ir_base + 0.2*self.ir(x).tanh()).clamp(0, 1)
        vis = (vis_base + 0.2*self.vis(x).tanh()).clamp(0, 1)
        # Apply this luminance residual after the caller enforces exact VIS identity.
        return (self.rgb_head(x).sigmoid() if self.plain else 0.5*self.fused(x).tanh()), ir, vis
