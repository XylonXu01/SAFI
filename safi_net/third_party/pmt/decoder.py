# Copyright (c) 2025 Mobile Perception Systems Lab at TU/e.
# Adapted from https://github.com/tue-mps/pmt, image/models/{pmt,pmd,layers,utils}.py
# MIT license; portions of the upstream decoder derive from timm (Apache-2.0).
"""PMT's unmasked inference graph, retaining the author's checkpoint key names.

The encoder is supplied externally. Its normalized full token sequences include
CLS and register tokens; RoPE applies only to the final spatial patch tokens.
"""
import torch
from torch import nn
from torch.nn import functional as F


def apply_rotary_pos_emb(q, k, cos, sin):
    patches = sin.shape[-2]
    prefix = q.shape[-2] - patches
    if prefix < 0:
        raise ValueError('RoPE contains more patches than the attention sequence')
    qp, qs = q.split((prefix, patches), dim=-2)
    kp, ks = k.split((prefix, patches), dim=-2)
    cos, sin = cos.to(qs.dtype), sin.to(qs.dtype)
    def rotate_half(x):
        a, b = x.chunk(2, dim=-1)
        return torch.cat((-b, a), dim=-1)
    return (torch.cat((qp, qs*cos + rotate_half(qs)*sin), dim=-2),
            torch.cat((kp, ks*cos + rotate_half(ks)*sin), dim=-2))


class LayerNorm2d(nn.LayerNorm):
    """Channel-wise NCHW LayerNorm, equivalent to timm's eps=1e-6 default."""
    def __init__(self, channels):
        super().__init__(channels, eps=1e-6)

    def forward(self, x):
        return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape,
                            self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class ScaleBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(embed_dim, embed_dim, 2, stride=2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(embed_dim, embed_dim, 3, padding=1,
                               groups=embed_dim, bias=False)
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        return self.norm(self.conv2(self.act(self.conv1(x))))


class DINOv3ViTAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states, position_embeddings):
        batch, length, dim = hidden_states.shape
        def project(layer):
            return layer(hidden_states).view(batch, length, self.num_heads,
                                              self.head_dim).transpose(1, 2)
        q, k, v = (project(layer) for layer in (self.q_proj, self.k_proj, self.v_proj))
        q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
        attended = F.scaled_dot_product_attention(q, k, v)
        return self.o_proj(attended.transpose(1, 2).reshape(batch, length, dim))


class DINOv3ViTLayerScale(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        return x * self.lambda1


class DINOv3ViTMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        self.act_fn = nn.GELU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.up_proj(x)))


class DINOv3ViTLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attention = DINOv3ViTAttention(hidden_size, num_heads)
        self.layer_scale1 = DINOv3ViTLayerScale(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = DINOv3ViTMLP(hidden_size, intermediate_size)
        self.layer_scale2 = DINOv3ViTLayerScale(hidden_size)

    def forward(self, x, rope):
        x = x + self.layer_scale1(self.attention(self.norm1(x), rope))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))


class PlainMaskDecoder(nn.Module):
    def __init__(self, *, embed_dim=384, hidden_dim=384, num_prefix_tokens=5,
                 grid_size=(40, 40), num_classes=133, num_q=200, num_blocks=6,
                 num_levels=4, num_heads=6):
        super().__init__()
        self.num_prefix_tokens, self.grid_size = num_prefix_tokens, tuple(grid_size)
        self.num_q = num_q
        self.register_buffer('attn_mask_probs', torch.ones(num_blocks))
        self.lateral_projections = nn.ModuleList(nn.Sequential(
            nn.Linear(embed_dim, embed_dim//2), nn.GELU(),
            nn.Linear(embed_dim//2, embed_dim)) for _ in range(num_levels))
        self.scalers = nn.ParameterList(nn.Parameter(torch.ones(embed_dim))
                                        for _ in range(num_levels))
        self.q = nn.Embedding(num_q, embed_dim)
        self.blocks = nn.ModuleList(DINOv3ViTLayer(embed_dim, hidden_dim, num_heads)
                                    for _ in range(num_blocks))
        self.decoder_norm = nn.LayerNorm(embed_dim)
        # PMT-S uses patch_size=16: exactly two x2 blocks (output stride 4).
        self.upscale = nn.Sequential(ScaleBlock(embed_dim), ScaleBlock(embed_dim))
        self.mask_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(),
                                       nn.Linear(embed_dim, embed_dim), nn.GELU(),
                                       nn.Linear(embed_dim, embed_dim))
        self.class_head = nn.Linear(embed_dim, num_classes+1)

    def forward(self, features, rope):
        if len(features) != len(self.lateral_projections):
            raise ValueError('PMT needs its four configured encoder layers')
        expected_tokens = self.num_prefix_tokens + self.grid_size[0]*self.grid_size[1]
        if any(f.ndim != 3 or f.shape[1] != expected_tokens for f in features):
            raise ValueError('PMT requires full tokens including CLS and four registers')
        x = torch.stack([(feature + project(feature))*scale for feature, project, scale
                         in zip(features, self.lateral_projections, self.scalers)]).sum(0)
        x = torch.cat((self.q.weight[None].expand(x.shape[0], -1, -1), x), dim=1)
        for block in self.blocks:
            x = block(x, rope)
        x = self.decoder_norm(x)
        queries = x[:, :self.num_q]
        patches = x[:, self.num_q+self.num_prefix_tokens:].transpose(1, 2)
        patches = patches.reshape(x.shape[0], -1, *self.grid_size)
        masks = torch.einsum('bqc,bchw->bqhw', self.mask_head(queries), self.upscale(patches))
        return masks, self.class_head(queries)


class PMTHead(nn.Module):
    """Author-compatible head-only state with bn.* and decoder.* keys."""
    def __init__(self, **decoder_kwargs):
        super().__init__()
        dim = decoder_kwargs.get('embed_dim', 384)
        levels = decoder_kwargs.get('num_levels', 4)
        self.bn = nn.ModuleList(nn.SyncBatchNorm(dim) for _ in range(levels))
        self.decoder = PlainMaskDecoder(**decoder_kwargs)

    def forward(self, features, rope):
        normalized = [norm(value.transpose(1, 2)).transpose(1, 2)
                      for norm, value in zip(self.bn, features)]
        return self.decoder(normalized, rope)
