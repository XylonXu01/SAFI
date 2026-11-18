"""Flow-centred sparse matching with actual source/target node marginals."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from safi_net.geometry import resize_flow


def segment_logsumexp(values: torch.Tensor, indices: torch.Tensor, valid: torch.Tensor,
                      count: int) -> torch.Tensor:
    """Batched edge -> node reduction; never constructs an N x N matrix."""
    values, indices, valid = values.flatten(1), indices.flatten(1), valid.flatten(1)
    maxima = values.new_full((values.shape[0], count), -1e4)
    maxima.scatter_reduce_(1, indices, values.detach().masked_fill(~valid, -1e4),
                           reduce="amax", include_self=True)
    mass = torch.zeros_like(maxima).scatter_add(
        1, indices, (values - maxima.gather(1, indices)).clamp(-80, 0).exp() * valid
    )
    return maxima + mass.clamp_min(1e-30).log()


def sparse_uot(log_kernel: torch.Tensor, indices: torch.Tensor, valid: torch.Tensor,
               epsilon: float, rho: float, iterations: int) -> torch.Tensor:
    """Generalized Sinkhorn: rows=query nodes, columns=actual source node IDs.

    Uniform a=b=1/N, KL-unbalanced marginals. Returned edge masses are NOT
    row-normalized; conditional weights are constructed only for aggregation.
    """
    b, n, _ = log_kernel.shape
    exponent = rho / (rho + epsilon)
    log_mass = -math.log(n)
    log_v = log_kernel.new_zeros(b, n)
    kernel = log_kernel.masked_fill(~valid, -1e4)
    for _ in range(iterations):
        gathered_v = log_v.gather(1, indices.flatten(1)).view_as(kernel)
        log_u = exponent * (log_mass - torch.logsumexp(kernel + gathered_v, dim=-1))
        log_u = torch.where(valid.any(-1), log_u, 0)
        columns = segment_logsumexp(kernel + log_u[..., None], indices, valid, n)
        log_v = exponent * (log_mass - columns)
        log_v = torch.where(columns > -1e3, log_v, 0)
    gathered_v = log_v.gather(1, indices.flatten(1)).view_as(kernel)
    return (kernel + log_u[..., None] + gathered_v).clamp(-80, 20).exp() * valid


class FlowSparseTransport(nn.Module):
    def __init__(self, channels: int, radius: int = 2, mode: str = "softmax",
                 epsilon: float = 0.05, rho: float = 1.0, iterations: int = 10):
        super().__init__()
        if mode not in {"softmax", "uot"} or radius < 0 or min(epsilon, rho, iterations) <= 0:
            raise ValueError("Invalid sparse transport configuration")
        self.channels, self.radius, self.mode = channels, radius, mode
        self.epsilon, self.rho, self.iterations = epsilon, rho, iterations
        yy, xx = torch.meshgrid(torch.arange(-radius, radius + 1),
                                torch.arange(-radius, radius + 1), indexing="ij")
        self.register_buffer("offsets", torch.stack((xx.flatten(), yy.flatten()), -1))
        self.fallback_count = 0

    @property
    def candidate_count(self):
        return len(self.offsets)

    def forward(self, query, source, flow):
        b, c, h, w = query.shape
        if source.shape != query.shape or c != self.channels:
            raise ValueError("Transport expects equally shaped endpoint features")
        n, k = h * w, self.candidate_count
        flow = resize_flow(flow.float(), (h, w))
        yy, xx = torch.meshgrid(torch.arange(h, device=query.device),
                                torch.arange(w, device=query.device), indexing="ij")
        centre = torch.stack((xx, yy), -1).float()[None] + flow.permute(0, 2, 3, 1)
        coordinates = centre.detach().round().long().reshape(b, n, 1, 2) + self.offsets
        x, y = coordinates.unbind(-1)
        valid = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        indices = (y.clamp(0, h - 1) * w + x.clamp(0, w - 1)).long()
        patches = source.float().flatten(2).gather(
            2, indices.flatten(1)[:, None].expand(-1, c, -1)
        ).view(b, c, n, k)
        q = F.normalize(query.float().flatten(2), dim=1)[..., None]
        cost = 1 - (q * F.normalize(patches, dim=1)).sum(1)
        distance = (coordinates.float() - centre.reshape(b, n, 1, 2)).square().sum(-1)
        distance = distance / max(1, self.radius ** 2)
        logits = -(cost + 0.1 * distance) / self.epsilon
        soft = logits.masked_fill(~valid, -1e4).softmax(-1) * valid
        soft = soft / soft.sum(-1, keepdim=True).clamp_min(1e-12)
        plan = soft / n
        if self.mode == "uot":
            candidate = sparse_uot(logits, indices, valid, self.epsilon, self.rho, self.iterations)
            if torch.isfinite(candidate).all():
                plan = candidate
            else:
                self.fallback_count += 1
        mass = plan.sum(-1, keepdim=True)
        weights = plan / mass.clamp_min(1e-30)
        matched = (patches * weights[:, None]).sum(-1).view(b, c, h, w)
        confidence = (weights.max(-1).values * (mass[..., 0] * n).clamp(0, 1)).view(b, 1, h, w)
        target = (-distance.detach() / 0.5).masked_fill(~valid, -1e4).softmax(-1) * valid
        target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
        details = {"plan": plan, "weights": weights, "indices": indices, "valid": valid,
                   "target": target, "row_mass": mass}
        return matched.to(query.dtype), confidence, details
