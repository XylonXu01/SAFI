"""Same sparse matching arithmetic, omitting training-only diagnostic targets."""

import torch
import torch.nn.functional as F
from ..geometry import resize_flow
from ..models.transport import FlowSparseTransport, sparse_uot


class InferenceTransport(FlowSparseTransport):
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
        return matched.to(query.dtype), confidence, None
