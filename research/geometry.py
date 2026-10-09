"""Local observation distributions for DFormerv2 geometry (no learned parameters).

Depth values are in the author's normalized grayscale domain, not metric depth.
A support mask excludes padding/explicit deletions; grayscale zero is an observation.
"""
import torch
import torch.nn.functional as F

DEPTH_MIN = -0.48 / 0.28
DEPTH_MAX = (1.0 - 0.48) / 0.28
NUM_BINS = 16
EPS = 1e-6


def soft_histogram(depth, centers):
    """Linearly assign each finite scalar to its adjacent fixed bin centers."""
    coordinate = ((depth - centers[0]) / (centers[-1] - centers[0]) * (centers.numel() - 1)).clamp(0, centers.numel() - 1)
    lower = coordinate.floor().long()
    upper = (lower + 1).clamp_max(centers.numel() - 1)
    fraction = coordinate - lower
    histogram = depth.new_zeros((depth.shape[0], centers.numel(), *depth.shape[-2:]))
    histogram.scatter_add_(1, lower, 1.0 - fraction)
    histogram.scatter_add_(1, upper, fraction)
    return histogram


def kernel_log_relation(probabilities, valid, kernel, query_chunk=128, group_chunk=16):
    """G,L,K -> G,heads,L,L without position-pair by bin-pair tensors.

    Group and query chunking bound intermediate memory. Only the final relation
    has quadratic token size, and high-resolution stages call this axially.
    """
    results = []
    for start in range(0, probabilities.shape[0], group_chunk):
        p = probabilities[start:start + group_chunk]
        active = valid[start:start + group_chunk]
        pk = p[:, None] @ kernel[None]
        self_energy = (pk * p[:, None]).sum(-1).clamp_min(EPS)
        inverse_norm = self_energy.rsqrt()
        rows = []
        for query in range(0, p.shape[1], query_chunk):
            stop = query + query_chunk
            similarity = pk[:, :, query:stop] @ p[:, None].transpose(-1, -2)
            similarity = similarity * inverse_norm[:, :, query:stop, None] * inverse_norm[:, :, None, :]
            log_similarity = similarity.clamp(EPS, 1.0).log()
            pair_valid = active[:, None, query:stop, None] & active[:, None, None, :]
            rows.append(torch.where(pair_valid, log_similarity, torch.zeros_like(log_similarity)))
        results.append(torch.cat(rows, dim=-2))
    return torch.cat(results, dim=0)


class ObservationGeometry:
    """One-forward workspace, constructed BEFORE any stage depth compression."""
    def __init__(self, depth, support=None, mode="odg", bins=NUM_BINS,
                 depth_min=DEPTH_MIN, depth_max=DEPTH_MAX):
        if mode not in ("mean", "odg"):
            raise ValueError("ObservationGeometry mode must be mean or odg")
        if depth.ndim != 4 or depth.shape[1] != 1:
            raise ValueError("Depth must have shape B,1,H,W")
        if bins != NUM_BINS or depth_min >= depth_max:
            raise ValueError("ODG v1 requires 16 ordered fixed bin centers")
        if support is not None and support.shape != depth.shape:
            raise ValueError("Support must match scalar depth B,1,H,W")
        self.mode = mode
        with torch.autocast(device_type=depth.device.type, enabled=False):
            depth = depth.float()
            finite = torch.isfinite(depth)
            if support is None:
                support = torch.ones_like(depth)
            else:
                support = support.float()
                if not torch.isfinite(support).all() or ((support < 0) | (support > 1)).any():
                    raise ValueError("Support must contain finite weights in [0,1]")
            self.support = support * finite
            self.depth = torch.where(finite, depth, torch.zeros_like(depth))
            self.centers = torch.linspace(depth_min, depth_max, bins, device=depth.device, dtype=torch.float32)
            self.occupancy = soft_histogram(self.depth, self.centers) * self.support if mode == "odg" else None

    def distribution(self, size):
        """Area average occupied bins, then normalize by observed support mass."""
        with torch.autocast(device_type=self.depth.device.type, enabled=False):
            mass = F.adaptive_avg_pool2d(self.support, size)
            valid = mass > 0
            denominator = mass.clamp_min(EPS)
            if self.mode == "odg":
                p = F.adaptive_avg_pool2d(self.occupancy, size) / denominator
            else:
                mean = F.adaptive_avg_pool2d(self.depth * self.support, size) / denominator
                p = soft_histogram(mean, self.centers)
            # Renormalize even tiny fractional masses, and explicitly zero empty cells.
            p = p / p.sum(1, keepdim=True).clamp_min(EPS)
            return torch.where(valid, p, torch.zeros_like(p)), valid

    def relation(self, size, decay, split_or_not):
        with torch.autocast(device_type=self.depth.device.type, enabled=False):
            p, valid = self.distribution(size)
            kernel = torch.exp(decay.float()[:, None, None] * (self.centers[:, None] - self.centers[None, :]).abs())
            batch, bins, height, width = p.shape
            if split_or_not:
                p_h = p.permute(0, 3, 2, 1).reshape(batch * width, height, bins)
                v_h = valid[:, 0].permute(0, 2, 1).reshape(batch * width, height)
                p_w = p.permute(0, 2, 3, 1).reshape(batch * height, width, bins)
                v_w = valid[:, 0].reshape(batch * height, width)
                h = kernel_log_relation(p_h, v_h, kernel).reshape(batch, width, decay.numel(), height, height).transpose(1, 2)
                w = kernel_log_relation(p_w, v_w, kernel).reshape(batch, height, decay.numel(), width, width).transpose(1, 2)
                return h, w
            p_full = p.flatten(2).transpose(1, 2)
            return kernel_log_relation(p_full, valid.flatten(1), kernel)
