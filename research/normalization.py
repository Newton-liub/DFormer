"""Honor single-device BatchNorm for the author's hard-coded SyncBN layers."""
import torch.nn as nn


def replace_sync_batchnorm(module):
    """Replace children in place, preserving affine parameters and running state."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.SyncBatchNorm):
            replacement = nn.BatchNorm2d(child.num_features, eps=child.eps,
                                         momentum=child.momentum, affine=child.affine,
                                         track_running_stats=child.track_running_stats)
            if child.affine:
                replacement.weight = child.weight
                replacement.bias = child.bias
            replacement.running_mean = child.running_mean
            replacement.running_var = child.running_var
            replacement.num_batches_tracked = child.num_batches_tracked
            replacement.train(child.training)
            setattr(module, name, replacement)
        else:
            replace_sync_batchnorm(child)
    return module
