"""[E2] Normalization factory for the image encoder and decoder.

Both `ImageEncoder` and `ImageDecoder` take a `norm_layer` constructor and call it
as `norm_layer([C, H, W])` -- a spatial `nn.LayerNorm` over the full feature map.
That normalizes across channel *and* spatial axes jointly, so it removes per-channel
scale as well as per-position scale, and it bakes the spatial dimensions into the
parameter shape. The three alternatives here all normalize per channel instead and
are shape-agnostic, which is the actual content of the experiment.

Every factory keeps the `[C, H, W]` call signature so neither module needs editing
beyond the constructor argument, and `layer` returns exactly `nn.LayerNorm([C,H,W])`,
so the default is bit-identical to the released model.

See PROJECT_PLAN.md 3.6, E2.
"""

import torch.nn as nn


def _group_count(channels, target):
    """Largest divisor of `channels` that is <= `target`, found by halving.

    The first encoder layer is `ngf` wide (16 by default), so a fixed 32 groups
    would fail there. Halving keeps a power-of-two group count for every layer in
    both stacks, since all channel counts here are `ngf * 2**k`.
    """
    groups = min(target, channels)
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return max(1, groups)


def make_img_norm(kind='layer', groups=32):
    """Return a callable taking `[C, H, W]` and returning a norm module."""

    if kind == 'layer':
        return lambda shape: nn.LayerNorm(shape)

    if kind == 'group':
        return lambda shape: nn.GroupNorm(_group_count(shape[0], groups), shape[0])

    if kind == 'batch':
        # affine=True matches LayerNorm's elementwise_affine default; the parameter
        # count drops from C*H*W to C, which is itself part of what E2 changes.
        return lambda shape: nn.BatchNorm2d(shape[0])

    if kind == 'instance':
        # InstanceNorm2d defaults to affine=False, which would drop the learned
        # scale and shift entirely and confound the comparison with a capacity
        # change. Force affine=True so the only difference is which axes are pooled.
        return lambda shape: nn.InstanceNorm2d(shape[0], affine=True)

    raise ValueError(f"unknown --img_norm {kind!r}")
