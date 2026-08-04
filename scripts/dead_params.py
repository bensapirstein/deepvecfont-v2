"""[E6] Count the parameters that are constructed and never used.

`Transformer.__init__` builds a full Perceiver-IO stack -- a learned latent array, a
cross-attention and cross-feedforward block at every depth, a classifier head, a patch
embedding, a pre-LSTM projection -- and then `forward` and `att_residual` use only the
self-attention sublayers. `cross_attn` and `cross_ff` are unpacked from the ModuleList
in both loops and never called. The parameters are still allocated, still handed to the
optimizer in `parameters_all`, and still carry Adam moment buffers.

This script reports how many, so the reconstruction section can state it as a number.

    python scripts/dead_params.py

Note on why E6 is an audit rather than a training run: the dead modules are
constructed *before* several live ones, and every `nn.Linear` / `nn.Parameter`
construction draws from the global RNG stream. Deleting them therefore shifts the
initialization of everything built afterwards. A "deleted dead parameters" run is
numerically a different seed, and the seed-noise floor (0.0093 L1) is larger than any
effect the sweep is chasing, so such a run could not be interpreted either way.
Reported as a reconstruction finding. See PROJECT_PLAN.md 3.6, E6.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from models.model_main import ModelMain  # noqa: E402
from options import get_parser_main_model  # noqa: E402


# Attribute paths on model.transformer_main that nothing ever reads.
# `posr` is a PositionalEncoding, whose table is registered as a buffer rather than a
# parameter, so it contributes 0 here and is listed only for completeness.
DEAD_ATTRS = ['latents', 'to_logits', 'to_patch_embedding', 'pre_lstm_fc', 'posr']

# Index 0 and 1 of each entry in `layers` / `layers_cnnsvg`: the cross-attention block
# and the cross-feedforward. Unpacked in both forward loops, never invoked.
DEAD_LAYER_SLOTS = [('layers', 0), ('layers', 1),
                    ('layers_cnnsvg', 0), ('layers_cnnsvg', 1)]


def count(module_or_param):
    if isinstance(module_or_param, torch.nn.Parameter):
        return module_or_param.numel()
    return sum(p.numel() for p in module_or_param.parameters())


def main():
    opts = get_parser_main_model().parse_args([])
    model = ModelMain(opts, mode='train')
    tm = model.transformer_main

    total = sum(p.numel() for p in model.parameters())
    rows, dead = [], 0

    for name in DEAD_ATTRS:
        obj = getattr(tm, name, None)
        if obj is None:
            continue
        n = count(obj)
        dead += n
        rows.append((f'transformer_main.{name}', n))

    for list_name, slot in DEAD_LAYER_SLOTS:
        n = sum(count(blk[slot]) for blk in getattr(tm, list_name))
        dead += n
        label = 'cross_attn' if slot == 0 else 'cross_ff'
        rows.append((f'transformer_main.{list_name}[*][{slot}]  ({label})', n))

    width = max(len(r[0]) for r in rows)
    print(f"\n{'component':<{width}}  {'params':>12}")
    print('-' * (width + 14))
    for name, n in rows:
        print(f"{name:<{width}}  {n:>12,}")
    print('-' * (width + 14))
    print(f"{'dead total':<{width}}  {dead:>12,}")
    print(f"{'model total':<{width}}  {total:>12,}")
    print(f"\ndead fraction: {100.0 * dead / total:.2f}%")
    print(f"Adam moment state on dead params: {2 * dead * 4 / 1e6:.1f} MB (fp32, 2 moments)")


if __name__ == '__main__':
    main()
