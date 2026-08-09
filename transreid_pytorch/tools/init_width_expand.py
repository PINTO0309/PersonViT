"""Build a width-expanded initialization checkpoint from a trained narrower
OSNet (the inverse of tools/init_width_select.py, for custom multipliers such
as x1.25/x1.5 that have no ImageNet zoo weights).

Function-preserving expansion: every source tensor is copied into the leading
slice of the target tensor and all new slices are zero-initialized
(BatchNorm running_var of new channels is set to 1 for numerical safety).
New channels therefore output exactly zero and contribute nothing at
initialization — the expanded model computes the same function as the source
— and are gradually recruited by gradients through their BatchNorm scales
during training.

Accepts either a raw OSNet state dict (e.g. the ImageNet zoo weights) or a
full build_transformer checkpoint (base.* keys); the output keeps the same
format, so it loads via PRETRAIN_CHOICE 'imagenet' (raw) or 'self' (full).
The BNNeck and classifier operate on the multiplier-independent feature_dim
and are copied unchanged when present.

Usage (from transreid_pytorch/):
    python tools/init_width_expand.py \
        --src ../pretrained/osnet_x1_0_imagenet.pth \
        --dst ../pretrained/osnet_x1_5_init.pth \
        --arch osnet_x1_5
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.backbones import osnet as osnet_module
from model.backbones import osnet_ain as osnet_ain_module


def expand_into(target, source):
    """Copy `source` into the leading slice of a zero tensor shaped like `target`."""
    out = torch.zeros_like(target)
    region = tuple(slice(0, min(t, s)) for t, s in zip(target.shape, source.shape))
    out[region] = source[region]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--src', required=True, help='source checkpoint (glob allowed)')
    ap.add_argument('--dst', required=True, help='output checkpoint path')
    ap.add_argument('--arch', required=True,
                    help='target factory name, e.g. osnet_x1_25 / osnet_x1_5')
    args = ap.parse_args()

    matches = sorted(glob.glob(args.src))
    if not matches:
        raise FileNotFoundError('no checkpoint matches {}'.format(args.src))
    src_path = matches[-1]
    state = torch.load(src_path, map_location='cpu', weights_only=False)
    if 'state_dict' in state:
        state = state['state_dict']
    state = {k.replace('module.', ''): v for k, v in state.items()}

    prefixed = any(k.startswith('base.') for k in state)
    backbone_src = {k[len('base.'):]: v for k, v in state.items()
                    if k.startswith('base.')} if prefixed else state

    factory_module = (
        osnet_ain_module if args.arch.startswith('osnet_ain') else osnet_module
    )
    target = getattr(factory_module, args.arch)().state_dict()
    out, expanded, copied = {}, 0, 0
    for key, tgt in target.items():
        src = backbone_src.get(key)
        if src is None:
            continue  # e.g. ImageNet classifier keys are absent from the target
        if src.shape == tgt.shape:
            out[key] = src.clone()
            copied += 1
        else:
            grown = expand_into(tgt, src)
            if key.endswith('running_var'):
                n = src.shape[0]
                grown[n:] = 1.0
            out[key] = grown
            expanded += 1

    if prefixed:
        out = {'base.' + k: v for k, v in out.items()}
        for key, value in state.items():
            # feature_dim is multiplier-independent, so neck/classifier copy as-is
            if key.startswith('bottleneck.') or key.startswith('classifier.'):
                out[key] = value.clone()

    torch.save(out, args.dst)
    n_params = sum(v.numel() for v in out.values() if v.dim() > 0)
    print('source     : {} ({} format)'.format(
        src_path, 'build_transformer' if prefixed else 'raw OSNet'))
    print('target arch: {}'.format(args.arch))
    print('tensors    : {} copied, {} zero-expanded'.format(copied, expanded))
    print('parameters : {:.2f}M'.format(n_params / 1e6))
    print('written to : {}'.format(args.dst))


if __name__ == '__main__':
    main()
