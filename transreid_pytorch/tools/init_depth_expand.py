"""Build a depth-expanded initialization checkpoint from a trained shallower
OSNet(-AIN) — the depth counterpart of tools/init_width_expand.py, for the
`*_deep` factories that append plain OSBlocks at the end of each stage.

Function-preserving expansion: appended blocks are residual
(`relu(F(x) + x)`), so zeroing the affine of each appended block's closing
BatchNorm makes F(x) = 0 and the block an exact identity at initialization
(existing activations are post-ReLU non-negative, so the outer ReLU is also
an identity). Because blocks are appended at the end of a stage, every
existing parameter name is unchanged and copies over verbatim; the new
blocks keep their random branch weights and are recruited gradually through
their zeroed BN scales during training.

Accepts either a raw OSNet state dict or a full build_transformer checkpoint
(base.* keys); the output keeps the same format, so it loads via
PRETRAIN_CHOICE 'imagenet' (raw) or 'self' (full). The BNNeck and classifier
are copied unchanged when present.

Usage (from transreid_pytorch/):
    python tools/init_depth_expand.py \
        --src "logs/reid_osnet_p_8gb_distill_ain/transformer_best_*.pth" \
        --dst ../pretrained/osnet_ain_x1_0_deep_init_from_p.pth \
        --arch osnet_ain_x1_0_deep
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.backbones import osnet as osnet_module
from model.backbones import osnet_ain as osnet_ain_module


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--src', required=True, help='source checkpoint (glob allowed)')
    ap.add_argument('--dst', required=True, help='output checkpoint path')
    ap.add_argument('--arch', required=True,
                    help='target factory name, e.g. osnet_ain_x1_0_deep')
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
    out, copied, fresh, zeroed = {}, 0, 0, 0
    for key, tgt in target.items():
        src = backbone_src.get(key)
        if src is not None and src.shape == tgt.shape:
            out[key] = src.clone()
            copied += 1
        else:
            # a tensor of an appended block: keep the fresh init, but zero the
            # closing BN affine so the whole block starts as an exact identity
            grown = tgt.clone()
            if key.endswith('conv3.bn.weight') or key.endswith('conv3.bn.bias'):
                grown.zero_()
                zeroed += 1
            out[key] = grown
            fresh += 1

    if prefixed:
        out = {'base.' + k: v for k, v in out.items()}
        for key, value in state.items():
            # feature_dim is depth-independent, so neck/classifier copy as-is
            if key.startswith('bottleneck.') or key.startswith('classifier.'):
                out[key] = value.clone()

    torch.save(out, args.dst)
    n_params = sum(v.numel() for v in out.values() if v.dim() > 0)
    print('source     : {} ({} format)'.format(
        src_path, 'build_transformer' if prefixed else 'raw OSNet'))
    print('target arch: {}'.format(args.arch))
    print('tensors    : {} copied, {} fresh ({} identity-zeroed BN affine)'.format(
        copied, fresh, zeroed))
    print('parameters : {:.2f}M'.format(n_params / 1e6))
    print('written to : {}'.format(args.dst))


if __name__ == '__main__':
    main()
