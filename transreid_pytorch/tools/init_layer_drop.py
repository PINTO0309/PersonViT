"""Build a layer-dropped initialization checkpoint (tier T) from a trained
ViT-S student.

Copies the patch embedding, positional embedding, cls token, final LayerNorm,
BNNeck and classifier unchanged, and inherits a subset of the transformer
blocks, renaming them to consecutive indices:

    base.blocks.{0,2,4,6,8,11}  ->  base.blocks.{0..5}      (default)

The output is a full build_transformer state dict loadable with
MODEL.PRETRAIN_CHOICE: 'self'.

Usage (from transreid_pytorch/):
    python tools/init_layer_drop.py \
        --src "logs/reid_vit_small_8gb_distill/transformer_best_*.pth" \
        --dst ../pretrained/vit_t_init.pth \
        [--blocks 0,2,4,6,8,11]
"""

import argparse
import glob
import re

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--src', required=True,
                    help='source checkpoint (glob patterns allowed)')
    ap.add_argument('--dst', required=True, help='output checkpoint path')
    ap.add_argument('--blocks', default='0,2,4,6,8,11',
                    help='comma-separated source block indices to inherit')
    args = ap.parse_args()

    matches = sorted(glob.glob(args.src))
    if not matches:
        raise FileNotFoundError('no checkpoint matches {}'.format(args.src))
    src_path = matches[-1]
    selected = [int(i) for i in args.blocks.split(',')]

    state = torch.load(src_path, map_location='cpu', weights_only=False)
    if 'state_dict' in state:
        state = state['state_dict']

    block_re = re.compile(r'^(base\.blocks\.)(\d+)(\..+)$')
    out, kept, dropped = {}, 0, 0
    remap = {src: dst for dst, src in enumerate(selected)}
    for key, value in state.items():
        m = block_re.match(key)
        if m is None:
            out[key] = value          # patch/pos embed, cls, norm, neck, classifier
            continue
        src_idx = int(m.group(2))
        if src_idx in remap:
            out['{}{}{}'.format(m.group(1), remap[src_idx], m.group(3))] = value
            kept += 1
        else:
            dropped += 1

    torch.save(out, args.dst)
    n_params = sum(v.numel() for v in out.values() if hasattr(v, 'numel'))
    print('source          : {}'.format(src_path))
    print('inherited blocks: {} -> 0..{}'.format(selected, len(selected) - 1))
    print('tensors kept    : {} block tensors ({} dropped), {} total keys'.format(
        kept, dropped, len(out)))
    print('parameters      : {:.2f}M (incl. classifier)'.format(n_params / 1e6))
    print('written to      : {}'.format(args.dst))


if __name__ == '__main__':
    main()
