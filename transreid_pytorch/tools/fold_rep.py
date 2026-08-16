"""Fold a rep-trained OSNet checkpoint back to the plain architecture.

The `osnet_ain_x1_0_rep` variant trains every LightConv depthwise 3x3 with
two extra linear branches (a depthwise 1x1 `rep_conv` and a per-channel
identity scale `rep_id`), summed before the shared BatchNorm. Because all
three branches are linear in the same input, they collapse exactly into
the 3x3 kernel's center tap:

    W'[c, 0, 1, 1] = W[c, 0, 1, 1] + rep_conv[c, 0, 0, 0] + rep_id[c]

The folded state dict has only plain `osnet_ain_x1_0` keys, so it can be
evaluated, exported, or used as a warm start anywhere the plain P weights
are accepted. Folding is exact (no approximation); parity is asserted at
transform time when --verify is given.

Usage (from transreid_pytorch/):
    python tools/fold_rep.py \
        --input logs/reid_osnet_p_8gb_distill_ain_aug3_jpeg_rep/transformer_best_*.pth \
        --output logs/reid_osnet_p_8gb_distill_ain_aug3_jpeg_rep/folded_best.pth \
        --verify
"""

import argparse
import glob

import torch

REP_CONV = '.rep_conv.weight'
REP_ID = '.rep_id'


def fold_rep_state(state):
    """Return a plain state dict with every rep branch folded into conv2.

    Loss-only auxiliaries (the embedding-KD and intermediate-hint
    projectors) are dropped too: the folded file is a deployment artifact.
    """
    folded = {k: v.clone() for k, v in state.items()
              if REP_CONV not in k and not k.endswith(REP_ID)
              and 'embed_proj.' not in k and 'hint_proj.' not in k}
    for key, value in state.items():
        if key.endswith(REP_CONV):
            conv2 = folded[key[:-len(REP_CONV)] + '.conv2.weight']
            conv2[:, 0, 1, 1] += value[:, 0, 0, 0]
        elif key.endswith(REP_ID):
            conv2 = folded[key[:-len(REP_ID)] + '.conv2.weight']
            conv2[:, 0, 1, 1] += value
    return folded


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--input', required=True, help='rep checkpoint (glob allowed)')
    ap.add_argument('--output', required=True, help='folded plain checkpoint path')
    ap.add_argument('--verify', action='store_true',
                    help='assert forward parity rep-vs-folded on random input')
    args = ap.parse_args()

    paths = sorted(glob.glob(args.input))
    if not paths:
        raise FileNotFoundError('no checkpoint matches {}'.format(args.input))
    state = torch.load(paths[-1], map_location='cpu', weights_only=False)
    for wrapper in ('model', 'state_dict'):
        if wrapper in state:
            state = state[wrapper]

    rep_keys = [k for k in state if REP_CONV in k or k.endswith(REP_ID)]
    if not rep_keys:
        raise ValueError('no rep branch keys found — already folded?')
    folded = fold_rep_state(state)
    print('folded {} rep branches ({} -> {} keys)'.format(
        len(rep_keys) // 2, len(state), len(folded)))

    if args.verify:
        import os
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from model.backbones.osnet_ain import (osnet_ain_x1_0, osnet_ain_x1_0_gem,
                                               osnet_ain_x1_0_rep,
                                               osnet_ain_x1_0_rep_gem,
                                               osnet_ain_x1_25, osnet_ain_x1_25_rep,
                                               osnet_ain_x1_5, osnet_ain_x1_5_rep)

        def backbone_state(full, model):
            wanted = model.state_dict()
            return {k.replace('base.', '', 1): v for k, v in full.items()
                    if k.replace('base.', '', 1) in wanted}

        # width is identified by the stem's output channels
        stem = next(v for k, v in state.items() if k.endswith('conv1.conv.weight')
                    and v.dim() == 4 and v.shape[1] == 3)
        width = stem.shape[0]
        gem = any(k.endswith('global_avgpool.p') for k in state)
        factories = {
            64: (osnet_ain_x1_0_rep_gem if gem else osnet_ain_x1_0_rep,
                 osnet_ain_x1_0_gem if gem else osnet_ain_x1_0),
            80: (osnet_ain_x1_25_rep, osnet_ain_x1_25),
            96: (osnet_ain_x1_5_rep, osnet_ain_x1_5),
        }
        if width not in factories or (gem and width != 64):
            raise ValueError('no verify factory for stem width {} (gem={})'.format(width, gem))
        rep_factory, plain_factory = factories[width]
        rep_model = rep_factory().eval()
        rep_model.load_state_dict(backbone_state(state, rep_model), strict=False)
        plain = plain_factory().eval()
        plain.load_state_dict(backbone_state(folded, plain), strict=False)
        x = torch.randn(4, 3, 256, 128)
        with torch.no_grad():
            diff = (rep_model(x) - plain(x)).abs().max().item()
        print('verify: rep-vs-folded max abs diff {:.3e}'.format(diff))
        assert diff < 1e-4, 'fold parity failed'

    torch.save(folded, args.output)
    print('saved {}'.format(args.output))


if __name__ == '__main__':
    main()
