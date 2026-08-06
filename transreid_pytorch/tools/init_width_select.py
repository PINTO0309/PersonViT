"""Build a width-selected initialization checkpoint from a trained wider ViT.

Shrinks a TransReID ViT along three axes by magnitude-based channel
selection ("initializing models with larger ones"):

- residual stream (embed dim): one GLOBAL top-k index set, ranked by the
  summed L2 norms of every consumer column (qkv, mlp.fc1) and producer row
  (attn.proj, mlp.fc2) across all blocks. The same indices slice the patch
  embedding, positional embedding, cls token, all LayerNorms, the BNNeck and
  the classifier columns, keeping the residual stream consistent end to end.
- attention heads: per block, keep the heads with the largest Frobenius norm
  of their attn.proj input columns (head dim stays 64).
- MLP hidden units: per block, keep units with the largest
  ||fc1[h, :]|| + ||fc2[:, h]||.

The output is a full build_transformer state dict loadable with
MODEL.PRETRAIN_CHOICE: 'self'.

Usage (from transreid_pytorch/):
    python tools/init_width_select.py \
        --src "logs/reid_vit_small_8gb_distill/transformer_best_*.pth" \
        --dst ../pretrained/vit_t256_init.pth \
        --dim 256 --heads 4
"""

import argparse
import glob

import torch


def topk_sorted(scores, k):
    return torch.topk(scores, k).indices.sort().values


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--src', required=True, help='source checkpoint (glob allowed)')
    ap.add_argument('--dst', required=True, help='output checkpoint path')
    ap.add_argument('--dim', type=int, default=256, help='target embed dim')
    ap.add_argument('--heads', type=int, default=4, help='target head count')
    args = ap.parse_args()

    matches = sorted(glob.glob(args.src))
    if not matches:
        raise FileNotFoundError('no checkpoint matches {}'.format(args.src))
    src_path = matches[-1]
    state = torch.load(src_path, map_location='cpu', weights_only=False)
    if 'state_dict' in state:
        state = state['state_dict']

    d_old = state['base.pos_embed'].shape[-1]
    d_new = args.dim
    n_blocks = 1 + max(int(k.split('.')[2]) for k in state if k.startswith('base.blocks.'))
    head_dim = 64
    h_old, h_new = d_old // head_dim, args.heads
    hid_old = state['base.blocks.0.mlp.fc1.weight'].shape[0]
    hid_new = 4 * d_new
    assert d_new % head_dim == 0 and h_new * head_dim == d_new

    # ---- global residual channel ranking -------------------------------
    imp = torch.zeros(d_old)
    for b in range(n_blocks):
        p = 'base.blocks.{}.'.format(b)
        imp += state[p + 'attn.qkv.weight'].norm(dim=0)
        imp += state[p + 'mlp.fc1.weight'].norm(dim=0)
        imp += state[p + 'attn.proj.weight'].norm(dim=1)
        imp += state[p + 'mlp.fc2.weight'].norm(dim=1)
    res = topk_sorted(imp, d_new)

    def slice_vec(key):
        out[key] = state[key][res].clone()

    out = {}
    # ---- embeddings and heads shared across blocks ---------------------
    out['base.patch_embed.proj.weight'] = state['base.patch_embed.proj.weight'][res].clone()
    out['base.patch_embed.proj.bias'] = state['base.patch_embed.proj.bias'][res].clone()
    out['base.pos_embed'] = state['base.pos_embed'][..., res].clone()
    out['base.cls_token'] = state['base.cls_token'][..., res].clone()
    for key in ('base.norm.weight', 'base.norm.bias'):
        slice_vec(key)
    for key in ('bottleneck.weight', 'bottleneck.bias',
                'bottleneck.running_mean', 'bottleneck.running_var'):
        if key in state:
            slice_vec(key)
    if 'bottleneck.num_batches_tracked' in state:
        out['bottleneck.num_batches_tracked'] = state['bottleneck.num_batches_tracked'].clone()
    if 'classifier.weight' in state:
        out['classifier.weight'] = state['classifier.weight'][:, res].clone()

    # ---- per-block slicing ---------------------------------------------
    for b in range(n_blocks):
        p = 'base.blocks.{}.'.format(b)
        for key in (p + 'norm1.weight', p + 'norm1.bias',
                    p + 'norm2.weight', p + 'norm2.bias'):
            slice_vec(key)

        proj_w = state[p + 'attn.proj.weight']              # [D, H*64]
        head_imp = torch.stack([
            proj_w[res][:, h * head_dim:(h + 1) * head_dim].norm()
            for h in range(h_old)])
        heads = topk_sorted(head_imp, h_new)
        head_rows = torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim)
                               for h in heads])

        qkv_w = state[p + 'attn.qkv.weight']                # [3D, D]
        qkv_b = state[p + 'attn.qkv.bias']
        rows = torch.cat([head_rows + s * d_old for s in range(3)])
        out[p + 'attn.qkv.weight'] = qkv_w[rows][:, res].clone()
        out[p + 'attn.qkv.bias'] = qkv_b[rows].clone()
        out[p + 'attn.proj.weight'] = proj_w[res][:, head_rows].clone()
        out[p + 'attn.proj.bias'] = state[p + 'attn.proj.bias'][res].clone()

        fc1_w = state[p + 'mlp.fc1.weight']                 # [4D, D]
        fc2_w = state[p + 'mlp.fc2.weight']                 # [D, 4D]
        hid_imp = fc1_w.norm(dim=1) + fc2_w.norm(dim=0)
        hid = topk_sorted(hid_imp, hid_new)
        out[p + 'mlp.fc1.weight'] = fc1_w[hid][:, res].clone()
        out[p + 'mlp.fc1.bias'] = state[p + 'mlp.fc1.bias'][hid].clone()
        out[p + 'mlp.fc2.weight'] = fc2_w[res][:, hid].clone()
        out[p + 'mlp.fc2.bias'] = state[p + 'mlp.fc2.bias'][res].clone()

    torch.save(out, args.dst)
    n_params = sum(v.numel() for v in out.values() if v.dim() > 0)
    print('source     : {}'.format(src_path))
    print('residual   : {} -> {} (global top-k)'.format(d_old, d_new))
    print('heads      : {} -> {} per block, MLP hidden {} -> {}'.format(
        h_old, h_new, hid_old, hid_new))
    print('parameters : {:.2f}M (incl. classifier)'.format(n_params / 1e6))
    print('written to : {}'.format(args.dst))


if __name__ == '__main__':
    main()
