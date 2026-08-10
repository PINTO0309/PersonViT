"""Per-domain benchmark of a trained model on the unified reid test split.

The unified dataset encodes an anonymous domain id in every file name
(p<pid>_d<dom>_c<cam>_<seq>), and the split holds out identities per domain,
so the merged evaluation can be decomposed without touching the data. Two
protocols are reported per domain:

- within:  domain queries vs the same domain's gallery only — the
  "single-source benchmark" analogue on this split. Not comparable to the
  official benchmarks of the source datasets (different identity splits).
- merged:  domain queries vs the full merged gallery — the contribution of
  each domain to the aggregate metric, including cross-domain distractors.

Usage (from transreid_pytorch/):
    python tools/eval_per_domain.py \
        --config configs/reid/vit_small_8gb_distill.yml \
        --weight "logs/reid_vit_small_8gb_distill/transformer_best_*.pth"
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from config import cfg
from datasets import make_dataloader
from datasets.reid import REID
from model import make_model
from utils.metrics import euclidean_distance, eval_func


def extract_features(model, val_loader, device='cuda'):
    feats = []
    model.eval()
    with torch.no_grad():
        for img, pid, camid, camids, target_view, _ in val_loader:
            img = img.to(device)
            camids_t = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids_t, view_label=target_view)
            # the ViT forward returns a view into the full token tensor; move
            # to CPU per batch so the backing GPU activations are freed
            feats.append(feat.detach().cpu())
    return torch.cat(feats, dim=0)


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids):
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)[:2]
    return mAP, cmc[0], cmc[4]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', required=True)
    ap.add_argument('--weight', required=True, help='trained checkpoint (glob allowed)')
    ap.add_argument('--markdown', action='store_true',
                    help='print the results table in Markdown (paste-ready for README/docs)')
    ap.add_argument('opts', nargs=argparse.REMAINDER,
                    help='extra config overrides in KEY VALUE form')
    args = ap.parse_args()

    weight = sorted(glob.glob(args.weight))
    if not weight:
        raise FileNotFoundError('no checkpoint matches {}'.format(args.weight))
    weight = weight[-1]

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(['DATALOADER.NUM_WORKERS', '4',
                         'MODEL.PRETRAIN_CHOICE', 'none'])
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    _, _, val_loader, num_query, num_classes, cam_num, view_num = make_dataloader(cfg)
    model = make_model(cfg, num_class=num_classes, camera_num=cam_num, view_num=view_num)
    model.load_param(weight)
    model.to('cuda')

    # sample metadata in val_loader order (dataset.query + dataset.gallery)
    dataset = REID(root=cfg.DATASETS.ROOT_DIR, verbose=False)
    samples = dataset.query + dataset.gallery
    pids = np.array([s[1] for s in samples])
    camids = np.array([s[2] for s in samples])
    doms = np.array([s[3] for s in samples])

    feats = extract_features(model, val_loader).cpu()
    del model
    torch.cuda.empty_cache()  # distances run on CPU; free the GPU immediately
    if cfg.TEST.FEAT_NORM == 'yes':
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    qf, gf = feats[:num_query], feats[num_query:]
    q_pids, g_pids = pids[:num_query], pids[num_query:]
    q_camids, g_camids = camids[:num_query], camids[num_query:]
    q_doms, g_doms = doms[:num_query], doms[num_query:]

    distmat = euclidean_distance(qf, gf)

    print('\nmodel  : {}'.format(weight))
    print('overall: mAP {:.4f}  Rank-1 {:.4f}  Rank-5 {:.4f}  '
          '({:,} query / {:,} gallery)'.format(
              *evaluate(distmat, q_pids, g_pids, q_camids, g_camids),
              num_query, len(g_pids)))
    if args.markdown:
        print('| domain | queries | gallery | within mAP | within R1 | within R5 | merged mAP | merged R1 | merged R5 |')
        print('| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |')
    else:
        print('domain | queries | gallery | within: mAP    R1     R5   | merged: mAP    R1     R5')
    for d in sorted(set(q_doms.tolist())):
        iq = q_doms == d
        ig = g_doms == d
        within = evaluate(
            euclidean_distance(qf[torch.as_tensor(iq)], gf[torch.as_tensor(ig)]),
            q_pids[iq], g_pids[ig], q_camids[iq], g_camids[ig])
        merged = evaluate(distmat[iq], q_pids[iq], g_pids, q_camids[iq], g_camids)
        if args.markdown:
            print('| d{:02d} | {:,} | {:,} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                d, int(iq.sum()), int(ig.sum()), *within, *merged))
        else:
            print('d{:02d}    | {:7,d} | {:7,d} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f}'.format(
                d, int(iq.sum()), int(ig.sum()), *within, *merged))


if __name__ == '__main__':
    main()
