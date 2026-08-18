"""Scene-population probe: does soma-style feature whitening help or hurt?

Field observation (PINTO0309/soma, real videos): population-statistics
whitening of ReID embeddings collapses identity separation with <=4 people
in the scene and stabilizes it with more. The cause is structural — the
whitening statistics are estimated from the scene population, and with few
people they are dominated by identity variation itself.

This probe reproduces that deployment scenario offline and answers, per
model: FROM WHICH scene population K does external whitening stop hurting
(and start helping)? Scenes of K identities are sampled from the unified
gallery (one domain per scene, like a single deployment site), embeddings
are whitened with statistics fit ON THAT SCENE ONLY, and identity
separation is measured before/after:

  - AUC    : P(within-id cosine > between-id cosine) (threshold-free)
  - margin : (mean within - mean between) / std between  (the soma view)

Typical use — run once per model and compare tables:
    python tools/probe_whitening_need.py \
        --config configs/reid/osnet_p_8gb_distill_ain_aug2_jpeg.yml \
        --weight logs/reid_osnet_p_8gb_distill_ain_synth_shint/folded_best.pth
A model with internalized camera/nuisance suppression (e.g. the cam-branch
arm) should show whitening as unnecessary (delta <= 0 everywhere); a plain
model should reproduce the soma crossover.
"""

import argparse
import collections
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from datasets.bases import ImageDataset
from datasets.make_dataloader import val_collate_fn
from datasets.reid import REID
from model import make_model


def auc_and_margin(feats, ids):
    """Separation of within-id vs between-id cosine similarities."""
    f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
    sims = f @ f.T
    same = ids[:, None] == ids[None, :]
    iu = np.triu_indices(len(ids), k=1)
    pos = sims[iu][same[iu]]
    neg = sims[iu][~same[iu]]
    if len(pos) == 0 or len(neg) == 0:
        return None, None
    # Mann-Whitney AUC via ranks
    scores = np.concatenate([neg, pos])
    ranks = scores.argsort().argsort().astype(np.float64) + 1
    auc = (ranks[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (
        len(pos) * len(neg))
    margin = (pos.mean() - neg.mean()) / (neg.std() + 1e-12)
    return auc, margin


def whiten(feats, mode, shrinkage=0.1):
    mu = feats.mean(axis=0, keepdims=True)
    x = feats - mu
    if mode == 'zscore':
        return x / (x.std(axis=0, keepdims=True) + 1e-6)
    # zca with shrinkage (population may be far smaller than dim)
    cov = x.T @ x / max(len(x) - 1, 1)
    cov = (1 - shrinkage) * cov + shrinkage * np.trace(cov) / cov.shape[0] * \
        np.eye(cov.shape[0], dtype=cov.dtype)
    vals, vecs = np.linalg.eigh(cov)
    inv_sqrt = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-8))) @ vecs.T
    return x @ inv_sqrt


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', required=True)
    ap.add_argument('--weight', required=True, help='checkpoint (glob allowed)')
    ap.add_argument('--mode', choices=('zscore', 'zca'), default='zscore',
                    help='whitening flavor fit per scene (default zscore)')
    ap.add_argument('--counts', default='2,3,4,6,8,12,16',
                    help='scene population sizes K to test')
    ap.add_argument('--trials', type=int, default=200, help='scenes per K')
    ap.add_argument('--imgs-per-id', type=int, default=8)
    ap.add_argument('--domains', default='0,1,2,3,4',
                    help='domains scenes are drawn from (one domain per scene)')
    ap.add_argument('--scene', choices=('camera', 'domain'), default='camera',
                    help="'camera' (default): all images of a scene come from "
                         "ONE camera — the soma deployment condition, where "
                         "per-id variance is small and small-K statistics "
                         "collapse onto identity directions; 'domain': images "
                         "may span the domain's cameras (multi-camera hub)")
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('opts', nargs=argparse.REMAINDER)
    args = ap.parse_args()

    matches = sorted(glob.glob(args.weight))
    if not matches:
        raise FileNotFoundError('no checkpoint matches {}'.format(args.weight))
    weight = matches[-1]

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(['MODEL.PRETRAIN_CHOICE', 'none'])
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    domains = {int(d) for d in args.domains.split(',')}
    counts = [int(k) for k in args.counts.split(',')]

    dataset = REID(root=cfg.DATASETS.ROOT_DIR, verbose=False)
    pat = re.compile(r'_d(\d+)_')

    # pool: gallery images of the chosen domains, grouped by
    # (scene-unit, pid) where the scene unit is a camera or a whole domain
    pool = collections.defaultdict(list)
    samples = []
    for s in dataset.gallery:
        dom = int(pat.search(os.path.basename(s[0])).group(1))
        if dom in domains:
            unit = s[2] if args.scene == 'camera' else dom
            pool[(unit, s[1])].append(len(samples))
            samples.append(s)
    pool = {k: v for k, v in pool.items() if len(v) >= 3}
    by_unit = collections.defaultdict(list)
    for (unit, pid) in pool:
        by_unit[unit].append((unit, pid))
    by_unit = {u: ids for u, ids in by_unit.items() if len(ids) >= 2}
    print('pool: {} (unit, id) groups over {} {} units ({} images)'.format(
        len(pool), len(by_unit), args.scene, len(samples)))

    model = make_model(cfg, num_class=751, camera_num=0, view_num=0)
    model.load_param(weight)
    model.to('cuda').eval()
    transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])
    loader = DataLoader(ImageDataset(samples, transforms), batch_size=128,
                        shuffle=False, num_workers=8, collate_fn=val_collate_fn)
    feats = []
    with torch.no_grad():
        for img, pid, camid, camids, target_view, _ in tqdm(
                loader, desc='embeddings', dynamic_ncols=True):
            feats.append(model(img.to('cuda'), cam_label=camids.to('cuda'),
                               view_label=target_view.to('cuda')).cpu())
    feats = torch.cat(feats).numpy().astype(np.float64)

    rng = np.random.default_rng(args.seed)
    unit_keys = sorted(by_unit)
    print('\nmodel  : {}'.format(weight))
    print('mode   : {} whitening, scene unit = one {} / {} trials per K'.format(
        args.mode, args.scene, args.trials))
    print('| K (people) | AUC raw | AUC whitened | dAUC | margin raw | margin whitened |')
    print('| ---: | ---: | ---: | ---: | ---: | ---: |')
    verdict = {}
    for K in counts:
        rows = []
        for _ in range(args.trials):
            unit = unit_keys[rng.integers(len(unit_keys))]
            cands = by_unit[unit]
            if len(cands) < K:
                continue
            keys = [cands[i] for i in rng.choice(len(cands), size=K, replace=False)]
            idx, ids = [], []
            for j, key in enumerate(keys):
                pick = pool[key]
                take = min(args.imgs_per_id, len(pick))
                idx.extend(rng.choice(pick, size=take, replace=False))
                ids.extend([j] * take)
            X, ids = feats[idx], np.array(ids)
            raw = auc_and_margin(X, ids)
            wht = auc_and_margin(whiten(X, args.mode), ids)
            if raw[0] is not None and wht[0] is not None:
                rows.append((raw[0], wht[0], raw[1], wht[1]))
        if not rows:
            continue
        r = np.array(rows)
        d_auc = r[:, 1].mean() - r[:, 0].mean()
        verdict[K] = d_auc
        print('| {} | {:.4f} | {:.4f} | {:+.4f} | {:.2f} | {:.2f} |'.format(
            K, r[:, 0].mean(), r[:, 1].mean(), d_auc, r[:, 2].mean(), r[:, 3].mean()))

    helped = [k for k, d in verdict.items() if d > 0.002]
    hurt = [k for k, d in verdict.items() if d < -0.002]
    if not helped:
        print('\nverdict: external whitening is UNNECESSARY at every tested K'
              ' (dAUC <= 0)')
    else:
        print('\nverdict: whitening hurts at K={} and helps from K={} up'
              ' (crossover between)'.format(hurt, min(helped)))


if __name__ == '__main__':
    main()
