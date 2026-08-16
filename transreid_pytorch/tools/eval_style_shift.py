"""Style-shift robustness probe on the unified test split.

Measures how much a model's retrieval quality degrades when the capture
style changes, without leaving the 5-domain protocol: the queries are
re-rendered under deterministic photometric shifts (illumination gain,
contrast, color temperature, gamma) while the gallery stays clean —
simulating new cameras/lighting joining a deployment. The drop versus the
clean condition is the in-protocol generalization signal used to compare
the BN ladder against the -ain ladder.

Gallery features are extracted once and reused across conditions. With
--mode all, the gallery is shifted too (a fully re-deployed network);
that setting is less discriminative because consistently shifted features
partially preserve relative distances.

Usage (from transreid_pytorch/):
    python tools/eval_style_shift.py \
        --config configs/reid/vit_base_8gb.yml \
        --weight "logs/reid_vit_base_8gb/transformer_best_*.pth"
"""

import argparse
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
from datasets.style_shift import CONDITIONS, StyleShift
from model import make_model
from tools.eval_cache import EvalCache, checkpoint_signature
from tools.eval_official import OFFICIAL_DATASETS
from utils.metrics import euclidean_distance, eval_func


def build_transforms(condition_fn):
    return T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        StyleShift(condition_fn),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])


def _make_loader(samples, transforms):
    return DataLoader(
        ImageDataset(samples, transforms),
        batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=4, collate_fn=val_collate_fn,
    )


def extract(model, samples, transforms, device='cuda', desc='features'):
    feats = []
    model.eval()
    with torch.no_grad():
        for img, pid, camid, camids, target_view, _ in tqdm(
                _make_loader(samples, transforms), desc=desc,
                dynamic_ncols=True, leave=False):
            feat = model(img.to(device), cam_label=camids.to(device),
                         view_label=target_view.to(device))
            feats.append(feat.detach().cpu())
    feats = torch.cat(feats, dim=0)
    if cfg.TEST.FEAT_NORM == 'yes':
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    return feats


def extract_onnx(session, samples, transforms, desc='features'):
    """ONNX Runtime feature extraction (I/O tensor names auto-detected, so
    third-party graphs such as the upstream torchreid OSNet-AIN work)."""
    input_meta = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    fixed_batch = isinstance(input_meta.shape[0], int)
    feats = []
    for img, *_ in tqdm(_make_loader(samples, transforms), desc=desc,
                        dynamic_ncols=True, leave=False):
        batch = img.numpy()
        if fixed_batch and input_meta.shape[0] == 1:
            outputs = [session.run([output_name], {input_meta.name: row[None]})[0]
                       for row in batch]
            feats.append(np.concatenate(outputs, axis=0))
        else:
            feats.append(session.run([output_name], {input_meta.name: batch})[0])
    feats = torch.from_numpy(np.concatenate(feats, axis=0))
    if cfg.TEST.FEAT_NORM == 'yes':
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    return feats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', required=True)
    ap.add_argument('--weight', help='trained checkpoint (glob allowed)')
    ap.add_argument('--onnx', help='evaluate an ONNX model instead of a checkpoint '
                                   '(I/O names auto-detected; third-party graphs work — '
                                   'override normalization via trailing opts)')
    ap.add_argument('--mode', choices=('query', 'all'), default='query',
                    help="'query': shift queries only (default); 'all': shift both sides")
    ap.add_argument('--dataset', choices=('reid', 'official'), default='reid',
                    help="'reid': unified test split (default); 'official': the five "
                         "source datasets' official splits, matched within each "
                         "dataset and summarized as one query-weighted table")
    ap.add_argument('--markdown', action='store_true',
                    help='print the results table in Markdown (paste-ready for README/docs)')
    ap.add_argument('--recompute', action='store_true',
                    help='ignore eval_cache.json and re-evaluate')
    ap.add_argument('opts', nargs=argparse.REMAINDER,
                    help='extra config overrides in KEY VALUE form')
    args = ap.parse_args()

    # opts is REMAINDER: a trailing --markdown lands in it, so recover it here
    if '--markdown' in args.opts:
        args.opts = [token for token in args.opts if token != '--markdown']
        args.markdown = True

    if bool(args.weight) == bool(args.onnx):
        ap.error('exactly one of --weight / --onnx is required')
    if args.onnx:
        target = args.onnx
    else:
        weight = sorted(glob.glob(args.weight))
        if not weight:
            raise FileNotFoundError('no checkpoint matches {}'.format(args.weight))
        target = weight[-1]

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(['MODEL.PRETRAIN_CHOICE', 'none'])
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    cache = EvalCache(target, enabled=not args.recompute)
    key = {'tool': 'eval_style_shift', 'checkpoint': checkpoint_signature(target),
           'config': os.path.basename(args.config), 'mode': args.mode,
           'opts': args.opts}
    if args.dataset != 'reid':  # keep pre-existing unified-split cache keys valid
        key['dataset'] = args.dataset
    else:
        # the unified split can change on disk (new domains get integrated),
        # so fingerprint the val set in the key: TEST.VAL_DOMAINS restricts
        # it (legacy-comparable numbers) and the counts invalidate entries
        # cached against an older build
        reid_dataset = REID(root=cfg.DATASETS.ROOT_DIR, verbose=False)
        if cfg.TEST.VAL_DOMAINS:
            keep = set(cfg.TEST.VAL_DOMAINS)

            def _domain_of(sample):
                return int(re.search(r'_d(\d+)_',
                                     os.path.basename(sample[0])).group(1))

            reid_dataset.query = [s for s in reid_dataset.query
                                  if _domain_of(s) in keep]
            reid_dataset.gallery = [s for s in reid_dataset.gallery
                                    if _domain_of(s) in keep]
        key['val'] = {'domains': sorted(cfg.TEST.VAL_DOMAINS) or 'all',
                      'query': len(reid_dataset.query),
                      'gallery': len(reid_dataset.gallery)}
    rows = cache.get(key)
    if rows is not None and {row['condition'] for row in rows} != set(CONDITIONS):
        rows = None  # the condition set grew since this entry was cached
    from_cache = rows is not None

    if rows is None:
        if args.onnx:
            from tools.eval_official_onnx import build_session
            session = build_session(target)

            def run_extract(samples, transforms, desc):
                return extract_onnx(session, samples, transforms, desc=desc)
        else:
            model = make_model(cfg, num_class=751, camera_num=0, view_num=0)
            model.load_param(target)
            model.to('cuda')

            def run_extract(samples, transforms, desc):
                return extract(model, samples, transforms, desc=desc)

        if args.dataset == 'reid':
            datasets = [('reid', reid_dataset)]
        else:
            datasets = [(ds_name, loader(root=cfg.DATASETS.ROOT_DIR, verbose=False))
                        for ds_name, loader in OFFICIAL_DATASETS.items()]

        # accumulate query-weighted sums per condition; matching stays within
        # each dataset, so the aggregate equals the mean over all queries
        totals = {name: [0.0, 0.0, 0] for name in CONDITIONS}
        for ds_name, dataset in datasets:
            q_pids = np.array([s[1] for s in dataset.query])
            q_camids = np.array([s[2] for s in dataset.query])
            g_pids = np.array([s[1] for s in dataset.gallery])
            g_camids = np.array([s[2] for s in dataset.gallery])

            clean_gallery = run_extract(dataset.gallery, build_transforms(None),
                                        '{} gallery (clean)'.format(ds_name))

            for name, fn in CONDITIONS.items():
                transforms = build_transforms(fn)
                qf = run_extract(dataset.query, transforms,
                                 '{} queries ({})'.format(ds_name, name))
                gf = (clean_gallery if args.mode == 'query' or fn is None
                      else run_extract(dataset.gallery, transforms,
                                       '{} gallery ({})'.format(ds_name, name)))
                cmc, mAP = eval_func(euclidean_distance(qf, gf),
                                     q_pids, g_pids, q_camids, g_camids)[:2]
                totals[name][0] += float(mAP) * len(q_pids)
                totals[name][1] += float(cmc[0]) * len(q_pids)
                totals[name][2] += len(q_pids)

        rows = [{'condition': name,
                 'mAP': totals[name][0] / totals[name][2],
                 'r1': totals[name][1] / totals[name][2]}
                for name in CONDITIONS]
        cache.put(key, rows,
                  log_lines=['dataset: {} / mode: {} shifted'.format(args.dataset, args.mode)]
                  + _render(rows, markdown=False))

    print('\nmodel  : {}'.format(target))
    if args.dataset == 'reid':
        print('dataset: unified reid test split{}'.format(
            ' (domains {})'.format(sorted(cfg.TEST.VAL_DOMAINS))
            if cfg.TEST.VAL_DOMAINS else ''))
    else:
        print('dataset: official splits (query-weighted aggregate)')
    print('mode   : {} shifted'.format('query only' if args.mode == 'query' else 'query+gallery'))
    if from_cache:
        print('cache  : reused from {}'.format(cache.path))
    if args.markdown:
        print('| condition | mAP | R1 | dmAP | dR1 |')
        print('| --- | ---: | ---: | ---: | ---: |')
    else:
        print('condition    |    mAP     R1   | dmAP    dR1')
    for line in _render(rows, markdown=args.markdown):
        print(line)


def _render(rows, markdown):
    clean = next(row for row in rows if row['condition'] == 'clean')
    lines = []
    for row in rows:
        name, mAP, r1 = row['condition'], row['mAP'], row['r1']
        if name == 'clean':
            lines.append('| {} | {:.4f} | {:.4f} | — | — |'.format(name, mAP, r1)
                         if markdown else
                         '{:12s} | {:.4f} {:.4f} |    —      —'.format(name, mAP, r1))
        else:
            lines.append(
                '| {} | {:.4f} | {:.4f} | {:+.4f} | {:+.4f} |'.format(
                    name, mAP, r1, mAP - clean['mAP'], r1 - clean['r1'])
                if markdown else
                '{:12s} | {:.4f} {:.4f} | {:+.4f} {:+.4f}'.format(
                    name, mAP, r1, mAP - clean['mAP'], r1 - clean['r1']))
    return lines


if __name__ == '__main__':
    main()
