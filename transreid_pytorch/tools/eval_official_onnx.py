"""Evaluate an ONNX model on the original datasets' official splits.

The ONNX counterpart of tools/eval_official.py: feature extraction runs
through ONNX Runtime on the deployment artifact itself, so exported models
(and third-party ONNX graphs such as the upstream torchreid OSNet-AIN) get
the same per-dataset reference numbers. Input/output tensor names are taken
from the session, so foreign naming conventions (e.g. ``base_images`` /
``features``) work unchanged; features are L2-normalized on the evaluation
side, so unnormalized outputs are handled too.

The config supplies the input pipeline only (SIZE_TEST, PIXEL_MEAN/STD,
batch size). Third-party torchreid models expect ImageNet normalization —
override it via the trailing opts:

Usage (from transreid_pytorch/):
    python tools/eval_official_onnx.py \
        --config configs/reid/osnet_p_8gb_distill_ain.yml \
        --onnx ../onnx/osnet_ain_ms_d_c_Nx3x256x128.onnx \
        INPUT.PIXEL_MEAN "[0.485,0.456,0.406]" INPUT.PIXEL_STD "[0.229,0.224,0.225]"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import cfg
from datasets.bases import ImageDataset
from datasets.make_dataloader import val_collate_fn
from tools.eval_cache import EvalCache, checkpoint_signature
from tools.eval_official import OFFICIAL_DATASETS, _plain_row
from utils.metrics import euclidean_distance, eval_func


def build_session(onnx_path: str) -> ort.InferenceSession:
    available = ort.get_available_providers()
    providers = [
        provider
        for provider in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if provider in available
    ]
    session = ort.InferenceSession(onnx_path, providers=providers)
    print("providers  :", session.get_providers())
    return session


def extract_features(session: ort.InferenceSession, loader) -> torch.Tensor:
    input_meta = session.get_inputs()[0]
    output_name = session.get_outputs()[0].name
    fixed_batch = isinstance(input_meta.shape[0], int)
    feats = []
    for img, *_ in loader:
        batch = img.numpy()
        if fixed_batch and input_meta.shape[0] == 1:
            outputs = [
                session.run([output_name], {input_meta.name: row[None]})[0]
                for row in batch
            ]
            feats.append(np.concatenate(outputs, axis=0))
        else:
            feats.append(session.run([output_name], {input_meta.name: batch})[0])
    return torch.from_numpy(np.concatenate(feats, axis=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', required=True,
                    help='config supplying the input pipeline (size, mean/std, batch)')
    ap.add_argument('--onnx', required=True, help='ONNX model path')
    ap.add_argument('--datasets', nargs='+', default=['all'],
                    help='subset of: all, {}'.format(', '.join(OFFICIAL_DATASETS)))
    ap.add_argument('--markdown', action='store_true',
                    help='print the results table in Markdown (paste-ready for README/docs)')
    ap.add_argument('--recompute', action='store_true',
                    help='ignore eval_cache.json and re-evaluate')
    ap.add_argument('opts', nargs=argparse.REMAINDER,
                    help='extra config overrides in KEY VALUE form '
                         '(e.g. ImageNet normalization for torchreid models)')
    args = ap.parse_args()

    # opts is REMAINDER: a trailing --markdown lands in it, so recover it here
    if '--markdown' in args.opts:
        args.opts = [token for token in args.opts if token != '--markdown']
        args.markdown = True

    # --datasets is greedy: KEY VALUE override tokens that follow it are
    # captured into args.datasets, so re-route them to opts here
    known = {'all', *OFFICIAL_DATASETS}
    for index, token in enumerate(args.datasets):
        if token not in known:
            args.opts = args.datasets[index:] + args.opts
            args.datasets = args.datasets[:index] or ['all']
            break

    cfg.merge_from_file(args.config)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    selected = list(OFFICIAL_DATASETS) if 'all' in args.datasets else args.datasets

    cache = EvalCache(args.onnx, enabled=not args.recompute)
    base_key = {'tool': 'eval_official_onnx',
                'checkpoint': checkpoint_signature(args.onnx),
                'config': os.path.basename(args.config), 'opts': args.opts}
    cached = {name: cache.get({**base_key, 'dataset': name}) for name in selected}
    pending = [name for name in selected if cached[name] is None]

    session = None
    if pending:
        val_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        ])
        session = build_session(args.onnx)
        print('io     : {} -> {}'.format(session.get_inputs()[0].name,
                                         session.get_outputs()[0].name))

    print('\nmodel  : {}'.format(args.onnx))
    print('pixel  : mean {} / std {}'.format(cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD))
    if len(pending) < len(selected):
        print('cache  : {} of {} dataset(s) reused from {}'.format(
            len(selected) - len(pending), len(selected), cache.path))
    if args.markdown:
        print('| dataset | queries | gallery | mAP | R1 | R5 | R10 |')
        print('| --- | ---: | ---: | ---: | ---: | ---: | ---: |')
    else:
        print('dataset  | queries | gallery |    mAP     R1     R5    R10')
    for name in selected:
        row = cached[name]
        if row is None:
            dataset = OFFICIAL_DATASETS[name](root=cfg.DATASETS.ROOT_DIR, verbose=False)
            samples = dataset.query + dataset.gallery
            loader = DataLoader(
                ImageDataset(samples, val_transforms),
                batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
                num_workers=4, collate_fn=val_collate_fn,
            )
            feats = extract_features(session, loader)
            if cfg.TEST.FEAT_NORM == 'yes':
                feats = torch.nn.functional.normalize(feats, dim=1, p=2)
            num_query = len(dataset.query)
            qf, gf = feats[:num_query], feats[num_query:]
            pids = np.array([s[1] for s in samples])
            camids = np.array([s[2] for s in samples])

            distmat = euclidean_distance(qf, gf)
            cmc, mAP = eval_func(distmat, pids[:num_query], pids[num_query:],
                                 camids[:num_query], camids[num_query:])[:2]
            row = {'queries': num_query, 'gallery': len(gf), 'mAP': float(mAP),
                   'r1': float(cmc[0]), 'r5': float(cmc[4]), 'r10': float(cmc[9])}
            cache.put({**base_key, 'dataset': name}, row,
                      log_lines=[_plain_row(name, row)])
            del feats, qf, gf, distmat
        if args.markdown:
            print('| {} | {:,} | {:,} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
                name, row['queries'], row['gallery'], row['mAP'],
                row['r1'], row['r5'], row['r10']))
        else:
            print(_plain_row(name, row))


if __name__ == '__main__':
    main()
