"""Per-domain benchmark of an exported ONNX model on the unified test split.

Feature extraction runs through ONNX Runtime on the deployment artifact
itself; preprocessing reuses the exact evaluation dataloader of the training
pipeline, so the numbers are directly comparable to the PyTorch evaluations
(and validate the artifact end to end). Reports the same two protocols as
tools/eval_per_domain.py: within-domain and merged-gallery.

Usage (from transreid_pytorch/):
    python tools/eval_per_domain_onnx.py \
        --config configs/reid/osnet_n_8gb_distill.yml \
        --onnx ../onnx/osnet_x1_25_n_unified.onnx
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import onnxruntime as ort
import torch

from config import cfg
from datasets import make_dataloader
from datasets.reid import REID
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


def extract_features(session: ort.InferenceSession, val_loader) -> torch.Tensor:
    input_meta = session.get_inputs()[0]
    fixed_batch = isinstance(input_meta.shape[0], int)
    feats = []
    for img, *_ in val_loader:
        batch = img.numpy()
        if fixed_batch and input_meta.shape[0] == 1:
            outputs = [
                session.run(["embeddings"], {"images": row[None]})[0]
                for row in batch
            ]
            feats.append(np.concatenate(outputs, axis=0))
        else:
            feats.append(session.run(["embeddings"], {"images": batch})[0])
    return torch.from_numpy(np.concatenate(feats, axis=0))


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids):
    cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids)[:2]
    return mAP, cmc[0], cmc[4]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', required=True,
                    help='training config that defines the evaluation input pipeline')
    ap.add_argument('--onnx', required=True, help='exported ONNX model path')
    ap.add_argument('opts', nargs=argparse.REMAINDER,
                    help='extra config overrides in KEY VALUE form')
    args = ap.parse_args()

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(['DATALOADER.NUM_WORKERS', '4',
                         'MODEL.PRETRAIN_CHOICE', 'none'])
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    _, _, val_loader, num_query, _, _, _ = make_dataloader(cfg)
    dataset = REID(root=cfg.DATASETS.ROOT_DIR, verbose=False)
    samples = dataset.query + dataset.gallery
    pids = np.array([s[1] for s in samples])
    camids = np.array([s[2] for s in samples])
    doms = np.array([s[3] for s in samples])

    session = build_session(args.onnx)
    feats = extract_features(session, val_loader)
    # exported embeddings are already L2-normalized; renormalizing is a no-op
    feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    qf, gf = feats[:num_query], feats[num_query:]
    q_pids, g_pids = pids[:num_query], pids[num_query:]
    q_camids, g_camids = camids[:num_query], camids[num_query:]
    q_doms, g_doms = doms[:num_query], doms[num_query:]

    distmat = euclidean_distance(qf, gf)

    print('\nmodel  : {}'.format(args.onnx))
    print('overall: mAP {:.4f}  Rank-1 {:.4f}  Rank-5 {:.4f}  '
          '({} query / {} gallery)'.format(
              *evaluate(distmat, q_pids, g_pids, q_camids, g_camids),
              num_query, len(g_pids)))
    print('domain |    #q |     #g | within: mAP    R1     R5   | merged: mAP    R1     R5')
    for d in sorted(set(q_doms.tolist())):
        iq = q_doms == d
        ig = g_doms == d
        within = evaluate(
            euclidean_distance(qf[torch.as_tensor(iq)], gf[torch.as_tensor(ig)]),
            pids[:num_query][iq], g_pids[ig], q_camids[iq], g_camids[ig])
        merged = evaluate(distmat[iq], q_pids[iq], g_pids, q_camids[iq], g_camids)
        print('d{:02d}    | {:5d} | {:6d} | {:.4f} {:.4f} {:.4f} | {:.4f} {:.4f} {:.4f}'.format(
            d, int(iq.sum()), int(ig.sum()), *within, *merged))


if __name__ == '__main__':
    main()
