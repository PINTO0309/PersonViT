"""Evaluate a trained model on the original datasets' official splits.

Runs the standard single-dataset protocols (official query/gallery of each
source dataset) as a per-dataset performance reference for models from this
repository. Requires the original datasets under DATASETS.ROOT_DIR with the
canonical loader names (symlinks are fine):

    market1501 -> Market-1501-v15.09.15
    MSMT17     -> MSMT17_V1
    Occluded_Duke -> Occluded-DukeMTMC
    CUHK03-NP/detected, Occluded_REID (as-is)

Usage (from transreid_pytorch/):
    python tools/eval_official.py \
        --config configs/reid/osnet_n_8gb_distill.yml \
        --weight "logs/reid_osnet_n_8gb_distill/transformer_best_*.pth" \
        [--datasets market cuhk03np ...]
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import cfg
from datasets.bases import ImageDataset
from datasets.cuhk03np import CUHK03NP
from datasets.make_dataloader import val_collate_fn
from datasets.market1501 import Market1501
from datasets.msmt17 import MSMT17
from datasets.occ_duke import OCC_DukeMTMCreID
from datasets.occ_reid import OccludedREID
from model import make_model
from utils.metrics import euclidean_distance, eval_func

OFFICIAL_DATASETS = {
    'market': Market1501,
    'msmt17': MSMT17,
    'duke_occ': OCC_DukeMTMCreID,
    'cuhk03np': CUHK03NP,
    'occ_reid': OccludedREID,
}


def extract_features(model, loader, device='cuda'):
    feats = []
    model.eval()
    with torch.no_grad():
        for img, pid, camid, camids, target_view, _ in loader:
            img = img.to(device)
            camids_t = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids_t, view_label=target_view)
            feats.append(feat.detach().cpu())
    return torch.cat(feats, dim=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', required=True,
                    help='config defining the model architecture and input pipeline')
    ap.add_argument('--weight', required=True, help='trained checkpoint (glob allowed)')
    ap.add_argument('--datasets', nargs='+', default=['all'],
                    choices=['all', *OFFICIAL_DATASETS])
    ap.add_argument('opts', nargs=argparse.REMAINDER,
                    help='extra config overrides in KEY VALUE form')
    args = ap.parse_args()

    weight = sorted(glob.glob(args.weight))
    if not weight:
        raise FileNotFoundError('no checkpoint matches {}'.format(args.weight))
    weight = weight[-1]

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(['MODEL.PRETRAIN_CHOICE', 'none'])
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    selected = list(OFFICIAL_DATASETS) if 'all' in args.datasets else args.datasets

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])

    # the classifier head is skipped by load_param, so its size is irrelevant
    model = make_model(cfg, num_class=751, camera_num=0, view_num=0)
    model.load_param(weight)
    model.to('cuda')

    print('\nmodel  : {}'.format(weight))
    print('dataset  |    #q |     #g |    mAP     R1     R5    R10')
    for name in selected:
        dataset = OFFICIAL_DATASETS[name](root=cfg.DATASETS.ROOT_DIR, verbose=False)
        samples = dataset.query + dataset.gallery
        loader = DataLoader(
            ImageDataset(samples, val_transforms),
            batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
            num_workers=4, collate_fn=val_collate_fn,
        )
        feats = extract_features(model, loader)
        if cfg.TEST.FEAT_NORM == 'yes':
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)
        num_query = len(dataset.query)
        qf, gf = feats[:num_query], feats[num_query:]
        pids = np.array([s[1] for s in samples])
        camids = np.array([s[2] for s in samples])

        distmat = euclidean_distance(qf, gf)
        cmc, mAP = eval_func(distmat, pids[:num_query], pids[num_query:],
                             camids[:num_query], camids[num_query:])[:2]
        print('{:8s} | {:5d} | {:6d} | {:.4f} {:.4f} {:.4f} {:.4f}'.format(
            name, num_query, len(gf), mAP, cmc[0], cmc[4], cmc[9]))
        del feats, qf, gf, distmat


if __name__ == '__main__':
    main()
