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
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from config import cfg
from datasets.bases import ImageDataset
from datasets.make_dataloader import val_collate_fn
from datasets.reid import REID
from model import make_model
from utils.metrics import euclidean_distance, eval_func


def _gain(g):
    return lambda x: x * g


def _contrast(c):
    return lambda x: (x - 0.5) * c + 0.5


def _channel_gain(r, g, b):
    def fn(x):
        return x * torch.tensor([r, g, b], dtype=x.dtype).view(3, 1, 1)
    return fn


def _gamma(y):
    return lambda x: x.clamp(min=1e-6) ** y


# deterministic photometric shifts applied on the [0, 1] tensor
CONDITIONS = {
    'clean': None,
    'bright+30%': _gain(1.3),
    'dark-30%': _gain(0.7),
    'contrast-40%': _contrast(0.6),
    'contrast+40%': _contrast(1.4),
    'warm': _channel_gain(1.25, 1.0, 0.8),
    'cool': _channel_gain(0.8, 1.0, 1.25),
    'gamma0.6': _gamma(0.6),
    'gamma1.6': _gamma(1.6),
}


class StyleShift:
    def __init__(self, fn):
        self.fn = fn

    def __call__(self, tensor):
        if self.fn is None:
            return tensor
        return self.fn(tensor).clamp(0.0, 1.0)


def build_transforms(condition_fn):
    return T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        StyleShift(condition_fn),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])


def extract(model, samples, transforms, device='cuda'):
    loader = DataLoader(
        ImageDataset(samples, transforms),
        batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=4, collate_fn=val_collate_fn,
    )
    feats = []
    model.eval()
    with torch.no_grad():
        for img, pid, camid, camids, target_view, _ in loader:
            feat = model(img.to(device), cam_label=camids.to(device),
                         view_label=target_view.to(device))
            feats.append(feat.detach().cpu())
    feats = torch.cat(feats, dim=0)
    if cfg.TEST.FEAT_NORM == 'yes':
        feats = torch.nn.functional.normalize(feats, dim=1, p=2)
    return feats


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', required=True)
    ap.add_argument('--weight', required=True, help='trained checkpoint (glob allowed)')
    ap.add_argument('--mode', choices=('query', 'all'), default='query',
                    help="'query': shift queries only (default); 'all': shift both sides")
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

    model = make_model(cfg, num_class=751, camera_num=0, view_num=0)
    model.load_param(weight)
    model.to('cuda')

    dataset = REID(root=cfg.DATASETS.ROOT_DIR, verbose=False)
    q_pids = np.array([s[1] for s in dataset.query])
    q_camids = np.array([s[2] for s in dataset.query])
    g_pids = np.array([s[1] for s in dataset.gallery])
    g_camids = np.array([s[2] for s in dataset.gallery])

    clean_gallery = extract(model, dataset.gallery, build_transforms(None))

    print('\nmodel  : {}'.format(weight))
    print('mode   : {} shifted'.format('query only' if args.mode == 'query' else 'query+gallery'))
    print('condition    |    mAP     R1   | dmAP    dR1')
    clean_map = clean_r1 = None
    for name, fn in CONDITIONS.items():
        transforms = build_transforms(fn)
        qf = extract(model, dataset.query, transforms)
        gf = (clean_gallery if args.mode == 'query' or fn is None
              else extract(model, dataset.gallery, transforms))
        cmc, mAP = eval_func(euclidean_distance(qf, gf),
                             q_pids, g_pids, q_camids, g_camids)[:2]
        if name == 'clean':
            clean_map, clean_r1 = mAP, cmc[0]
            print('{:12s} | {:.4f} {:.4f} |    —      —'.format(name, mAP, cmc[0]))
        else:
            print('{:12s} | {:.4f} {:.4f} | {:+.4f} {:+.4f}'.format(
                name, mAP, cmc[0], mAP - clean_map, cmc[0] - clean_r1))


if __name__ == '__main__':
    main()
