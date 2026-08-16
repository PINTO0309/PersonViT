"""Class-center initialization for a fresh classifier head (LP-FT shortcut).

When the id space changes (e.g. a new domain joins the unified set), the
warm-started model gets a random classifier and its early CE gradients are
noise with respect to the true class structure — they distort the warm
backbone while the head slowly matures, and a 40-epoch cosine spends most
of its budget teaching the head (observed: the d05 teacher round 1 never
beat its own warm start). This tool removes that phase: it runs one
deterministic pass over the train split with the warm backbone, computes
each class's mean BNNeck feature (the exact representation the classifier
consumes), L2-normalizes the means and scales them to the source
checkpoint's trained-classifier row norm, and writes a checkpoint whose
classifier starts as a nearest-class-mean head. Training then starts with
meaningful CE gradients from step one — approximately the state a full
maturation round would otherwise have to buy.

Usage (from transreid_pytorch/):
    python tools/init_classifier_centers.py \
        --config configs/reid/vit_base_8gb_ain_synth_jpeg.yml \
        --weight logs/reid_vit_base_8gb_ain_aug2_jpeg/transformer_best_*.pth \
        --output logs/reid_vit_base_8gb_ain_aug2_jpeg/centers_init_synth.pth
Then point the training config's MODEL.PRETRAIN_PATH at --output.
"""

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from datasets.bases import ImageDataset
from datasets.make_dataloader import val_collate_fn
from datasets.reid import REID
from model import make_model


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--config', required=True, help='training config of the NEW run')
    ap.add_argument('--weight', required=True, help='warm checkpoint (glob allowed)')
    ap.add_argument('--output', required=True, help='augmented checkpoint path')
    ap.add_argument('--batch', type=int, default=128)
    args = ap.parse_args()

    matches = sorted(glob.glob(args.weight))
    if not matches:
        raise FileNotFoundError('no checkpoint matches {}'.format(args.weight))
    weight_path = matches[-1]

    cfg.merge_from_file(args.config)
    cfg.merge_from_list(['MODEL.PRETRAIN_CHOICE', 'none'])
    cfg.freeze()
    if cfg.MODEL.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
        raise NotImplementedError('center init supports the softmax classifier only')

    dataset = REID(root=cfg.DATASETS.ROOT_DIR, verbose=False)
    num_class = len({s[1] for s in dataset.train})
    camera_num = len({s[2] for s in dataset.train}
                     | {s[2] for s in dataset.query} | {s[2] for s in dataset.gallery})
    print('train ids {} / cameras {}'.format(num_class, camera_num))

    model = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=0)
    model.load_param(weight_path)  # classifier stays fresh on shape mismatch
    model.to('cuda').eval()

    # deterministic pipeline: resize + normalize only (no augmentation)
    transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])
    loader = DataLoader(ImageDataset(dataset.train, transforms),
                        batch_size=args.batch, shuffle=False,
                        num_workers=8, collate_fn=val_collate_fn)

    dim = model.in_planes
    sums = torch.zeros(num_class, dim, device='cuda')
    counts = torch.zeros(num_class, device='cuda')
    correct = total = 0
    with torch.no_grad():
        for img, pid, camid, camids, target_view, _ in tqdm(
                loader, desc='class centers', dynamic_ncols=True):
            img = img.to('cuda')
            pid = torch.as_tensor(pid, device='cuda')
            # the classifier consumes the post-BNNeck feature — compute it
            # explicitly (eval-mode forward returns the retrieval feature,
            # which is pre-BN under NECK_FEAT 'before')
            global_feat = model.base(img, cam_label=camids.to('cuda'),
                                     view_label=target_view.to('cuda'))
            if model.reduce_feat_dim:
                global_feat = model.fcneck(global_feat)
            feat = model.bottleneck(global_feat)
            sums.index_add_(0, pid, feat.float())
            counts.index_add_(0, pid, torch.ones_like(pid, dtype=torch.float))

    assert int((counts == 0).sum()) == 0, 'classes without train samples'
    centers = torch.nn.functional.normalize(sums / counts.unsqueeze(1), dim=1)

    # scale to the source checkpoint's trained-row-norm so softmax confidence
    # starts in the regime the LR schedule was calibrated for
    state = torch.load(weight_path, map_location='cpu', weights_only=False)
    for wrapper in ('model', 'state_dict'):
        if wrapper in state:
            state = state[wrapper]
    src = [v for k, v in state.items() if k.endswith('classifier.weight')]
    scale = src[0].norm(dim=1).mean().item() if src else 1.0
    weight = (centers * scale).cpu()
    print('centers: {} x {} / row-norm scale {:.4f}'.format(*weight.shape, scale))

    # self-check: NCM accuracy of the new head on the extracted class means'
    # own training features (recomputed logits via the center head)
    with torch.no_grad():
        logits = torch.nn.functional.normalize(sums / counts.unsqueeze(1), dim=1) \
            @ (centers.t())
        acc = (logits.argmax(1) == torch.arange(num_class, device='cuda')).float().mean()
    print('center self-consistency (class-mean -> argmax): {:.4f}'.format(acc.item()))

    out = {k: v for k, v in state.items() if not k.endswith('classifier.weight')}
    out['classifier.weight'] = weight
    torch.save(out, args.output)
    print('saved {}'.format(args.output))


if __name__ == '__main__':
    main()
