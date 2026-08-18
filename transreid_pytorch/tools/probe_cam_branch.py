"""Stage-0 feasibility probe for the camera-proxy mimicry branch.

Question: can ANY small head on top of the FROZEN no-cam CNN student's
features represent the L_cam teacher geometry (or the cam/no-cam teacher
delta)? If yes, a gradient-isolated mimicry branch (Stage 1+) is worth
building; if every tap point floors near the historic 0.77 relational-KD
wall, the branch cannot work either and the stem-IN integration path is
the only remaining route.

Design: the student trunk and both teachers stay frozen; per tap point
(stem / conv2 / conv3 / conv4 / final embedding) two small MLP heads are
trained jointly in one pass over a train subsample —
  - target 'cam'  : the L_cam teacher embedding; trained with
                    relational-KD + cosine, judged by val relational-KD
                    reported in training units (x REL_WEIGHT 30) against
                    the historic floors (CNN 0.77 / ViT-S 0.49 /
                    representable 0.50 — those Distill logs also carry a
                    small logit-KD term, so compare with ~0.1 slack;
                    feasibility gate < 0.6)
  - target 'delta': (cam teacher − no-cam teacher) embedding — the pure
                    L_cam component; judged by val cosine similarity.
Early taps matter because of the IN-information-loss hypothesis: if the
INs discard what L_cam needs, only pre-IN-cascade features can carry it.

Usage (from transreid_pytorch/):
    python tools/probe_cam_branch.py \
        --student-weight logs/reid_osnet_p_8gb_distill_ain_synth_shint/folded_best.pth
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import cfg
from datasets.bases import ImageDataset
from datasets.make_dataloader import val_collate_fn
from datasets.reid import REID
from model import make_model
from model.teacher import build_teacher

TAPS = ('stem', 'conv2', 'conv3', 'conv4', 'embed')
REFS = 'floors: CNN-vs-L_cam 0.77 / ViT-S-vs-L_cam 0.49 / representable 0.50'


class ProbeHead(nn.Module):
    """GAP -> 2-layer MLP into the teacher's 768-dim space."""

    def __init__(self, in_ch, out_dim=768):
        super(ProbeHead, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_ch, 512), nn.GELU(),
                                 nn.Linear(512, out_dim))

    def forward(self, x):
        if x.dim() == 4:
            x = x.mean(dim=(2, 3))
        return self.mlp(x.float())


def rel_loss(student, teacher):
    s = F.normalize(student.float(), dim=1)
    t = F.normalize(teacher.float(), dim=1)
    return F.mse_loss(s @ s.t(), t @ t.t())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--student-config',
                    default='configs/reid/osnet_p_8gb_distill_ain_aug2_jpeg.yml',
                    help='plain-arch config matching the folded student weights')
    ap.add_argument('--student-weight', required=True, help='folded student checkpoint')
    ap.add_argument('--cam-teacher-config',
                    default='configs/reid/vit_base_8gb_ain_synth_cam_jpeg2.yml')
    ap.add_argument('--cam-teacher-weight',
                    default='logs/reid_vit_base_8gb_ain_synth_cam_jpeg2/transformer_best_*.pth')
    ap.add_argument('--nocam-teacher-config',
                    default='configs/reid/vit_base_8gb_ain_synth_jpeg.yml')
    ap.add_argument('--nocam-teacher-weight',
                    default='logs/reid_vit_base_8gb_ain_synth_jpeg/teacher_synth_e40.pth')
    ap.add_argument('--limit', type=int, default=64000,
                    help='train subsample size (seeded); 0 = full train set')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    args = ap.parse_args()

    cfg.merge_from_file(args.student_config)
    cfg.merge_from_list(['MODEL.PRETRAIN_CHOICE', 'none'])

    dataset = REID(root=cfg.DATASETS.ROOT_DIR, verbose=False)
    num_class = len({s[1] for s in dataset.train})
    camera_num = len({s[2] for s in dataset.train}
                     | {s[2] for s in dataset.query} | {s[2] for s in dataset.gallery})

    student = make_model(cfg, num_class=num_class, camera_num=camera_num, view_num=0)
    student.load_param(args.student_weight)
    student.to('cuda').eval()
    for p in student.parameters():
        p.requires_grad_(False)

    def frozen_teacher(tconfig, tweight):
        tcfg = cfg.clone()
        tcfg.defrost()
        tcfg.DISTILL.ENABLED = True
        tcfg.DISTILL.TEACHER_CONFIG = tconfig
        tcfg.DISTILL.TEACHER_WEIGHT = tweight
        tcfg.DISTILL.HINT_WEIGHT = 0.0
        tcfg.freeze()
        teacher = build_teacher(tcfg, num_classes=num_class,
                                camera_num=camera_num, view_num=0)
        return teacher.to('cuda')

    teacher_cam = frozen_teacher(args.cam_teacher_config, args.cam_teacher_weight)
    teacher_nocam = frozen_teacher(args.nocam_teacher_config, args.nocam_teacher_weight)

    # non-destructive taps on the frozen trunk
    captured = {}
    base = student.base
    for name, module in (('stem', base.maxpool), ('conv2', base.conv2),
                         ('conv3', base.conv3), ('conv4', base.conv4)):
        module.register_forward_hook(
            lambda m, i, o, key=name: captured.__setitem__(key, o))

    tap_dims = {'stem': 64, 'conv2': 256, 'conv3': 384, 'conv4': 512, 'embed': 512}
    heads = {}
    optimizers = {}
    for tap in TAPS:
        for target in ('cam', 'delta'):
            key = '{}/{}'.format(tap, target)
            heads[key] = ProbeHead(tap_dims[tap]).to('cuda')
            optimizers[key] = torch.optim.Adam(heads[key].parameters(), lr=args.lr)

    transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
    ])
    train_samples = list(dataset.train)
    if args.limit and args.limit < len(train_samples):
        train_samples = random.Random(0).sample(train_samples, args.limit)
    train_loader = DataLoader(ImageDataset(train_samples, transforms),
                              batch_size=args.batch, shuffle=True,
                              num_workers=8, drop_last=True,
                              collate_fn=val_collate_fn)
    val_loader = DataLoader(ImageDataset(list(dataset.query), transforms),
                            batch_size=128, shuffle=False, num_workers=8,
                            drop_last=True, collate_fn=val_collate_fn)

    def forward_frozen(img, camids, target_view):
        with torch.no_grad(), torch.autocast('cuda'):
            embed = student(img, cam_label=camids, view_label=target_view)
            t_cam = teacher_cam(img, cam_label=camids, view_label=target_view)[1]
            t_nocam = teacher_nocam(img, cam_label=camids, view_label=target_view)[1]
        feats = {k: v.detach() for k, v in captured.items()}
        feats['embed'] = embed.detach()
        return feats, t_cam.detach().float(), (t_cam - t_nocam).detach().float()

    for epoch in range(args.epochs):
        for img, pid, camid, camids, target_view, _ in tqdm(
                train_loader, desc='probe train e{}'.format(epoch + 1),
                dynamic_ncols=True):
            feats, t_cam, t_delta = forward_frozen(
                img.to('cuda'), camids.to('cuda'), target_view.to('cuda'))
            for tap in TAPS:
                for target, tgt in (('cam', t_cam), ('delta', t_delta)):
                    key = '{}/{}'.format(tap, target)
                    out = heads[key](feats[tap])
                    if target == 'cam':
                        loss = rel_loss(out, tgt) + \
                            0.5 * (1 - F.cosine_similarity(out, tgt, dim=1)).mean()
                    else:
                        loss = (1 - F.cosine_similarity(out, tgt, dim=1)).mean()
                    optimizers[key].zero_grad(set_to_none=True)
                    loss.backward()
                    optimizers[key].step()

    # validation
    sums = {k: 0.0 for k in heads}
    batches = 0
    for img, pid, camid, camids, target_view, _ in tqdm(val_loader, desc='probe val',
                                                        dynamic_ncols=True):
        feats, t_cam, t_delta = forward_frozen(
            img.to('cuda'), camids.to('cuda'), target_view.to('cuda'))
        batches += 1
        with torch.no_grad():
            for tap in TAPS:
                for target, tgt in (('cam', t_cam), ('delta', t_delta)):
                    key = '{}/{}'.format(tap, target)
                    out = heads[key](feats[tap])
                    if target == 'cam':
                        sums[key] += rel_loss(out, tgt).item()
                    else:
                        sums[key] += F.cosine_similarity(out, tgt, dim=1).mean().item()

    print('\nstudent : {}'.format(args.student_weight))
    print('samples : {} train / {} val batches ({})'.format(
        len(train_samples), batches, REFS))
    print('| tap | rel-KD x30 vs cam teacher (gate < 0.6) | cos to L_cam delta |')
    print('| --- | ---: | ---: |')
    for tap in TAPS:
        print('| {} | {:.4f} | {:.4f} |'.format(
            tap, 30.0 * sums['{}/cam'.format(tap)] / batches,
            sums['{}/delta'.format(tap)] / batches))


if __name__ == '__main__':
    main()
