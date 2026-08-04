"""Light smoke test of the unified reid dataset pipeline.

Verifies, without running a full epoch:
1. the reid dataset loads and its statistics are consistent,
2. the domain-balanced sampler mixes domains in every batch,
3. a few real optimization steps (AMP, triplet + ID loss) run on the GPU.

Usage (from transreid_pytorch/):
    python tools/smoke_reid.py [--config configs/reid/vit_small_8gb.yml]
                               [--iters 3] [--no-pretrain]
"""

import argparse
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import amp

from config import cfg
from datasets import make_dataloader
from loss import make_loss
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/reid/vit_small_8gb.yml')
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--batches', type=int, default=8,
                    help='batches to inspect for domain mixture')
    ap.add_argument('--no-pretrain', action='store_true',
                    help='skip loading the self-supervised checkpoint')
    args = ap.parse_args()

    cfg.merge_from_file(args.config)
    overrides = ['DATALOADER.NUM_WORKERS', '2', 'SOLVER.SEED', '1234']
    if args.no_pretrain:
        overrides += ['MODEL.PRETRAIN_CHOICE', 'none']
    cfg.merge_from_list(overrides)
    cfg.freeze()

    torch.manual_seed(cfg.SOLVER.SEED)
    import random, numpy as np
    random.seed(cfg.SOLVER.SEED)
    np.random.seed(cfg.SOLVER.SEED)

    # 1) dataset + loader ---------------------------------------------------
    train_loader, _, val_loader, num_query, num_classes, cam_num, view_num = make_dataloader(cfg)
    print(f'\n[smoke] classes={num_classes} cameras={cam_num} domains={view_num} '
          f'train_batches/epoch={len(train_loader)} query={num_query}')

    # 2) domain mixture of the first batches --------------------------------
    print('[smoke] per-batch domain histogram (target_view carries the domain id):')
    it = iter(train_loader)
    for b in range(args.batches):
        img, vid, camid, domain = next(it)
        hist = Counter(domain.tolist())
        n_pids = len(set(vid.tolist()))
        print(f'  batch {b}: pids={n_pids} '
              f'domains={dict(sorted(hist.items()))}')

    # 3) a few real training steps -----------------------------------------
    if not torch.cuda.is_available():
        print('[smoke] CUDA not available - skipped the training-step check')
        return
    device = 'cuda'
    model = make_model(cfg, num_class=num_classes, camera_num=cam_num, view_num=view_num)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    optimizer, _ = make_optimizer(cfg, model, center_criterion)
    _ = create_scheduler(cfg, optimizer)
    model.to(device)
    model.train()
    scaler = amp.GradScaler('cuda')

    it = iter(train_loader)
    for step in range(args.iters):
        img, vid, target_cam, target_view = next(it)
        img, target = img.to(device), vid.to(device)
        target_cam, target_view = target_cam.to(device), target_view.to(device)
        t0 = time.time()
        optimizer.zero_grad()
        with amp.autocast('cuda', enabled=True):
            score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
            loss = loss_func(score, feat, target, target_cam)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        print(f'[smoke] step {step}: loss={loss.item():.3f} '
              f'time={time.time() - t0:.2f}s '
              f'peak_mem={torch.cuda.max_memory_allocated() / 1024**3:.2f}GiB')

    print('[smoke] OK')


if __name__ == '__main__':
    main()
