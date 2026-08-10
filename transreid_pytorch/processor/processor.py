import logging
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, euclidean_distance, eval_func
from torch import amp
import torch.distributed as dist


def _raw_model(model):
    return model.module if hasattr(model, 'module') else model


def save_resume_state(cfg, model, optimizer, optimizer_center, scheduler, scaler,
                      epoch, best_map, best_path):
    """Write a full-restore checkpoint, atomically replacing the previous one."""
    state = {
        'epoch': epoch,
        'model': _raw_model(model).state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_center': optimizer_center.state_dict() if optimizer_center is not None else None,
        'scheduler': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        'scaler': scaler.state_dict(),
        'best_map': best_map,
        'best_path': best_path,
        'rng_python': random.getstate(),
        'rng_numpy': np.random.get_state(),
        'rng_torch': torch.get_rng_state(),
        'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path = os.path.join(cfg.OUTPUT_DIR, 'checkpoint_last.pth')
    tmp_path = path + '.tmp'
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_resume_state(path, model, optimizer, optimizer_center, scheduler, scaler, logger):
    """Restore the state written by save_resume_state; returns (start_epoch, best_map, best_path)."""
    state = torch.load(path, map_location='cpu', weights_only=False)
    _raw_model(model).load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    if state.get('optimizer_center') is not None and optimizer_center is not None:
        optimizer_center.load_state_dict(state['optimizer_center'])
    if state.get('scheduler') is not None and hasattr(scheduler, 'load_state_dict'):
        scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    random.setstate(state['rng_python'])
    np.random.set_state(state['rng_numpy'])
    torch.set_rng_state(state['rng_torch'])
    if torch.cuda.is_available() and state.get('rng_cuda') is not None:
        torch.cuda.set_rng_state_all(state['rng_cuda'])
    start_epoch = state['epoch'] + 1
    best_map = state.get('best_map', 0.0)
    best_path = state.get('best_path')
    logger.info('Resumed from {} (finished epoch {}, best mAP {:.5f})'.format(
        path, state['epoch'], best_map))
    return start_epoch, best_map, best_path


def save_best_model(cfg, model, epoch, mAP, best_path, logger):
    """Save the new best model and drop the previous best file; returns its path."""
    new_best = os.path.join(
        cfg.OUTPUT_DIR,
        '{}_best_e{:06d}_map{:.5f}.pth'.format(cfg.MODEL.NAME, epoch, mAP))
    torch.save(_raw_model(model).state_dict(), new_best)
    if best_path and best_path != new_best and os.path.exists(best_path):
        os.remove(best_path)
    logger.info('New best model (mAP {:.5f}) saved to {}'.format(mAP, new_best))
    return new_best


def _shifted_query_map(model, loader, evaluator, gf, device):
    """mAP of style-shifted queries against the clean-gallery features `gf`
    of the evaluation that just ran (gf is already feat-normalized whenever
    the evaluator normalized)."""
    feats = []
    with torch.no_grad():
        for img, pid, camid, camids, target_view, _ in loader:
            feat = model(img.to(device), cam_label=camids.to(device),
                         view_label=target_view.to(device))
            feats.append(feat.detach().cpu())
    qf = torch.cat(feats, dim=0)
    if evaluator.feat_norm:
        qf = torch.nn.functional.normalize(qf, dim=1, p=2)
    n = evaluator.num_query
    return eval_func(euclidean_distance(qf, gf),
                     np.asarray(evaluator.pids[:n]), np.asarray(evaluator.pids[n:]),
                     np.asarray(evaluator.camids[:n]), np.asarray(evaluator.camids[n:]))[1]


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank,
             teacher=None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            logger.info('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    distill_criterion = None
    if teacher is not None:
        from loss.distill_loss import DistillLoss
        distill_criterion = DistillLoss(cfg.DISTILL.LOGIT_WEIGHT,
                                        cfg.DISTILL.REL_WEIGHT,
                                        cfg.DISTILL.EMBED_WEIGHT,
                                        cfg.DISTILL.TEMPERATURE)
        teacher.to(local_rank)
        teacher.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    distill_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler('cuda')

    val_shift_loader = None
    if cfg.SOLVER.VAL_SHIFT:
        if cfg.MODEL.DIST_TRAIN:
            raise NotImplementedError('SOLVER.VAL_SHIFT supports single-GPU training only')
        from torch.utils.data import DataLoader
        from datasets.bases import ImageDataset
        from datasets.make_dataloader import val_collate_fn
        from datasets.style_shift import build_shift_val_transforms
        val_shift_loader = DataLoader(
            ImageDataset(val_loader.dataset.dataset[:num_query],
                         build_shift_val_transforms(cfg, cfg.SOLVER.VAL_SHIFT)),
            batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS, collate_fn=val_collate_fn)
        logger.info('shift-aware validation enabled: best selected on mean of '
                    'clean and {!r}-shifted query mAP'.format(cfg.SOLVER.VAL_SHIFT))

    start_epoch = 1
    best_map = 0.0
    best_path = None
    if cfg.SOLVER.RESUME:
        start_epoch, best_map, best_path = load_resume_state(
            cfg.SOLVER.RESUME, model, optimizer, optimizer_center, scheduler, scaler, logger)
        if start_epoch > epochs:
            logger.info('Nothing to do: resumed epoch {} already reached MAX_EPOCHS {}'.format(
                start_epoch - 1, epochs))

    # train
    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        distill_meter.reset()
        evaluator.reset()
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast('cuda', enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)
                if distill_criterion is not None:
                    teacher_score, teacher_feat = teacher(img, cam_label=target_cam,
                                                          view_label=target_view)
                    distill_loss = distill_criterion(score, feat, teacher_score, teacher_feat)
                    distill_meter.update(distill_loss.item(), img.shape[0])
                    loss = loss + distill_loss

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0 and (not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0):
                base_lr = scheduler._get_lr(epoch)[0] if cfg.SOLVER.WARMUP_METHOD == 'cosine' else scheduler.get_lr()[0]
                msg = "Epoch[{}] Iter[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}".format(
                    epoch, (n_iter + 1), len(train_loader), loss_meter.avg, acc_meter.avg, base_lr)
                if distill_criterion is not None:
                    msg += ", Distill: {:.3f}".format(distill_meter.avg)
                logger.info(msg)

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.SOLVER.WARMUP_METHOD == 'cosine':
            scheduler.step(epoch)
        else:
            scheduler.step()
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per epoch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch * (n_iter + 1), train_loader.batch_size / time_per_batch))

        # Periodic fixed-epoch checkpoints are disabled when best-model saving
        # is enabled; the best model and checkpoint_last.pth replace them.
        if not cfg.SOLVER.SAVE_BEST and epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    if cfg.SOLVER.SAVE_BEST and mAP > best_map:
                        best_map = mAP
                        best_path = save_best_model(cfg, model, epoch, mAP, best_path, logger)
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, gf = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                selection = mAP
                if val_shift_loader is not None:
                    shift_map = _shifted_query_map(model, val_shift_loader, evaluator, gf, device)
                    selection = 0.5 * (mAP + shift_map)
                    logger.info("Shifted-query ({}) mAP: {:.1%}; selection metric (mean): {:.1%}".format(
                        cfg.SOLVER.VAL_SHIFT, shift_map, selection))
                if cfg.SOLVER.SAVE_BEST and selection > best_map:
                    best_map = selection
                    best_path = save_best_model(cfg, model, epoch, selection, best_path, logger)
                torch.cuda.empty_cache()

        if not cfg.MODEL.DIST_TRAIN or dist.get_rank() == 0:
            save_resume_state(cfg, model, optimizer, optimizer_center, scheduler, scaler,
                              epoch, best_map, best_path)


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


