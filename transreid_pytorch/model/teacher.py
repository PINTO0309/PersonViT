"""Frozen teacher model for knowledge distillation.

The teacher is built from its own training config (architecture, stride,
SIE flags, ...) and a trained checkpoint, so any registered backbone can act
as a teacher regardless of the student architecture. The wrapper reproduces
the training-branch forward (logits + pre-BN embedding) while keeping every
module in eval mode, so BatchNorm running statistics are never updated and
no dropout is applied.
"""

import glob

import torch
import torch.nn as nn

from config.defaults import _C as _defaults
from .make_model import make_model


def resolve_teacher_weight(pattern):
    """Resolve a concrete checkpoint path, allowing glob patterns such as
    logs/reid_vit_base_8gb/transformer_best_*.pth (the trainer keeps a single
    best file, so a pattern match is unambiguous; the newest name wins)."""
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError('teacher weight not found: {}'.format(pattern))
    return matches[-1]


class TeacherModel(nn.Module):
    """Wraps a trained build_transformer model as a frozen (score, feat) head."""

    def __init__(self, core):
        super(TeacherModel, self).__init__()
        if not hasattr(core, 'bottleneck') or not hasattr(core, 'classifier'):
            raise TypeError('unsupported teacher model: expected build_transformer')
        if getattr(core, 'ID_LOSS_TYPE', 'softmax') in ('arcface', 'cosface', 'amsoftmax', 'circle'):
            raise NotImplementedError('margin-based teacher classifiers need labels; '
                                      'use a softmax-classifier teacher')
        self.core = core
        self.core.eval()
        for p in self.core.parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        # stay in eval mode even when the surrounding trainer calls .train()
        return super(TeacherModel, self).train(False)

    @torch.no_grad()
    def forward(self, x, cam_label=None, view_label=None):
        core = self.core
        global_feat = core.base(x, cam_label=cam_label, view_label=view_label)
        if core.reduce_feat_dim:
            global_feat = core.fcneck(global_feat)
        feat = core.bottleneck(global_feat)
        score = core.classifier(feat)  # dropout is identity in eval mode
        return score, global_feat


def build_teacher(cfg, num_classes, camera_num, view_num):
    """Build the frozen teacher declared in cfg.DISTILL from its own config."""
    tcfg = _defaults.clone()
    tcfg.defrost()  # the global config may already be frozen; the clone inherits that
    tcfg.merge_from_file(cfg.DISTILL.TEACHER_CONFIG)
    # never load the self-supervised checkpoint; the trained weights follow
    tcfg.MODEL.PRETRAIN_CHOICE = 'none'
    tcfg.MODEL.DIST_TRAIN = False
    if tcfg.MODEL.JPM:
        raise NotImplementedError('JPM teachers are not supported')
    tcfg.freeze()

    core = make_model(tcfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    weight_path = resolve_teacher_weight(cfg.DISTILL.TEACHER_WEIGHT)
    state = torch.load(weight_path, map_location='cpu', weights_only=False)
    result = core.load_state_dict(state, strict=False)
    if result.missing_keys:
        raise RuntimeError('teacher checkpoint is missing {} keys, e.g. {}'.format(
            len(result.missing_keys), result.missing_keys[:3]))
    print('Loaded teacher weights from {} (unexpected keys: {})'.format(
        weight_path, len(result.unexpected_keys)))
    return TeacherModel(core)
