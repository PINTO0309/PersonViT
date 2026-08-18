import torch
import torch.nn as nn
import torch.nn.functional as F


def relational_loss(student_feat, teacher_feat):
    """Similarity-preserving loss between batch cosine-similarity matrices
    (the rel-KD component, dimension-agnostic; used by DistillLoss and by
    the camera-proxy mimicry branch)."""
    s = F.normalize(student_feat.float(), dim=1)
    t = F.normalize(teacher_feat.float(), dim=1)
    return F.mse_loss(s @ s.t(), t @ t.t())


class DistillLoss(nn.Module):
    """Knowledge-distillation losses for ReID, added on top of the base
    softmax + triplet objective.

    All default components are backbone-agnostic so the student architecture
    can be swapped freely (ViT-S, future lightweight CNNs, ...):

    - logit KD: temperature-scaled KL divergence between classifier logits.
      Teacher and student share the identity space of the same training set,
      so this works for any embedding dimension.
    - relational KD (similarity-preserving): MSE between the batch cosine
      similarity matrices of student and teacher embeddings. Distills the
      metric structure that retrieval actually uses, independent of the
      embedding dimensions on both sides.
    - embedding KD (optional): cosine distance between student and teacher
      embeddings. With mismatched dimensions a loss-only linear projector
      (student -> teacher space, FitNets-style hint) must be supplied via
      the forward `projector` argument; the projector lives on the student
      model (`embed_proj`, built from DISTILL.EMBED_PROJ_DIM) so it joins
      the optimizer/checkpoint flow, and is dropped at export. Unlike the
      batch-local relational loss this transfers each sample's absolute
      position in the teacher space.

    The module itself stays stateless (no trainable parameters), which keeps
    the optimizer, LR schedule and checkpoint_last.pth resume format
    unchanged; the projector's state belongs to the model.
    """

    def __init__(self, logit_weight=1.0, rel_weight=30.0, embed_weight=0.0,
                 temperature=4.0, hint_weight=0.0, hint_mode='global'):
        super(DistillLoss, self).__init__()
        self.logit_weight = logit_weight
        self.rel_weight = rel_weight
        self.embed_weight = embed_weight
        self.temperature = temperature
        self.hint_weight = hint_weight
        self.hint_mode = hint_mode

    def forward(self, student_score, student_feat, teacher_score, teacher_feat,
                projector=None, hint_feat=None, hint_projector=None,
                hint_target=None):
        if isinstance(student_score, list):
            student_score = student_score[0]
        if isinstance(student_feat, list):
            student_feat = student_feat[0]

        loss = student_feat.new_zeros(())

        if self.logit_weight > 0:
            t = self.temperature
            kd = F.kl_div(F.log_softmax(student_score.float() / t, dim=1),
                          F.softmax(teacher_score.float() / t, dim=1),
                          reduction='batchmean') * (t * t)
            loss = loss + self.logit_weight * kd

        if self.rel_weight > 0:
            loss = loss + self.rel_weight * relational_loss(student_feat,
                                                            teacher_feat)

        if self.embed_weight > 0:
            embed_feat = (projector(student_feat) if projector is not None
                          else student_feat)
            if embed_feat.shape[1] != teacher_feat.shape[1]:
                raise ValueError(
                    'DISTILL.EMBED_WEIGHT requires matching embedding dims, '
                    'got student {} vs teacher {}; set DISTILL.EMBED_PROJ_DIM '
                    'to the teacher dim to train a loss-only projector'.format(
                        embed_feat.shape[1], teacher_feat.shape[1]))
            emb = (1.0 - F.cosine_similarity(embed_feat.float(),
                                             teacher_feat.float(), dim=1)).mean()
            loss = loss + self.embed_weight * emb

        if self.hint_weight > 0:
            if hint_feat is None or hint_projector is None:
                raise ValueError('DISTILL.HINT_WEIGHT > 0 but the backbone '
                                 'provided no hint feature/projector')
            if self.hint_mode == 'spatial':
                # per-position cosine between the projected conv4 map and the
                # teacher's intermediate token map (grids match at stride 16)
                if hint_target is None:
                    raise ValueError('spatial hint needs the teacher token map')
                cos = F.cosine_similarity(hint_projector(hint_feat).float(),
                                          hint_target.float(), dim=1)
                hint = (1.0 - cos).mean()
            else:
                pooled = hint_feat.mean(dim=(2, 3))
                hint = (1.0 - F.cosine_similarity(hint_projector(pooled).float(),
                                                  teacher_feat.float(), dim=1)).mean()
            loss = loss + self.hint_weight * hint

        return loss
