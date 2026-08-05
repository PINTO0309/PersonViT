import torch
import torch.nn as nn
import torch.nn.functional as F


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
      embeddings. Only valid when both embedding dimensions match, therefore
      disabled by default.

    The module is stateless (no trainable parameters), which keeps the
    optimizer, LR schedule and checkpoint_last.pth resume format unchanged.
    """

    def __init__(self, logit_weight=1.0, rel_weight=30.0, embed_weight=0.0,
                 temperature=4.0):
        super(DistillLoss, self).__init__()
        self.logit_weight = logit_weight
        self.rel_weight = rel_weight
        self.embed_weight = embed_weight
        self.temperature = temperature

    def forward(self, student_score, student_feat, teacher_score, teacher_feat):
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
            s = F.normalize(student_feat.float(), dim=1)
            te = F.normalize(teacher_feat.float(), dim=1)
            rel = F.mse_loss(s @ s.t(), te @ te.t())
            loss = loss + self.rel_weight * rel

        if self.embed_weight > 0:
            if student_feat.shape[1] != teacher_feat.shape[1]:
                raise ValueError(
                    'DISTILL.EMBED_WEIGHT requires matching embedding dims, '
                    'got student {} vs teacher {}; use the relational/logit '
                    'losses for cross-dimension distillation'.format(
                        student_feat.shape[1], teacher_feat.shape[1]))
            emb = (1.0 - F.cosine_similarity(student_feat.float(),
                                             teacher_feat.float(), dim=1)).mean()
            loss = loss + self.embed_weight * emb

        return loss
