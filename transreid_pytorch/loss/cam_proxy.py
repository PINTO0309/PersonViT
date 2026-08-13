"""Camera-aware proxy contrastive loss (PPLR/CAP style, arXiv:2203.14675).

Keeps one momentum feature proxy per (identity, camera) pair. Each sample's
positives are the proxies of the SAME identity under OTHER cameras; the
negatives are the hardest proxies of other identities. The InfoNCE over
those proxies directly shrinks cross-camera intra-class variance — the core
invariance ReID needs — which plain ID/triplet losses only target
indirectly through batch sampling.

The proxy bank is training state only: it starts empty, fills as (id, cam)
pairs are first seen, and is intentionally not checkpointed (it rebuilds
within one epoch after a resume). Samples whose identity has no other-camera
proxy yet contribute nothing, so the loss ramps up smoothly during the
first epoch.

Implementation note: bank update and loss are fully vectorized (index_add
scatter means, masked matrix InfoNCE). The first version looped in Python
per key and per sample, which cost ~44% of ViT-B training throughput.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraProxyLoss(nn.Module):
    def __init__(self, tau=0.07, momentum=0.2, hard_negatives=50):
        super(CameraProxyLoss, self).__init__()
        self.tau = tau
        self.momentum = momentum  # retain factor of the old proxy
        self.hard_negatives = hard_negatives
        self.keys = {}  # (pid, camid) -> row index
        self.proxies = None  # [M, D], L2-normalized
        self.proxy_pids = None  # [M]

    @torch.no_grad()
    def _update(self, feats, pids, camids):
        """Momentum-update every proxy touched by this batch (vectorized)."""
        old_size = 0 if self.proxies is None else self.proxies.shape[0]
        row_of_sample = []
        new_pids = []
        for p, c in zip(pids.tolist(), camids.tolist()):  # ints only, no GPU work
            key = (p, c)
            index = self.keys.get(key)
            if index is None:
                index = old_size + len(new_pids)
                self.keys[key] = index
                new_pids.append(p)
            row_of_sample.append(index)

        size = old_size + len(new_pids)
        if new_pids:
            zeros = feats.new_zeros(len(new_pids), feats.shape[1])
            self.proxies = (zeros if self.proxies is None
                            else torch.cat([self.proxies, zeros]))
            pids_t = torch.as_tensor(new_pids, device=feats.device)
            self.proxy_pids = (pids_t if self.proxy_pids is None
                               else torch.cat([self.proxy_pids, pids_t]))

        row_index = torch.as_tensor(row_of_sample, device=feats.device)
        sums = feats.new_zeros(size, feats.shape[1])
        sums.index_add_(0, row_index, feats)
        counts = feats.new_zeros(size)
        counts.index_add_(0, row_index, torch.ones_like(row_index, dtype=feats.dtype))
        touched = counts > 0
        means = F.normalize(sums[touched] / counts[touched].unsqueeze(1), dim=1)

        is_new = (torch.arange(size, device=feats.device) >= old_size)[touched]
        blended = F.normalize(
            self.momentum * self.proxies[touched]
            + (1.0 - self.momentum) * means, dim=1)
        self.proxies[touched] = torch.where(is_new.unsqueeze(1), means, blended)

    def forward(self, feats, pids, camids):
        feats = F.normalize(feats.float(), dim=1)
        self._update(feats.detach(), pids, camids)

        sims = feats @ self.proxies.t() / self.tau  # [B, M]
        same_pid = pids.unsqueeze(1) == self.proxy_pids.unsqueeze(0)
        own_index = torch.as_tensor(
            [self.keys[(int(p), int(c))] for p, c in zip(pids.tolist(), camids.tolist())],
            device=feats.device)
        pos_mask = same_pid.clone()
        pos_mask.scatter_(1, own_index.unsqueeze(1), False)  # other cameras only

        neg_sims = sims.masked_fill(same_pid, float('-inf'))
        k = min(self.hard_negatives, neg_sims.shape[1])
        hard_neg = neg_sims.topk(k, dim=1).values  # -inf rows are ignored by logsumexp
        neg_logsum = torch.logsumexp(hard_neg, dim=1)  # [B]

        # -log( exp(s_p) / (exp(s_p) + sum_hard_neg) ) for every positive,
        # averaged per sample over its positives, then over samples that
        # have at least one other-camera positive — all without host syncs
        per_positive = torch.logaddexp(sims, neg_logsum.unsqueeze(1)) - sims  # [B, M]
        positive_counts = pos_mask.sum(dim=1)
        per_sample = (per_positive * pos_mask).sum(dim=1) / positive_counts.clamp(min=1)
        valid = (positive_counts > 0).to(per_sample.dtype)
        return (per_sample * valid).sum() / valid.sum().clamp(min=1.0)
