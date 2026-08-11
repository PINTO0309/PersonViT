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
        new_rows, new_pids = [], []
        for key in {(int(p), int(c)) for p, c in zip(pids, camids)}:
            mask = (pids == key[0]) & (camids == key[1])
            mean = F.normalize(feats[mask].mean(dim=0), dim=0)
            index = self.keys.get(key)
            if index is None:
                self.keys[key] = (0 if self.proxies is None else
                                  self.proxies.shape[0]) + len(new_rows)
                new_rows.append(mean)
                new_pids.append(key[0])
            else:
                self.proxies[index] = F.normalize(
                    self.momentum * self.proxies[index]
                    + (1.0 - self.momentum) * mean, dim=0)
        if new_rows:
            rows = torch.stack(new_rows)
            pids_t = torch.as_tensor(new_pids, device=feats.device)
            if self.proxies is None:
                self.proxies, self.proxy_pids = rows, pids_t
            else:
                self.proxies = torch.cat([self.proxies, rows])
                self.proxy_pids = torch.cat([self.proxy_pids, pids_t])

    def forward(self, feats, pids, camids):
        feats = F.normalize(feats.float(), dim=1)
        self._update(feats.detach(), pids, camids)

        sims = feats @ self.proxies.t() / self.tau  # [B, M]
        same_pid = pids.unsqueeze(1) == self.proxy_pids.unsqueeze(0)
        own_index = torch.as_tensor(
            [self.keys[(int(p), int(c))] for p, c in zip(pids, camids)],
            device=feats.device)
        pos_mask = same_pid.clone()
        pos_mask.scatter_(1, own_index.unsqueeze(1), False)  # other cameras only

        neg_sims = sims.masked_fill(same_pid, float('-inf'))
        k = min(self.hard_negatives, neg_sims.shape[1])
        hard_neg = neg_sims.topk(k, dim=1).values  # -inf rows are ignored by logsumexp
        neg_logsum = torch.logsumexp(hard_neg, dim=1)  # [B]

        # -log( exp(s_p) / (exp(s_p) + sum_hard_neg) ), averaged over positives
        losses, count = [], 0
        pos_sims = sims.masked_fill(~pos_mask, float('-inf'))
        for i in range(feats.shape[0]):
            positives = pos_sims[i][torch.isfinite(pos_sims[i])]
            if positives.numel() == 0:
                continue
            denom = torch.logaddexp(positives, neg_logsum[i])
            losses.append((denom - positives).mean())
            count += 1
        if count == 0:
            return feats.new_zeros(())
        return torch.stack(losses).mean()
