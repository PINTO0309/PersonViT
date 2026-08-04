import random
from collections import defaultdict

import numpy as np
from torch.utils.data.sampler import Sampler


class DomainBalancedIdentitySampler(Sampler):
    """PK sampler that keeps every mini batch mixed across domains.

    Each batch contains P = batch_size // num_instances identities. For every
    identity slot a domain is drawn first, with probability proportional to
    (number of identities of the domain) ** alpha, then the next identity is
    popped from that domain's shuffled queue (reshuffled when exhausted) and
    K = num_instances images of the identity are drawn.

    alpha = 1 reproduces plain proportional sampling (large domains dominate),
    alpha = 0 samples all domains uniformly (small domains are heavily
    repeated). The default 0.5 makes small occluded domains appear in nearly
    every batch without letting them dominate.

    Args:
    - data_source (list): list of (img_path, pid, camid, domain).
    - batch_size (int): number of images per batch.
    - num_instances (int): number of images per identity in a batch.
    - alpha (float): domain smoothing exponent in [0, 1].
    """

    def __init__(self, data_source, batch_size, num_instances, alpha=0.5):
        if batch_size % num_instances != 0:
            raise ValueError('batch_size ({}) must be divisible by '
                             'num_instances ({})'.format(batch_size, num_instances))
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // num_instances

        self.index_dic = defaultdict(list)
        pid_domain = {}
        for index, (_, pid, _, domain) in enumerate(data_source):
            self.index_dic[pid].append(index)
            pid_domain[pid] = domain

        self.domain_pids = defaultdict(list)
        for pid, dom in pid_domain.items():
            self.domain_pids[dom].append(pid)
        self.domains = sorted(self.domain_pids)
        for dom in self.domains:
            self.domain_pids[dom].sort()

        weights = np.array([len(self.domain_pids[d]) for d in self.domains],
                           dtype=np.float64) ** alpha
        self.domain_weights = (weights / weights.sum()).tolist()

        self.num_batches = len(data_source) // batch_size
        self._queues = {d: [] for d in self.domains}

    def _next_pid(self, dom, exclude):
        pid = None
        for _ in range(10):
            if not self._queues[dom]:
                queue = self.domain_pids[dom][:]
                random.shuffle(queue)
                self._queues[dom] = queue
            pid = self._queues[dom].pop()
            if pid not in exclude:
                return pid
        return pid

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen = set()
            batch = []
            doms = random.choices(self.domains, weights=self.domain_weights,
                                  k=self.num_pids_per_batch)
            for dom in doms:
                pid = self._next_pid(dom, chosen)
                chosen.add(pid)
                idxs = self.index_dic[pid]
                replace = len(idxs) < self.num_instances
                batch.extend(np.random.choice(idxs, size=self.num_instances,
                                              replace=replace).tolist())
            for idx in batch:
                yield idx

    def __len__(self):
        return self.num_batches * self.batch_size
