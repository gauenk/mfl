"""

Poisoning for the trimmed mean attack

"""

# -- linalg --
import numpy as np
import torch as th
from einops import rearrange,repeat

class TrimmedMeanAttack():

    def __init__(self,mode,total,percent):
        self.mode = mode
        self.total = total
        self.percent = percent
        self.nattack = int(total * percent)
        order = th.randperm(total)
        self.ids_attack = order[:self.nattack]
        self.ids_safe = order[self.nattack:]
        assert mode in ["full","partial"]

    def _compute_grads(grads):
        if mode == "full":
            return self._compute_grads_full(grads)
        elif mode == "partial":
            return self._compute_grads_partial(grads_attack)

    def _compute_grads_full(grads):
        direction = th.mean(grads,1)
        gmax = th.max(grads,1)
        gmin = th.min(grads,1)
        grads = th.zeros_like(grads)
        return grads_new

    def _compute_grads_partial(grads):
        means = th.mean(grads,1)
        std = th.std(grads,1)
        grads_new = th.zeros_like(grads)

        low = mean - 4 * std
        upp = mean - 3 * std
        grads_new.uniform_(low,upp)
        return grads_new

    def __call__(self,_grads):
        grads = _grads.clone()
        grads_new = self._compute_grads(grads)
        grads[self.id_attack] = grads_new
        return grads

