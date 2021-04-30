""" LR schedulers """

import torch
import json


class LRScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Base class for lr schedulers."""

    def __init__(self, optimizer):
        super().__init__(optimizer)

    def get_lr(self):
        raise NotImplementedError

    def save(self, path):
        """Save the state dict as ...."""
        state_dict = self.state_dict()
        with open(path, 'w') as f:
            json.dump(state_dict, f, indent=2)

    def load(self, path):
        """Load scheduler from a path."""
        with open(path, 'r') as f:
            state_dict = json.load(f)
        self.load_state_dict(state_dict)


class WarmupDecay(LRScheduler):
    """ Learning rate decay with warmup steps as in Attention is All You Need."""

    def __init__(self, optimizer, warmup_steps, d_model, lr_scale = 1.0):
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.lr_scale = lr_scale
        self.n_lrs = len(optimizer.param_groups)
        super().__init__(optimizer)

    def get_lr(self):
        t = self._step_count
        lr = (self.d_model ** -0.5) * min(t ** -0.5, t * (self.warmup_steps ** -1.5))
        return [self.lr_scale * lr for _ in range(self.n_lrs)]

