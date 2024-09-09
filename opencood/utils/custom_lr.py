import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_lr, max_lr, min_lr, warmup_epoch, max_epoch, last_epoch=-1):
        self.warmup_lr = warmup_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epoch = warmup_epoch
        self.max_epoch = max_epoch
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epoch:
            # Warmup phase: linearly scale up LR from warmup_lr to max_lr
            lr = self.warmup_lr + (self.max_lr - self.warmup_lr) * (self.last_epoch / self.warmup_epoch)
        else:
            # Cosine annealing phase
            cosine_epoch = self.last_epoch - self.warmup_epoch
            cosine_max_epoch = self.max_epoch - self.warmup_epoch
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * cosine_epoch / cosine_max_epoch))
        # Apply the calculated lr to all parameter groups
        if 'multiplier' in self.optimizer.param_groups[0]:
            lrs = []
            for param_groups in self.optimizer.param_groups:
                lr_multipliered = lr * param_groups['multiplier']
                lrs.append(lr_multipliered)
            return lrs
        else:
            return [lr for _ in self.base_lrs]