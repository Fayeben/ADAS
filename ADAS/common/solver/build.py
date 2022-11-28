"""Build optimizers and schedulers"""
import warnings
import torch
from .lr_scheduler import ClipLR

import torch.optim as optim

def build_optimizer(cfg, model):
    name = cfg.OPTIMIZER.TYPE
    if name == '':
        warnings.warn('No optimizer is built.')
        return None
    elif hasattr(torch.optim, name):
        return getattr(torch.optim, name)(
            model.parameters(),
            lr=cfg.OPTIMIZER.BASE_LR,
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY,
            **cfg.OPTIMIZER.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of optimizer.')


def build_scheduler(cfg, optimizer):
    name = cfg.SCHEDULER.TYPE
    if name == '':
        warnings.warn('No scheduler is built.')
        return None
    elif hasattr(torch.optim.lr_scheduler, name):
        scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer,
            **cfg.SCHEDULER.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of scheduler.')

    # clip learning rate
    if cfg.SCHEDULER.CLIP_LR > 0.0:
        print('Learning rate is clipped to {}'.format(cfg.SCHEDULER.CLIP_LR))
        scheduler = ClipLR(scheduler, min_lr=cfg.SCHEDULER.CLIP_LR)

    return scheduler


def build_optimizer_discriminator(model):
    # if optim_cfg.OPTIMIZER == 'adam':
    #     optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
    # elif optim_cfg.OPTIMIZER == 'sgd':
    #     optimizer = optim.SGD(
    #         model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
    #         momentum=optim_cfg.MOMENTUM
    #     )
    # elif optim_cfg.OPTIMIZER == 'adam_onecycle':
    #     def children(m: nn.Module):
    #         return list(m.children())

    #     def num_children(m: nn.Module) -> int:
    #         return len(children(m))

    #     flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
    #     get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

    #     optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
    #     optimizer = OptimWrapper.create(
    #         optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
    #     )
    # else:
    #     raise NotImplementedError
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.)
    return optimizer

def build_scheduler_discriminator(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler