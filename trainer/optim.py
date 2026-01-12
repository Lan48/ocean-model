import math
from typing import Optional, Union
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from transformers.trainer_utils import SchedulerType
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup
)


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, base_lr, min_lr, last_epoch=-1
):

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        
        return max(0.0, 1 - (1 - min_lr / base_lr) * progress)

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, base_lr: float, min_lr: float, num_cycles: float = 0.5, last_epoch: int = -1
):

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        ratio = min_lr / base_lr
        return max(0.0, ratio + 0.5 * (1 - ratio) * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


TYPE_TO_SCHEDULER_FUNCTION = {
    SchedulerType.LINEAR: get_linear_schedule_with_warmup,
    SchedulerType.COSINE: get_cosine_schedule_with_warmup,
    SchedulerType.CONSTANT: get_constant_schedule,
    SchedulerType.CONSTANT_WITH_WARMUP: get_constant_schedule_with_warmup,
}


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    base_lr: float,
    min_lr: Optional[int] = 0.0,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
):
    if name == 'onecycle':
        return OneCycleLR(
            optimizer,
            max_lr=base_lr,
            total_steps=num_training_steps,
        )
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, base_lr=base_lr, min_lr=min_lr)