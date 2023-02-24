import numpy as np


def get_lr_schedule(hp):
    if hp["lr_schedule"] == "constant_lr":
        lrs = lambda epoch, current_lr: constant_lr(
            epoch,
            current_lr,
            lr0=hp["lr0"]
        )
    elif hp["lr_schedule"] == "range_lr":
        lrs = lambda epoch, current_lr: range_lr(
            epoch,
            current_lr,
            min_lr=hp["min_lr"],
            max_lr=hp["max_lr"],
            period=["period"],
        )
    elif hp["lr_schedule"] == "cosine_restarts_lr":
        lrs = lambda epoch, current_lr: cosine_restarts_lr(
            epoch,
            current_lr,
            min_lr=hp["min_lr"],
            max_lr=hp["max_lr"],
            period=["period"],
        )
    elif hp["lr_schedule"] == "exponential_decay_lr":
        lrs = lambda epoch, current_lr: exponential_decay_lr(
            epoch,
            current_lr,
            lr0=hp["lr0"],
            decay_steps=hp["decay_steps"],
            decay_rate=hp["decay_rate"],
        )
    elif hp["lr_schedule"] == "exponential_decay_lr":
        lrs = lambda epoch, current_lr: exponential_decay_lr(
            epoch,
            current_lr,
            lr0=hp["lr0"],
            decay_steps=hp["decay_steps"],
            decay_rate=hp["decay_rate"],
        )
    elif hp["lr_schedule"] == "vaswani_lr":
        lrs = lambda epoch, current_lr: vaswani_lr(
            epoch,
            current_lr,
            lr0=hp["lr0"],
            warmup=hp["warmup"],
        )

    return lrs


def constant_lr(epoch, current_lr, lr0=None):
    return lr0


def range_lr(epoch, current_lr, min_lr=None, max_lr=None, period=None):
    slope = (max_lr - min_lr) / period
    return slope * epoch + min_lr


def cosine_restarts_lr(epoch, current_lr, min_lr=None, max_lr=None, period=None):
    decay = 1 + np.cos(np.pi * (epoch % period) / period)
    return min_lr + (1/2) * (max_lr - min_lr) * decay


def exponential_decay_lr(epoch, current_lr, lr0=None, decay_steps=None, decay_rate=None):
    return lr0 * decay_rate ** (epoch / decay_steps)


def vaswani_lr(epoch, current_lr, lr0=None, warmup=None):
    epoch += 1
    return lr0 * np.min(epoch**-0.5, epoch * warmup**-1.5)
