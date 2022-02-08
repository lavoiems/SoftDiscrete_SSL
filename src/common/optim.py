# -*- coding: utf-8 -*-
r"""
:mod:`optim` -- Optimization utility for training with Pytorch
==============================================================

.. module:: optim
   :platform: Unix
   :synopsis: This one just contains a builder for a torch.optim.Optimizer

"""
from functools import partial
from typing import (Tuple, Text)

import numpy as np
from torch import optim


Vector = Tuple[float]


def init_optimizer(variables, type: Text,
                   lr: float=1e-3,
                   betas: Vector=(0.9, 0.999),
                   wd: float=1e-6,
                   eps: float=1e-8) -> optim.Optimizer:
    Algo = getattr(optim, type)
    momentum = betas[0]
    partial_algo = partial(Algo, variables, lr=lr, weight_decay=wd)
    if 'Adam' in type:
        return partial_algo(betas=betas, eps=eps)
    elif type == 'SGD':
        return partial_algo(momentum=momentum)
    else:
        raise NotImplementedError


def cosine_annealing(eta_min, eta_max, T, t0=0, T_reset=float('inf'), multiplier=0.8):
    t = t0
    while True:
        yield eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos(2 * np.pi * float(t) / T))
        t += 1
        if t > T_reset:
            t = t0
            eta_min = multiplier * eta_min + (1 - multiplier) * eta_max


def exp_annealing(eta_min, eta_max, T, t0=0, T_reset=float('inf'), multiplier=0.8):
    t = t0
    while True:
        yield eta_max + (eta_min - eta_max) * np.exp(- float(t) / T)
        t += 1
        if t > T_reset:
            t = t0
            eta_min = multiplier * eta_min + (1 - multiplier) * eta_max


def linear_annealing(eta_min, eta_max, T, t0=0, T_reset=float('inf'), multiplier=0.8):
    t = t0
    while True:
        yield min(eta_max - ((eta_min - eta_max) / T) * (t - T), eta_max)
        t += 1
        if t > T_reset:
            t = t0
            eta_min = multiplier * eta_min + (1 - multiplier) * eta_max


def cosine_annealing_hard_restarts(T, t0=0, T_warmup=0):
    t = t0
    while True:
        if t < T_warmup:
            yield float(t) / float(max(1, T_warmup))
        else:
            yield max(0.0, 0.5 * (1.0 + np.cos(np.pi * ((float(t) / T) % 1.0))))
        t += 1


def step_annealing(lrs=None, t0=0):
    lrs = lrs or dict()
    t = t0
    lr = 1.
    while True:
        lr = lrs.get(t, lr)
        yield lr
        t += 1


def constant(eta):
    while True:
        yield eta

