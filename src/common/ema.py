from copy import deepcopy

import torch
from torch import nn

from common.models import reset_parameters


def reset_ema(which: nn.Module, reset_ema_parameters: bool = False):
    ema_model = deepcopy(which).requires_grad_(False)
    if reset_ema_parameters:
        ema_model.apply(reset_parameters)
    ema_model.train()
    return ema_model


def copy_ema(target: nn.Module, source: nn.Module):
    target.load_state_dict(source.state_dict()) # Need to do this because weight_norm is bad behaved with deepcopy: https://github.com/pytorch/pytorch/issues/28594.
    target.requires_grad_(False)
    target.train()
    return target


@torch.no_grad()
def update_ema(source: nn.Module, target: nn.Module, alpha: float = 0.998):
    r"""Apply the exponential moving average update to a target model.

    This function updates the parameters of `target` network using an
    exponential moving average of the parameters of a `source` network.
    In other words it implements:

    .. math::
       \tilde{\theta}_{k+1} = \alpha * \tilde{\theta}_k + (1 - \alpha) * \theta_{k+1}

    where :math:`\theta` are the parameters of `source` network and
    :math:`\tilde{\theta}` the parameters of `target` network.

    We use this to obtain a better and more stable generator model in GANs.

    .. info::
       In PyTorch, you can get access to a dictionary containing entries
       of (names, module's parameters) by invoking ``model.named_parameters()``.

    Args:
        source: A model whose exponential moving average parameters across
           training time we want to have.
        target: A cloned version of `source` which holds the exponential
           moving averaged parameters.
        alpha: The weight by which we apply exponential moving average,
           :math:`\alpha` in the formula above.

    Returns:
        None (all updates in the `target` model must happen inplace)

    """
    param_dict_src = dict(source.named_parameters())
    for p_name, p_target in target.named_parameters():
        p_source = param_dict_src[p_name]
        p_target.add_(p_source.sub(p_target).mul(1. - alpha))
