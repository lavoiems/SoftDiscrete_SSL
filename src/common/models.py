from typing import (Any, Mapping, Optional, Tuple, Text)

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import (Categorical, Normal, ContinuousBernoulli)
from sklearn import metrics

Vector = Tuple[int]


def reset_parameters(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()


def get_nonlinearity(nonlinearity: Optional[Text]='ReLU', inplace: bool=True):
    if nonlinearity == 'ReLU':
        nonlin = nn.ReLU
    elif nonlinearity == 'LeakyReLU':
        nonlin = nn.LeakyReLU
    elif nonlinearity == 'RReLU':
        nonlin = nn.RReLU
    elif nonlinearity is None:
        nonlin = nn.Identity
    else:
        raise RuntimeError(
            'Not supported nonlinearity for critic: {}'.format(nonlinearity))
    return nonlin(inplace=inplace)


class Normalize(nn.Module):
    def __init__(self, p=2, dim=-1, eps=1e-12):
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim, eps=self.eps)


def get_norm(dimh, kwargs: Optional[Mapping[Text, Any]]=None):
    if kwargs is None:
        return nn.Identity()

    kwargs = dict(kwargs)
    type_ = kwargs.pop('type', 'bn')
    if type_ == 'bn':
        return nn.BatchNorm1d(dimh, **kwargs)
    elif type_ == 'ln':
        return nn.LayerNorm(dimh)
    elif type_.startswith('l'):
        return Normalize(p=int(type_[1]))
    else:
        raise ValueError


def sn_conv2d(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(*args, **kwargs))


def sn_linear(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(*args, **kwargs))


def sn_embedding(*args, **kwargs):
    return nn.utils.spectral_norm(nn.Embedding(*args, **kwargs))


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dimh: Vector=(),
                 nonlinearity: Optional[Text]=None,
                 bias: bool = True,
                 norm: Optional[Tuple[Optional[Mapping[Text, Any]]]]=None,
                 dropout: Optional[Vector]=None, sn=False, init_scale: float=1.):
        super().__init__()
        self.init_scale = init_scale
        self.dim_in = dim_in
        self.dim_out = dim_out
        dimh_ = (self.dim_in, ) + dimh + (self.dim_out, )
        if norm is None:
            norm = (None, ) * len(dimh)
        if dropout is None:
            dropout = (0, ) * len(dimh)
        if len(norm) == len(dimh):
            norm += (None, )
        Linear = sn_linear if sn else nn.Linear

        layers = [None] * (4 * len(dimh) + 2)
        layers[0::4] = [Linear(dimh_[i], dimh_[i+1], bias=bias) for i in range(len(dimh) + 1)]
        layers[1::4] = [get_norm(dim, cfg) for i, (cfg, dim) in enumerate(zip(norm, dimh_[1:]))]
        layers[2::4] = [get_nonlinearity(nonlinearity) for _ in range(len(dimh))]
        layers[3::4] = [(nn.Dropout(p=dout, inplace=False) if dout > 0 else nn.Identity()) for dout in dropout]
        self.encoder = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self, p=1.):
        assert(0. <= p and p <= 1.)
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    mask = torch.bernoulli(torch.ones(m.weight.size(0), device=m.weight.device).fill_(p))
                    new_weight = m.weight.clone()
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(new_weight)
                    bound = np.sqrt(1./fan_in) * self.init_scale
                    nn.init.uniform_(new_weight, -bound, bound)
                    # nn.init.orthogonal_(new_weight)
                    mask_weight = mask.view(-1, *[1,] * len(m.weight.size()[1:]))
                    m.weight.data = (1 - mask_weight) * m.weight.data + mask_weight * new_weight
                    if m.bias is not None:
                        new_bias = m.bias.clone()
                        nn.init.uniform_(new_bias, -bound, bound)
                        mask_bias = mask.view(-1)
                        m.bias.data = (1 - mask_bias) * m.bias.data + mask_bias * new_bias

                elif isinstance(m, nn.BatchNorm2d):
                    m.reset_parameters()

    def forward(self, x):
        return self.encoder(x.flatten(1))


class Classifier(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def score(self, y, y_true, k=1, **kwargs):
        """Return accuracy."""
        y = y.cpu().numpy()
        y_true = y_true.cpu().numpy()
        return metrics.top_k_accuracy_score(y_true, y, k=k, labels=np.arange(y.shape[-1]), **kwargs)

    def predict(self, x):
        y = self(x).probs
        if y.size(-1) == 2:
            y = y[:, 1]
        return y

    def forward(self, x):
        logits = self.model(x)
        return Categorical(logits=logits)


class Regressor(nn.Module):
    def __init__(self, model, std=1.):
        super().__init__()
        self.model = model
        self.register_buffer('std', torch.tensor([std]))

    def score(self, y, y_true, **kwargs):
        """Return R2 score."""
        y = y.cpu().numpy()
        y_true = y_true.cpu().numpy()
        return metrics.r2_score(y_true, y)

    def predict(self, x):
        return self(x).loc

    def forward(self, x):
        mu = self.model(x).squeeze()
        return Normal(mu, self.std)

