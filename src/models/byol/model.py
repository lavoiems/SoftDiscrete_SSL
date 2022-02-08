import torch
from functools import partial
from torch import nn


class Encoder(nn.Module):
    def __init__(self, h_dim, hidden_size, data_norm=None,
                 data_channels=1, last_enc_bn=False, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.data_norm = data_norm or (0., 1.)
        self.out_dim = hidden_size
        if last_enc_bn:
            last_bn = nn.BatchNorm2d(hidden_size)
        else:
            last_bn = nn.Identity()

        self.encoder = nn.Sequential(
            nn.Conv2d(data_channels, h_dim, 3, 1, 1),
            nn.BatchNorm2d(h_dim),
            nn.ReLU(inplace=True),  # 3

            nn.Conv2d(h_dim, 2*h_dim, 3, 1, 1),
            nn.AdaptiveAvgPool2d((16, 16)),  # 5

            nn.Conv2d(2*h_dim, 2*h_dim, 3, 1, 1),
            nn.BatchNorm2d(2*h_dim),
            nn.ReLU(inplace=True),  # 8

            nn.Conv2d(2*h_dim, 2*h_dim, 3, 1, 1),
            nn.AdaptiveAvgPool2d((4, 4)),  # 10

            nn.Conv2d(2*h_dim, 2*h_dim, 3, 1, 1),
            nn.BatchNorm2d(2*h_dim),
            nn.ReLU(inplace=True),  # 13

            nn.Conv2d(2*h_dim, hidden_size, 3, 1, 1),
            last_bn,
            nn.AdaptiveAvgPool2d((1, 1)),  # 15
            )

    def forward(self, x):
        x = (x - self.data_norm[0]) / self.data_norm[1]
        x = self.encoder(x).flatten(1)
        return x


def encoder(encoder_type, **kwargs):
    return encoders[encoder_type](**kwargs)


def MLP(in_dim, out_dim, h_dim=1024):
    net = nn.Sequential(
        nn.Linear(in_dim, h_dim, bias=True),
        nn.BatchNorm1d(h_dim),
        nn.ReLU(inplace=True),
        nn.Linear(h_dim, out_dim, bias=False))
    return net


heads = {
    'Identity': nn.Identity,
    'Linear': partial(nn.Linear, bias=False),
    'MLP': MLP,
}


def batch_norm(out_dim, net, **kwargs):
    return nn.BatchNorm1d(out_dim), net


def identity(net, **kwargs):
    return nn.Identity(), net


norms = {
    'Identity': identity,
    'BatchNorm': batch_norm,
}


def head(repr_dim, hidden_size, head_type,
         norm_name='Identity', last_bn_affine=True,
         **kwargs):
    net = heads[head_type](repr_dim, hidden_size)
    norm, net = norms[norm_name](out_dim=hidden_size, net=net, **kwargs)
    return nn.Sequential(net, norm)


encoders = {
    'CNN': Encoder,
}
