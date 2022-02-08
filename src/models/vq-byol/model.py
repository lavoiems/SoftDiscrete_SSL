import torch
from functools import partial
from torch import nn
from torchvision import models


class Encoder(nn.Module):
    def __init__(self, h_dim, hidden_size, data_norm=None,
                 data_channels=1, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.data_norm = data_norm or (0., 1.)
        self.out_dim = hidden_size

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
            nn.AdaptiveAvgPool2d((1, 1)),  # 15
            )

    def forward(self, x):
        x = (x - self.data_norm[0]) / self.data_norm[1]
        x = self.encoder(x).flatten(1)
        return x
        #  return self.bn(x), F.normalize(x, dim=-1, p=2)


def encoder(encoder_type, **kwargs):
    return encoders[encoder_type](**kwargs)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, h_dim: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h_dim, bias=True),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(inplace=True),
            nn.Linear(h_dim, out_dim, bias=False))

    def forward(self, x):
        return self.net(x.flatten(1))


heads = {
    'Identity': nn.Identity,
    'Linear': partial(nn.Linear, bias=False),
    'MLP': MLP,
}


def head(repr_dim, hidden_size, head_type,
         last_bn=False, last_bn_affine=True,
         **kwargs):
    net = heads[head_type](repr_dim, hidden_size)
    if last_bn is True:
        bn = nn.BatchNorm1d(hidden_size, affine=last_bn_affine)
        net = nn.Sequential(net, bn)
    return net


encoders = {
    'CNN': Encoder,
}
