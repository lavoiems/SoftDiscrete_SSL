from typing import (NamedTuple, Optional, Text)

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical


class Code(NamedTuple):
    dims: int
    opts: int
    continuous: bool = False

    @property
    def size(self):
        if self.opts > 0:
            return torch.Size((self.dims, self.opts))
        return torch.Size((self.dims, ))

    @property
    def total_dims(self):
        if self.opts > 0:
            return self.dims * self.opts
        return self.dims

    @property
    def options(self):
        if self.opts > 1:
            return self.opts
        return self.opts + 1

    @property
    def num_codes(self):
        assert(self.opts > 0)
        return self.options ** self.dims

    @property
    def language(self):
        assert(self.opts > 0)
        return torch.from_numpy(np.mgrid[self.dims * [slice(0, self.options),]].reshape(self.dims, -1).T)

    @property
    def dataset(self):
        assert(self.opts > 0)
        if self.opts > 1:
            return F.one_hot(self.language, self.opts)
        return self.language

    @property
    def str(self):
        assert(self.opts > 0)
        return [tuple_to_str(x) for x in self.language]

    def __repr__(self):
        return f"message_size: {self.dims}, voc_size: {self.options}"

    def __str__(self):
        return f"<{self.dims}-{self.options}>"

    def to_vector(self, x):
        if self.opts > 0:
            if self.opts > 1:
                return F.one_hot(x, self.opts).float()
            return (2 * x - 1).float()

        return x.float()

    def to_decision(self, x):
        if self.opts > 0:
            if self.opts > 1:
                return x.argmax(-1).float()
            return (x > 0.5).float()

        return x.float()


def tuple_to_str(x):
    return ''.join([(str(int(i)) if int(i) >= 0 else '-') for i in x])


class Continuous(nn.Module):
    def __init__(self, embedder: nn.Module, code: Code, **kwargs):
        super().__init__()
        assert(code.continuous is True)
        self.embedder = embedder
        self.code = code

    def extra_repr(self):
        return repr(self.code)

    def forward(self, query):
        embedding = self.embedder(query)
        logits = embedding.view(-1, *self.code.size)

        return {'logits': logits, 'embedding': embedding, 'representation': logits, 'msg': logits}


class GumbelSoftmax(nn.Module):
    def __init__(self, embedder: nn.Module, code: Code,
                 channel_tau: Optional[float] = None, hard: bool = False,
                 **kwargs):
        super().__init__()
        self.embedder = embedder
        self.code = code
        self.tau = channel_tau or np.sqrt(self.code.options)
        self.hard = hard

    def extra_repr(self):
        return repr(self.code)

    def forward(self, query):
        logits = self.embedder(query)
        logits = logits.view(-1, *self.code.size)
        entropy = ((logits).softmax(-1) *
                   (logits).log_softmax(-1)).sum(-1).neg().mean()
        entropy_tau = ((logits/self.tau).softmax(-1) *
                       (logits/self.tau).log_softmax(-1)).sum(-1).neg().mean()

        gumbels = torch.empty_like(logits).exponential_().log().neg()  # ~Gumbel(0,1)
        gumbels = (logits + gumbels) / self.tau  # ~Gumbel(logits,tau)
        y_soft = gumbels.softmax(-1)
        msg = y_soft.argmax(-1)

        if self.hard:
            # Straight through.
            y_hard = F.one_hot(msg, num_classes=self.code.options).float()
            embedding = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            embedding = y_soft

        embedding = embedding.view(logits.shape[0], -1)

        # This is not equivalent to msg or embedding.
        representation = F.one_hot(logits.argmax(-1), num_classes=self.code.options).half()
        return {'logits': logits, 'embedding': embedding, 'msg': msg,
                'representation': representation,
                "metrics": {"entropy": entropy, "entropy_tau": entropy_tau}}


class Softmax(nn.Module):
    def __init__(self, embedder: nn.Module, code: Code,
                 channel_tau: Optional[float] = None, hard: bool = False,
                 channel_cross: float = 1., channel_dp: float = 0., **kwargs):
        super().__init__()
        self.embedder = embedder
        self.code = code
        self.tau = channel_tau
        self.pi = channel_cross
        self.dp = nn.Dropout(channel_dp)
        self.hard = hard

    def extra_repr(self):
        return repr(self.code)

    def forward(self, query):
        logits = self.embedder(query)
        logits = logits.view(-1, *self.code.size)
        entropy = ((logits).softmax(-1) *
                   (logits).log_softmax(-1)).sum(-1).neg().mean()
        entropy_tau = ((logits/self.tau).softmax(-1) *
                       (logits/self.tau).log_softmax(-1)).sum(-1).neg().mean()

        noise = logits / self.tau  # ~N(logits/tau,1/tau)
        y_soft = noise.softmax(-1)
        msg = y_soft.argmax(-1)

        if self.hard:
            # Straight through.
            y_hard = F.one_hot(msg, num_classes=self.code.options).float()
            embedding = y_hard - y_soft.detach() + y_soft
        else:
            # Reparametrization trick.
            embedding = y_soft

        embedding = embedding.view(logits.shape[0], -1)

        representation = F.one_hot(logits.argmax(-1), num_classes=self.code.options).half()
        return {'logits': logits, 'embedding': embedding, 'msg': msg, 'representation': representation,
                "metrics": {"entropy": entropy, "entropy_tau": entropy_tau}}


class VQ(nn.Module):
    def __init__(self,
                 embedder: nn.Module, code: Code,
                 channel_tau: float = 1., hard: bool = True,
                 vq_dimz: float = 8,
                 vq_beta: float = 0.25, vq_eps: float = 1e-8,
                 vq_update_rule: Text = 'loss',
                 vq_ema: float = 0.95,
                 **kwargs):
        super().__init__()
        self.embedder = embedder
        self.code = code
        assert(vq_dimz is not None)
        self.dimz = vq_dimz
        self.hard = hard
        self.tau = channel_tau
        self.beta = vq_beta
        self.eps = vq_eps

        self.update_rule = vq_update_rule
        self.ema = vq_ema
        self.register_buffer('has_init', torch.tensor(False, dtype=torch.bool))

        self.codebook = nn.Parameter(torch.zeros(code.options, vq_dimz))

        if not bool(self.has_init):
            self.init(code.dims)
            self.has_init.fill_(True)

    def init(self, code_dims):
        torch.nn.init.normal_(self.codebook,
                              mean=0.0, std=np.sqrt(2./(code_dims * self.dimz)))
        # torch.nn.init.uniform_(self.codebook, a=-1.0 / self.code.options, b=1.0 / self.code.options)

    def extra_repr(self):
        return f'options: {self.code.options}, dimz: {self.dimz}, update: {self.update_rule}'

    def latent(self, query):
        if query.dim() == 4:
            query = query.permute(0, 2, 3, 1).contiguous()
        else:
            query = query.view(query.size(0), -1, self.dimz)
        return query

    def score(self, latent):
        latent = latent[..., None, :]  # (B, ..., code.options, dimz)
        scores = - 0.5 * latent.sub(self.codebook).pow(2).sum(-1) / self.tau
        return scores

    @torch.no_grad()
    def ema_codebook_update(self, latent, y, ema=None):
        ema = ema or self.ema
        latent = latent[..., None, :]  # (B, ..., code.options, dimz)
        idx = y[..., None] == torch.arange(self.code.options, device=latent.device)  # (B, ..., code.options)
        batch_dims = tuple(range(idx.dim() - 1))
        new_codebook = latent.masked_fill(idx[..., None], 0.).sum(dim=batch_dims) / idx.sum(dim=batch_dims).clamp(min=1.)[:, None]
        p_y = idx.sum(dim=batch_dims) / idx.sum()
        diff_codebook = new_codebook.sub(self.codebook)
        self.codebook.add_(diff_codebook.mul(1. - ema))
        return diff_codebook.pow(2).sum(-1).max(), p_y

    def encode(self, latent):
        scores = self.score(latent)  # (B, ..., code.options)
        if not self.hard:
            return Categorical(logits=scores)
        select = scores.argmax(-1)  # (B, ...)
        return Categorical(probs=F.one_hot(select, self.code.options).to(dtype=latent.dtype))

    def decode(self, latent, code):
        values = self.codebook[code]  # (B, ..., dimz)
        if not self.hard:
            beta = self.beta**0.5
            values = (1 - beta) * values + beta * latent + torch.randn_like(values) * self.tau**0.5
        return values

    def forward(self, query, channel=None):
        channel = channel or (lambda x: x)
        latent = self.latent(self.embedder(query))  # (B, ..., dimz)

        q_y = channel(self.encode(latent))
        y = q_y.sample() if self.training else q_y.logits.argmax(-1)  # (B, ...)
        quantized = self.decode(latent, y)  # (B, ..., dimz)

        ema_update = self.update_rule == 'ema'
        if self.hard:
            embedding = (latent - latent.detach()) + (quantized.detach() if ema_update else quantized)
        else:
            embedding = quantized

        loss = 0.
        if self.training:
            if self.hard:
                loss = self.beta * quantized.detach().sub(latent).pow(2).sum((-1,-2)).mean()
                if self.update_rule == 'loss':
                    loss = loss + quantized.sub(latent.detach()).pow(2).sum((-1,-2)).mean()
                elif ema_update:
                    self.ema_codebook_update(latent, y)
            else:
                loss = self.beta * quantized.sub(latent).pow(2).sum((-1,-2)).mean()
            loss = 0.5 * loss / self.tau

        if query.dim() == 4:
            embedding = embedding.permute(0, 3, 1, 2).contiguous()

        representation = F.one_hot(y, num_classes=self.code.options).to(dtype=query.dtype)
        py = representation.mean(dim=tuple(range(representation.dim()-1)))
        Hy = - torch.sum(py * torch.log2(py + 1e-10))
        return {'latent': latent, 'embedding': embedding, 'msg': y,
                'representation': representation,
                'metrics': {'Hyx': q_y.entropy().mean() / np.log(2), 'Hy': Hy},
                'loss': loss}
