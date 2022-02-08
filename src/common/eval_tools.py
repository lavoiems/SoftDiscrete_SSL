import copy

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn.linear_model import RidgeClassifier, Ridge, LogisticRegression
from sklearn import metrics
from tqdm import tqdm

from common import util
from common.optim import init_optimizer
from common.models import MLP, Classifier, Regressor


@torch.enable_grad()
def discrepancy_test(X, Y=None, opt_steps=50000, tau=1., N_valid=1000, N_test=4000,
                     val_every=500, patience=5, lr=1e-3, wd=1e-2, batch_size=64):
    X = X.flatten(1)
    x_set = torch.utils.data.TensorDataset(X)
    lengths = [N_test, N_valid, len(x_set) - N_test - N_valid]
    x_test, x_valid, x_train = torch.utils.data.random_split(x_set, lengths,
                                                             generator=torch.Generator().manual_seed(103))
    x_train_loader = torch.utils.data.DataLoader(x_train, batch_size=batch_size, shuffle=True)
    x_train_sampler = util.infinite_sampler(x_train_loader)
    x_valid_loader = torch.utils.data.DataLoader(x_valid, batch_size=batch_size, shuffle=True)
    x_test_loader = torch.utils.data.DataLoader(x_test, batch_size=batch_size, shuffle=True)
    x_test_sampler = iter(x_test_loader)

    if Y is not None:
        Y = Y.flatten(1)
        y_set = torch.utils.data.TensorDataset(Y)
        lengths = [N_test, N_valid, len(y_set) - N_test - N_valid]
        y_test, y_valid, y_train = torch.utils.data.random_split(y_set, lengths,
                                                                 generator=torch.Generator().manual_seed(104))
        y_train_loader = torch.utils.data.DataLoader(y_train, batch_size=batch_size, shuffle=True)
        y_train_sampler = util.infinite_sampler(y_train_loader)
        y_valid_loader = torch.utils.data.DataLoader(y_valid, batch_size=batch_size, shuffle=True)
        y_test_loader = torch.utils.data.DataLoader(y_test, batch_size=batch_size, shuffle=True)
        y_test_sampler = iter(y_test_loader)

    critic = MLP(X.size(-1), 1, dimh=(2048, 1024, 512), nonlinearity='LeakyReLU')
    critic.to(device=X.device)
    optimizer = init_optimizer(critic.parameters(), 'AdamW', lr=lr, wd=wd)

    train_acc_meter = util.running_average_meter(momentum=0.8)
    best_val_acc = float('-inf')
    ctr = 0

    @util.train_ctx(critic, optimizer)
    def train_step(x, y):
        x = F.normalize(x, p=2, dim=-1) / tau
        if y is None:
            y = torch.randn_like(x)
        y = F.normalize(y, p=2, dim=-1) / tau

        cx, cy = torch.chunk(critic(torch.cat([x, y])), 2)
        loss = F.logsigmoid(cx).mean() + F.logsigmoid(-cy).mean()
        train_acc_meter(((cx > 0.).sum() + (cy < 0.).sum()) / (2. * batch_size))
        return dict(loss=loss.neg())

    for i in range(opt_steps):
        train_step(next(x_train_sampler)[0], next(y_train_sampler)[0] if Y is not None else None)

        with torch.no_grad():
            if i % val_every == 0:
                critic.eval()
                y_iter = iter(y_valid_loader) if Y is not None else None

                val_acc = 0.
                for x in iter(x_valid_loader):
                    x = F.normalize(x[0], p=2, dim=-1)
                    if Y is None:
                        y = torch.randn_like(x)
                    else:
                        y = next(y_iter)[0]
                    y = F.normalize(y, p=2, dim=-1)

                    cx, cy = torch.chunk(critic(torch.cat([x, y])), 2)
                    val_acc = val_acc + ((cx > 0.).sum() + (cy < 0.).sum()) / (2 * N_valid)

                if val_acc > best_val_acc:
                    ctr = 0
                    best_val_acc = val_acc
                    best_critic = copy.deepcopy(critic)
                else:
                    ctr += 1
                if ctr >= patience:
                    break

    best_critic.eval()
    best_critic.requires_grad_(False)
    jsd = 0.
    acc = 0.
    try:
        while True:
            x = next(x_test_sampler)[0]
            y = next(y_test_sampler)[0] if Y is not None else torch.randn_like(x)
            x = F.normalize(x, p=2, dim=-1) / tau
            y = F.normalize(y, p=2, dim=-1) / tau
            cx, cy = torch.chunk(best_critic(torch.cat([x, y])), 2)
            jsd = jsd + F.logsigmoid(cx).double().sum() / N_test
            jsd = jsd + F.logsigmoid(-cy).double().sum() / N_test
            acc = acc + ((cx > 0.).sum() + (cy < 0.).sum()) / (2 * N_test)
    except StopIteration:
        pass

    jsd = 0.5 * jsd / np.log(2) + 1

    return float(jsd.float()), float(train_acc_meter.avg), float(best_val_acc), float(acc), best_critic


@torch.no_grad()
def test_for_outlier(critic, z, threshold=0., tau=1.):
    z = z.flatten(1)
    N = z.size(0)
    zset = torch.utils.data.TensorDataset(z)
    loader = torch.utils.data.DataLoader(zset, batch_size=128, shuffle=True)
    f = lambda x: critic(F.normalize(x[0], p=2, dim=-1).div(tau))
    one_zero = sum((f(x) < threshold).sum() for x in iter(loader)).float()
    return float(one_zero / N)


@torch.no_grad()
def train_classifier(data, labels, attribute_info, solver='cholesky', fit_intercept=False, ridge=False):
    if ridge:
        classifiers = [(RidgeClassifier(solver=solver, fit_intercept=fit_intercept) if info[1] == 'classification' else Ridge(solver=solver, fit_intercept=fit_intercept)) for info in attribute_info.values()]
    else:
        classifiers = [(LogisticRegression(fit_intercept=True) if info[1] == 'classification' else Ridge(solver=solver, fit_intercept=fit_intercept)) for info in attribute_info.values()]
    for classifier, i in zip(classifiers, attribute_info):
        classifier.fit(data, labels[i])
    return classifiers


@torch.no_grad()
def fetch_scores(data, labels, classifiers, attribute_info):
    return [classifier.score(data, labels[i]) for classifier, i in zip(classifiers, attribute_info)]


@torch.no_grad()
def fetch_scores_gradually(loader, encoder, classifiers, attribute_info,
                           topk=1, device=None):
    y_true = []
    y = []
    for data in tqdm(iter(loader)):
        x = data[0].to(device=device)
        y_true.append(data[1:])
        z = encoder(x).cpu().numpy()
        y_ = []
        for classifier, info in zip(classifiers, attribute_info.values()):
            if info[1] == 'classification':
                y_.append(classifier.decision_function(z))
            elif info[1] == 'regression':
                y_.append(classifier.predict(z))
            else:
                raise ValueError(f'Invalid supervision type: {info[1]}')
        y.append(y_)
    y_true = [torch.cat(x).cpu().numpy() for x in zip(*y_true)]
    y = [np.concatenate(x) for x in zip(*y)]

    scores = []
    for i, info in attribute_info.items():
        if info[1] == 'classification':
            score = metrics.top_k_accuracy_score(y_true[i], y[i], k=topk, labels=np.arange(info[0]))
        elif info[1] == 'regression':
            score = metrics.r2_score(y_true[i], y[i])
        else:
            raise ValueError(f'Invalid supervision type: {info[1]}')
        scores.append(score)

    return scores

