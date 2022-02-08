import os
import json
from collections import defaultdict
import functools
import logging

import numpy
import torch
import random
import pandas


def set_paths(args):
    args.root_path = os.path.abspath(os.path.expanduser(args.root_path))
    args.save_path = os.path.join(args.root_path, args.model, args.exp_name, args.run_name)
    os.makedirs(args.save_path, exist_ok=True)
    args.state_path = os.path.join(args.save_path, 'states')
    os.makedirs(args.state_path, exist_ok=True)
    args.log_file = os.path.join(args.save_path, 'log.txt')
    args.results_file = os.path.join(args.save_path, 'results.pkl')


def dump_args(args):
    if args.reload and os.path.isfile(os.path.join(args.save_path, 'args.json')):
        return
    args_dict = get_args_dict(args)
    json.dump(args_dict, open(os.path.join(args.save_path, 'args.json'), 'w'))


def get_args(save_path):
    args_path = os.path.join(save_path, 'args.json')
    with open(args_path, 'r') as f:
        return json.load(f)


def get_args_dict(args):
    builtin = ('basestring', 'bool', 'complex', 'dict', 'float', 'int',
               'list', 'long', 'str', 'tuple')
    args_dict = {k: v for k, v in vars(args).items()
                 if type(v).__name__ in builtin}
    return args_dict


def infinite_sampler(loader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)


def config_logger(logpath, displaying=True, saving=True, debug=False):
    os.makedirs(os.path.dirname(logpath), exist_ok=True)
    logger = logging.getLogger('Exp')
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(name)s:: time %(asctime)s | %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)


class accumulator(object):

    def __init__(self, init=None, mode='list'):
        if init is None:
            init = dict()
        if isinstance(init, accumulator):
            init = init.state
        else:
            try:
                init = init.to_dict(orient='list')
            except AttributeError:
                pass

        self.state = defaultdict(list)
        self.state.update(init)
        for k, v in self.state.items():
            if mode == 'list':
                try:
                    self.state[k] = v.tolist()
                except AttributeError:
                    self.state[k] = v
            elif mode == 'numpy':
                self.state[k] = numpy.asarray(v)
            elif mode == 'torch':
                self.state[k] = torch.Tensor(v)

    def __getattr__(self, k):
        return self.state[k]

    @property
    def to_torch(self):
        return accumulator(self, mode='torch')

    @property
    def to_numpy(self):
        return accumulator(self, mode='numpy')

    @property
    def to_list(self):
        return accumulator(self, mode='list')

    @property
    def to_dataframe(self):
        return pandas.DataFrame.from_dict(self.to_numpy.state)


def seed_prng(seed, use_cuda=False, deterministic=False):
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if use_cuda:
            torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = False
    random.seed(seed)
    numpy.random.seed(random.randint(1, 100000))
    torch.random.manual_seed(random.randint(1, 100000))
    if use_cuda is True:
        torch.cuda.manual_seed_all(random.randint(1, 100000))


def eval_ctx(model, debug=False, no_grad=True):
    def wrap_step(efun):
        @functools.wraps(efun)
        def eval_step(*args, **kwargs):
            model.eval()
            torch.autograd.set_detect_anomaly(debug)
            with torch.set_grad_enabled(mode=not no_grad):
                returns = efun(*args, **kwargs)
            torch.autograd.set_detect_anomaly(False)
            model.train()
            return returns
        return eval_step
    return wrap_step


def train_ctx(scheduler=None, debug=False):
    def wrap_step(lfun):
        @functools.wraps(lfun)
        def train_step(model, optimizer, *args, _optim=True, **kwargs):
            torch.autograd.set_detect_anomaly(debug)
            model.train()
            if _optim:
                optimizer.zero_grad()
            outs = lfun(model, *args, **kwargs)
            if isinstance(outs, tuple):
                loss = outs[0]
            else:
                loss = outs
            loss_ = sum(loss.values(), 0.)
            if _optim:
                loss_.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()
            torch.autograd.set_detect_anomaly(False)
            return outs
        return train_step
    return wrap_step


class average_meter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __call__(self, val):
        self.val = val
        n = self.val.numel()
        self.sum += self.val.mean().to(dtype=torch.float64) * n
        self.count += n
        self.avg = self.sum.div(self.count).to(dtype=torch.float32)
        return self.avg


class running_average_meter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0
        self.count = 0

    def __call__(self, val, mom=None):
        mom = mom or self.momentum
        if mom is None:
            self.avg = 0.
        elif self.count == 0:
            self.avg = val
        else:
            self.avg = mom * self.avg + (1 - mom) * val
        self.val = val
        self.count += 1
        return self.avg
