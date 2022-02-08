import logging
import argparse
import os
import hashlib
import json
import binascii
from collections import namedtuple
from typing import (Optional, Text)
from types import SimpleNamespace

import torch
import pandas

from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from common.util import seed_prng, accumulator, set_paths, dump_args, config_logger


def flatten_ns_to_dict(ns):
    return pandas.json_normalize(vars(ns), sep='.').to_dict(orient='list')


def _str_to_utf8bytes(x, errors="strict"):
    return x.encode("utf-8", errors=errors) if isinstance(x, str) else x


def _pbkdf2(dkLen, password, salt="", rounds=1, hash="sha256"):
    password = _str_to_utf8bytes(password)
    salt = _str_to_utf8bytes(salt)
    return hashlib.pbkdf2_hmac(hash, password, salt, rounds, dkLen)


def hash_params(params, length=32, mingle='experiment'):
    hparams = flatten_ns_to_dict(params)
    hparams['_mingle'] = mingle
    s = _pbkdf2(length,
                json.dumps(hparams, sort_keys=True),
                salt="hyperparameters", rounds=100001)
    return binascii.hexlify(s).decode('utf-8', errors='strict')


class State(object):
    """A state object which is supposed to be a singleton."""

    def __init__(self, params):
        # Hyperparameters that matter
        self.params = params

        # Training components management
        self.models = dict()
        self.optimizers = dict()
        self.schedulers = dict()

        # Traning info management
        self.info = SimpleNamespace()

    def _register_entity(self, entity, name, dtype, collection):
        entity_ = getattr(self, collection).get(name)
        if isinstance(entity, dtype):
            getattr(self, collection)[name] = entity
        elif isinstance(entity, dict):
            entity_.load_state_dict(entity)
        else:
            raise TypeError(collection)
        return getattr(self, collection)[name]

    def register_model(self, model, name: Optional[Text] = None):
        name = name or 'model'
        self._register_entity(model, name, Module, 'models')

    def register_optimizer(self, optimizer, name: Optional[Text] = None):
        name = name or 'optimizer'
        self._register_entity(optimizer, name, Optimizer, 'optimizers')

    def register_scheduler(self, scheduler, name: Optional[Text] = None):
        name = name or 'scheduler'
        self._register_entity(scheduler, name, _LRScheduler, 'schedulers')

    def dump(self, state_file):
        state = SimpleNamespace()
        state.models = {name: model.state_dict()
                        for name, model in self.models.items()}
        state.optimizers = {name: opti.state_dict()
                            for name, opti in self.optimizers.items()}
        state.schedulers = {name: sched.state_dict()
                            for name, sched in self.schedulers.items()}
        state.info = self.info
        state.params = self.params
        torch.save(state, state_file)

    def load(self, state_file):
        state = torch.load(state_file)
        assert(hash_params(self.params) == hash_params(state.params))
        self.info = state.info

        for name, model in state.models.items():
            self.register_model(model, name)
        for name, optimizer in state.optimizers.items():
            self.register_optimizer(optimizer, name)
        for name, scheduler in state.schedulers.items():
            self.register_scheduler(scheduler, name)


class Experiment(object):

    def __init__(self, args, state):
        # Logistics and experiment arguments that don't matter
        self.args = args
        # State of experiment object
        self.state = state

    @property
    def params(self):
        return self.state.params

    @params.setter
    def params(self, params_):
        self.state.params = params_

    @property
    def info(self):
        return self.state.info

    @property
    def logger(self):
        return self.loggers.exp

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device_: int):
        if device_ >= 0:
            self._device = torch.device('cuda', device_)
            self.logger.info("Using CUDA device: %d", device_)
        else:
            self._device = torch.device('cpu')
            self.logger.info("Using CPU")

    @property
    def use_cuda(self):
        return self.device != 'cpu'

    @property
    def dtype(self):
        return self._dtype

    @dtype.setter
    def dtype(self, prec: Text):
        if prec == 'fp32':
            self._dtype = torch.float32
        elif prec == 'fp64':
            self._dtype = torch.float64
        else:  # TODO for mixed prec etc
            raise NotImplementedError
        torch.set_default_dtype(self._dtype)
        self.logger.info("Default floating precision: %s", self._dtype)

    def init(self):
        # Form run_name and set args
        group_name = f'{self.args.model}/{self.args.exp_name}'
        param_id = hash_params(self.params, mingle=group_name)
        self.args.run_name = self.args.run_id.format(**vars(self.params)) + '_' + param_id
        set_paths(self.args)
        dump_args(self.args)

        # Complete init by loggers, device, dtype and prng
        config_logger(self.args.log_file, saving=True)
        exp_logger = logging.getLogger('Exp')
        train_logger = logging.getLogger('Exp.Train')
        eval_logger = logging.getLogger('Exp.Eval ')
        Logger = namedtuple('Logger', 'exp train eval')
        self.loggers = Logger(train=train_logger, exp=exp_logger, eval=eval_logger)

        self.device = 0 if torch.cuda.is_available() else -1
        self.dtype = getattr(self.args, 'dtype', 'fp32')
        seed_prng(self.args.seed, use_cuda=self.use_cuda)

    def register(self, entity, name: Optional[Text] = None):
        if isinstance(entity, Module):
            return self.state.register_model(entity, name)
        elif isinstance(entity, Optimizer):
            return self.state.register_optimizer(entity, name)
        elif isinstance(entity, _LRScheduler):
            return self.state.register_scheduler(entity, name)
        else:
            raise TypeError(f"Tried to register entity of unknown type: {entity.__class__.__name__}")

    def resume(self, state_file=None, results_file=None):
        state_file = state_file or os.readlink(os.path.join(self.args.state_path, 'LATEST'))
        self.state.load(state_file)

        results_file = results_file or self.args.results_file
        eval_stats = accumulator(pandas.read_pickle(results_file))

        return eval_stats

    def checkpoint(self, name, eval_stats, is_best=False, is_latest=True):
        state_file = os.path.join(self.args.state_path, name)
        self.state.dump(state_file)

        if is_latest:
            link_name = os.path.join(self.args.state_path, 'LATEST')
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(state_file, link_name)

        if is_best:
            link_name = os.path.join(self.args.state_path, 'BEST')
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(state_file, link_name)

        eval_stats.to_dataframe.to_pickle(self.args.results_file, protocol=4)

    def finish(self):
        pass
