import time
import torch
import os
import argparse
import logging

from collections import namedtuple
from importlib import import_module

from common.util import set_paths, config_logger, seed_prng


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root-path', default='./experiments/')
    parser.add_argument('--exp-name', default='benchmark')
    parser.add_argument('--run-id', type=str, default=str(time.time()))
    parser.add_argument('--seed', type=int, default=0)
    return parser


def parse_names():
    parser = create_parser()
    parser.add_argument('model', type=str)
    parser.add_argument('evaluation', type=str)
    args, _ = parser.parse_known_args()
    return args.model, args.evaluation


def parse_args(model_name, evaluation_name, model):
    if not hasattr(model, 'parse_args') and not hasattr(model, 'execute'):
        raise NotImplementedError()
    parser = create_parser()

    parser.add_argument('model', type=str)
    subparsers = parser.add_subparsers(dest='evaluation')
    subparsers.required = True
    evaluation_parser = subparsers.add_parser(evaluation_name)
    model.parse_args(evaluation_parser)
    evaluation_parser.set_defaults(func=model.execute)
    return parser.parse_args()


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    root = os.path.dirname(os.path.realpath(__file__))
    models_root = os.path.join(root, 'models')
    models_path = [os.path.join(root, f) for f in os.listdir(models_root)]
    models_path = [m for m in models_path if not os.path.isfile(m)]
    models_name = [p.split('/')[-1] for p in models_path]
    model_name, evaluation_name = parse_names()
    evaluation = import_module('.'.join(('models', model_name, 'evaluate', evaluation_name)))
    args = parse_args(model_name, evaluation_name, evaluation)

    args.run_name = os.path.join(args.exp_name, 'evaluate', f'{args.run_id}-{args.seed}')

    set_paths(args)
    config_logger(args.log_file, saving=True)

    args.n_gpus = torch.cuda.device_count()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed:
        seed_prng(args.seed, use_cuda=torch.cuda.is_available())

    exp_logger = logging.getLogger('Exp')
    eval_logger = logging.getLogger('Exp.Eval ')
    Logger = namedtuple('Logger', 'exp eval')
    args.logger = Logger(exp=exp_logger, eval=eval_logger)

    args.func(args)

