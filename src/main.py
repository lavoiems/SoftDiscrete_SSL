import os
import copy
from importlib import import_module
import argparse

from common.state import (State, Experiment)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', default='benchmark')
    parser.add_argument('--run_id', type=str, default='HIDDEN:{hidden_size}-LR:{lr:.4f}-ENCODE:{encode_method}-PROJ:{projector_type}-EMB:{embedder_type}-PRED:{predictor_type}')
    parser.add_argument('--root_path', default=os.getenv('ROOT_PATH', './experiments/'))
    parser.add_argument('--data_path', type=str, default=os.getenv('DATA_PATH', './data'))
    parser.add_argument('--reload', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_every', type=int, default=1)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--final_eval_only', action='store_true')
    return parser


def parse_model_name():
    parser = create_parser()
    parser.add_argument('model', type=str)
    args, _ = parser.parse_known_args()
    return args.model


def filter_args(args, exclude_from_hypers=None):
    """Exclude all from hparams all arguments in `main` + ones in model, except `seed`."""
    exclude_from_hypers = exclude_from_hypers or []
    default_main_args = vars(create_parser().parse_args('')).keys()
    exclude_from_hypers.extend(list(default_main_args) + ['func', ])
    params = copy.deepcopy(vars(args))
    params = dict(filter(lambda x: x[0] not in exclude_from_hypers, params.items()))
    return argparse.Namespace(**params)


def parse_args(name, model):
    parser = create_parser()
    subparsers = parser.add_subparsers(dest='model')
    subparsers.required = True
    if not hasattr(model, 'parse_args') and not hasattr(model, 'execute'):
        raise NotImplementedError()
    model_parser = subparsers.add_parser(name)
    model.parse_args(model_parser)
    model_parser.set_defaults(func=model.execute)
    args = parser.parse_args()
    params = filter_args(args, getattr(model, 'exclude_from_hypers', None))
    return args, params


if __name__ == '__main__':
    root = os.path.dirname(os.path.realpath(__file__))
    models_root = os.path.join(root, 'models')
    models_path = [os.path.join(root, f) for f in os.listdir(models_root)]
    models_path = [m for m in models_path if not os.path.isfile(m)]
    models_name = [p.split('/')[-1] for p in models_path]
    model_name = parse_model_name()
    model = import_module('.'.join(('models', model_name)))
    args, params = parse_args(model_name, model)
    args.job_type = 'train'

    state = State(params)
    exp = Experiment(args, state)
    exp.init()
    try:
        loss = args.func(exp)
    except RuntimeError:  # such as out-of-memory
        exp.logger.error("Encountered RuntimeError... reporting a bad trial.")
        raise
    else:
        exp.logger.info("Finalizing trial with loss: %f", loss)
        exp.finish()

