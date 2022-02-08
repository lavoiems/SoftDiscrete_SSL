from importlib import import_module
import random

from .train import train
from common import loaders


CONTINUOUS_ENC = ['Continuous',]
DISCRETE_ENC = ['Softmax', 'GumbelSoftmax']
exclude_from_hypers = ['test_batch_size', 'max_steps']


def parse_args(parser):
    parser.add_argument('--max_steps', type=int, default=50000000)
    parser.add_argument('--warmup_steps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=random.randint(0, 999999))

    parser.add_argument('--dataset', type=str, default='dsprites')
    parser.add_argument('--dataset_K', type=int, default=1)
    parser.add_argument('--n_train', type=int, default=130000)
    parser.add_argument('--n_sub_train', type=int, default=50000)
    parser.add_argument('--datasplit', type=str, default='composition')

    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=256)

    parser.add_argument('--augment', type=str, default='resize',
                        choices=['resize', 'gaussian', 'crop'])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0.)
    parser.add_argument('--h_dim', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--last_enc_bn', type=eval, choices=[True, False], default=False)

    parser.add_argument('--optimizer', type=str, default='AdamW')
    parser.add_argument('--scheduler', type=eval,
                        choices=[True, False], default=False)
    parser.add_argument('--ridge', type=eval,
                        choices=[True, False], default=True)
    parser.add_argument('--norm_feats', type=str, default='l2',
                        choices=['no', 'standardize', 'l2'])

    parser.add_argument('--pass_views_together', type=eval,
                        choices=[True, False], default=False)
    parser.add_argument('--ema', type=float, default=0.99)

    parser.add_argument('--tau', type=float, default=1.)
    parser.add_argument('--queue_size', type=int, default=16,
                        help="memory bank size is `query_size` x 2 x train_batch_size")

    parser.add_argument('--message_size', type=int, default=128)
    parser.add_argument('--voc_size', type=int, default=16)
    parser.add_argument('--channel_tau_student', type=float, default=1., help='Gumbel softmax temperature parameter')
    parser.add_argument('--channel_tau_teacher', type=float, default=1., help='Gumbel softmax temperature parameter')
    parser.add_argument('--hard', type=eval, choices=[True, False], default=False)

    parser.add_argument('--encoder_type', type=str, default='CNN', choices=['CNN',])
    parser.add_argument('--embedder_type', type=str, default='Linear', choices=['Identity', 'Linear', 'MLP'])
    parser.add_argument('--projector_type', type=str, default='MLP', choices=['Identity', 'Linear', 'MLP'])
    parser.add_argument('--last_norm_proj', type=str,
                        choices=['BatchNorm', 'Identity'], default='BatchNorm')
    parser.add_argument('--last_norm_emb', type=str,
                        choices=['BatchNorm', 'Identity'], default='Identity')
    parser.add_argument('--last_bn_affine', type=eval,
                        choices=[True, False], default=True)
    parser.add_argument('--encode_method', type=str, default='Softmax',
                        choices=['Continuous', 'Softmax', 'GumbelSoftmax',])
    parser.add_argument('--predictor_type', type=str, default='Linear', choices=['Identity', 'Linear', 'MLP'])


def execute(exp):
    if exp.args.encode_method == 'Continuous':
        exp.args.continuous_code = True
        if exp.args.voc_size > 0:
            exp.logger.warning("Warning: voc size set to 0 because of continuous code.")
        exp.args.voc_size = 0
        if exp.args.message_size > 0:
            exp.logger.warning(f'Warning: message size set to hidden_size={exp.args.hidden_size} because of continuous code.')
        exp.args.message_size = exp.args.hidden_size
    else:
        exp.args.continuous_code = False

    data_loaders, attribute_info, norm_info, channels, augment = import_module(f'common.loaders.{exp.args.dataset}').load(**vars(exp.args), device=exp.device)

    exp.args.loaders = data_loaders
    exp.args.attribute_info = attribute_info
    exp.args.data_norm = norm_info
    exp.args.data_channels = channels
    exp.args.augment = augment

    return train(exp)
