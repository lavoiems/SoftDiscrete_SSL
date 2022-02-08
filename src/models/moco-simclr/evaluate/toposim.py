import json
import pprint
import logging
import os
import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import scipy
from tqdm import tqdm

from common.loaders import dsprites, mpi3d
from common import util

from common.models import MLP
from common.gumbel import (Code, GumbelSoftmax)

from .. import model


np.set_printoptions(precision=4)


def topographical_similarity(y, labels):
    def inp_dist(w1, w2):
        w10 = w1[:, 0].int()
        w20 = w2[:, 0].int()
        h0 = (w10 != w20).float()
        l11 = w1[:, 1].sub(w2[:, 1]).abs()
        #l12 = w1[:, 2].sub(w2[:, 2]).abs()
        #l12 = torch.minimum(w1[:, 2].sub(w2[:, 2]).abs(), w1[:, 2].add(1.).sub(w2[:, 2]).abs())
        w12 = ((w1[:, 2] % 0.25) * 4).masked_fill(w10 != 0, 0.) + ((w1[:, 2] % 0.5) * 2).masked_fill(w10 != 1, 0.) + w1[:, 2].masked_fill(w10 != 2, 0.)
        w22 = ((w2[:, 2] % 0.25) * 4).masked_fill(w20 != 0, 0.) + ((w2[:, 2] % 0.5) * 2).masked_fill(w20 != 1, 0.) + w2[:, 2].masked_fill(w20 != 2, 0.)
        l12 = torch.minimum(w12.sub(w22).abs(), w12.add(1.).sub(w22).abs())
        l13 = w1[:, 3].sub(w2[:, 3]).abs()
        l14 = w1[:, 4].sub(w2[:, 4]).abs()
        return h0 + l11 + l12 + l13 + l14

    def jsd(q1, q2):
        def kl(p1, p2):
            return p1.probs.mul(p1.logits - p2.logits).sum(-1)
        m = Categorical(probs=0.5 * (q1.probs + q2.probs))
        res = 0.5 * (kl(q1, m) + kl(q2, m)) / np.log(2)
        return res.sum(-1)

    bs = labels.size(0)
    idx = torch.randperm(bs, device=labels.device)
    dy = jsd(y, Categorical(probs=y.probs[idx]))
    dlabels = inp_dist(labels, labels[idx])
    return scipy.stats.spearmanr(dy.cpu().numpy(), dlabels.cpu().numpy())[0]


@torch.no_grad()
def extract_features(discrete_encoder, loader, device=None):
    tmp = util.accumulator()
    for data in tqdm(iter(loader)):
        x = data[0].to(device=device)
        tmp.labels.append(data[1:])
        y = discrete_encoder(x)
        tmp.y.append(Categorical(probs=y.probs.cpu()))

    labels = [torch.cat(x).cpu() for x in zip(*tmp.labels)]
    Y = Categorical(probs=torch.cat([y.probs for y in tmp.y]))
    return Y, labels


def parse_args(parser):
    parser.add_argument('--root-path', default='./experiments/')
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-batch-size', type=int, default=64)
    parser.add_argument('--test-batch-size', type=int, default=64)
    parser.add_argument('--reload', action='store_true')

    parser.add_argument('--dataset', type=str, default='dsprites')
    parser.add_argument('--model-name', type=str)


def execute(args):
    torch.multiprocessing.set_sharing_strategy('file_system')
    args.run_name = os.path.basename(args.model_name)
    util.set_paths(args)
    util.dump_args(args)
    args.n_gpus = torch.cuda.device_count()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed:
        util.seed_prng(args.seed, use_cuda=torch.cuda.is_available())

    pprint.pprint(vars(args))

    # Manage logistics
    util.config_logger(args.log_file, saving=True)
    exp_logger = logging.getLogger('Exp')
    eval_logger = logging.getLogger('Exp.Eval ')
    eval_stats = util.accumulator()

    # Prepare model
    model_name = os.path.join(args.model_name, 'states/BEST')
    json_name = os.path.join(args.model_name, 'args.json')
    with open(json_name) as f:
        parameters = json.load(f)
    exp_logger.info("Model args:\n%s", parameters)

    # Load dataset
    if args.dataset == 'dsprites':
        loaders, attribute_info, norm_info, channels, _ = dsprites.load_dsprites(**vars(args))
    elif args.dataset in ('mpi3d_real', 'mpi3d_toy'):
        loaders, attribute_info, norm_info, channels, _ = mpi3d.load_mpi3d(**vars(args))
    else:
        raise NotImplementedError()
    parameters['data_norm'] = norm_info
    parameters['data_channels'] = channels

    parameters['model'] = 'discrete-simclr-il'
    parameters['eval_seed'] = args.seed

    ##################################
    #  Define Models and Optimizers  #
    ##################################
    code = Code(parameters['message_size'], parameters['voc_size'])
    encoder = model.Encoder(**parameters)
    projector = MLP(parameters['hidden_size'], code.total_dims, bias=False)
    bridge = GumbelSoftmax(code, tau=parameters['channel_tau'], hard=parameters['hard'])
    predictor = MLP(code.total_dims, parameters['hidden_size'], bias=False)

    models = nn.ModuleDict(dict(encoder=encoder, predictor=predictor,
                                projector=projector, bridge=bridge))
    models.load_state_dict(torch.load(model_name)['model'])
    models.eval()
    models.to(args.device)
    m = models

    def discrete_encoder(x):
        z = m.encoder(x)
        outs = m.bridge(m.projector(z))
        y = outs['q_y']  # Differentiable categorical embedding (e.g. from gumbel-softmax)
        return y

    exp_logger.info('Extracting discrete features')
    t_y, t_labels = extract_features(discrete_encoder, loaders['train_full'], device=args.device)
    iid_y, iid_labels = extract_features(discrete_encoder, loaders['iid'], device=args.device)
    ood_y, ood_labels = extract_features(discrete_encoder, loaders['ood'], device=args.device)
    y = Categorical(probs=torch.cat([t_y.probs, iid_y.probs, ood_y.probs]))
    labels = [torch.cat(x) for x in zip(t_labels, iid_labels, ood_labels)]
    labels = torch.stack(labels, dim=-1)
    t_labels = torch.stack(t_labels, dim=-1)
    ood_labels = torch.stack(ood_labels, dim=-1)

    toposim_all = topographical_similarity(y, labels)
    eval_stats.split.append('all')
    eval_stats.toposim.append(float(toposim_all))
    toposim_iid = topographical_similarity(t_y, t_labels)
    eval_stats.split.append('iid')
    eval_stats.toposim.append(float(toposim_iid))
    toposim_ood = topographical_similarity(ood_y, ood_labels)
    eval_stats.split.append('ood')
    eval_stats.toposim.append(float(toposim_ood))

    eval_logger.info("Model's topographical_similarity from generative factors to discrete latents is:\n%.4f/%.4f/%.4f",
                     float(toposim_all), float(toposim_iid), float(toposim_ood))

    res = eval_stats.to_dataframe.join(pd.DataFrame.from_dict({k: [str(v), ] * 3 for k, v in parameters.items()}))
    res.to_pickle(args.results_file, protocol=4)
