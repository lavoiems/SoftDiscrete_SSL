import json
import os
import torch
import pprint

import numpy as np
import pandas as pd

from importlib import import_module
from scipy.sparse import csr_matrix

from common.eval_tools import (train_classifier, fetch_scores_gradually, fetch_scores)
from common import encode, util

from ..train import init_models, embed_, extract_features, normalize


np.set_printoptions(precision=4)


@torch.no_grad()
def extract(logger, m, loader, discrete, norm_feats, device):
    z, y, labels = extract_features(m, loader, device, discrete)
    logger.exp.info('Features extracted')

    z, stats_z = normalize(z, norm_feats)

    return z, y, labels


def evaluate(logger, eval_stats, x, labels, classifier, split, feature, attribute_info, device):
    scores = fetch_scores(x, labels, classifier, attribute_info)
    scores = np.asarray(scores)
    logger.eval.info(f"step: BEST | [{feature}] {split}:\n{scores}")
    for i, score in zip(attribute_info, scores):
        eval_stats[f'score_{str(i)}:{split}:{feature}'] = [score]
    return eval_stats


def parse_args(parser):
    parser.add_argument('--data-path', type=str, default='./data')
    parser.add_argument('--solver', type=str, default='cholesky')
    parser.add_argument('--run-path', type=str)
    parser.add_argument('--ridge', type=eval, default=True)
    parser.add_argument('--out-path', type=str, default='.')


def execute(args):
    logger = args.logger

    print('Evaluation parameters:')
    pprint.pprint(vars(args))
    # Prepare model
    params_path = os.path.join(args.run_path, 'args.json')
    with open(params_path) as f:
        parameters = json.load(f)
    print(parameters)

    discrete = not (parameters['encode_method'] in ['Continuous', 'Sphere'])

    if not discrete:
        if parameters['encode_method'] == 'Continuous':
            parameters['voc_size'] = 0
            parameters['message_size'] = parameters['hidden_size']

    print('Model parameters:')
    pprint.pprint(parameters)
    logger.exp.info("Model args:\n%s", parameters)

    # Load dataset
    parameters['device'] = args.device
    parameters['data_path'] = args.data_path
    dataset = parameters['dataset']
    loaders, attribute_info, norm_info, channels, _ = import_module(f'common.loaders.{dataset}').load(**parameters)

    parameters['data_channels'] = channels

    ##################################
    #  Define Models and Optimizers  #
    ##################################

    code = encode.Code(parameters['message_size'], parameters['voc_size'],
                       continuous=not discrete)

    models = init_models(code, data_norm=norm_info, **parameters)

    model_path = os.path.join(args.run_path, 'states', 'BEST')
    models.load_state_dict(torch.load(model_path).models['model'])
    models.eval()
    models.to(args.device)

    m = embed_(models)

    eval_stats = util.accumulator()
    score_df = eval_stats.to_dataframe

    logger.exp.info(f'Extracting features on set train')
    z, y, labels = extract(logger, m, loaders['train'], discrete, parameters['norm_feats'], args.device)
    logger.exp.info('Training classifiers')
    classifiers_on_z = train_classifier(z.cpu().numpy(), labels, attribute_info, solver=args.solver, ridge=args.ridge)
    classifiers_on_y = train_classifier(y, labels, attribute_info, ridge=args.ridge, solver='auto')

    logger.exp.info('Evaluating at val')
    logger.exp.info(f'Extracting features on set val')
    z, y, labels = extract(logger, m, loaders['val'], discrete, parameters['norm_feats'], args.device)
    eval_stats = evaluate(logger, score_df, z, labels, classifiers_on_z, 'val', 'z', attribute_info, args.device)
    eval_stats = evaluate(logger, score_df, y, labels, classifiers_on_y, 'val', 'y', attribute_info, args.device)

    logger.exp.info('Evaluating at test ood')
    logger.exp.info(f'Extracting features on set test')
    z, y, labels = extract(logger, m, loaders['ood'], discrete, parameters['norm_feats'], args.device)
    eval_stats = evaluate(logger, score_df, z, labels, classifiers_on_z, 'ood', 'z', attribute_info, args.device)
    eval_stats = evaluate(logger, score_df, y, labels, classifiers_on_y, 'ood', 'y', attribute_info, args.device)

    logger.exp.info('Extracting features on set test iid')
    z, y, labels = extract(logger, m, loaders['iid'], discrete, parameters['norm_feats'], args.device)
    eval_stats = evaluate(logger, score_df, z, labels, classifiers_on_z, 'iid', 'z', attribute_info, args.device)
    eval_stats = evaluate(logger, score_df, y, labels, classifiers_on_y, 'iid', 'y', attribute_info, args.device)

    args_df = pd.DataFrame.from_records([parameters])
    df = pd.concat((score_df, args_df), axis=1)

    os.makedirs(args.out_path, exist_ok=True)
    out_file = os.path.join(args.out_path, parameters['run_name'])
    df.to_pickle(out_file, protocol=4)
