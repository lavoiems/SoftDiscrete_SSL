import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix
from scipy import sparse

from common import util
from common.optim import init_optimizer
from common.ema import copy_ema, update_ema
from common import encode
from common.eval_tools import (train_classifier, fetch_scores)

from . import model


np.set_printoptions(precision=4)


def normalize(z, norm_feats='no', mean_z=None, std_z=None):
    if norm_feats == 'l2':
        z = F.normalize(z, p=2, dim=-1)
    elif norm_feats == 'standardize':
        if mean_z is None:
            mean_z = z.mean(0, keepdim=True)
        if std_z is None:
            std_z = z.std(0, keepdim=True)
        z = (z - mean_z) / std_z
    else:
        pass
    return z, (mean_z, std_z)


def init_models(code, channel_tau=1.0, device=None,
                projector_type='Linear', embedder_type='Linear',
                encode_method='GumbelSoftmax',
                predictor_type='Linear', **parameters):
    encoder = model.encoder(**parameters)
    hidden_size = parameters.pop('hidden_size', 128)

    embedder = model.head(encoder.out_dim, code.total_dims, head_type=embedder_type,
                          norm_name=parameters['last_norm_emb'],
                          **parameters)
    bridge = getattr(encode, encode_method)
    bridge = bridge(embedder, code, channel_tau=channel_tau, **parameters)

    projector = model.head(code.total_dims, hidden_size, head_type=projector_type,
                           norm_name=parameters['last_norm_proj'],
                           **parameters)

    predictor = model.head(hidden_size, hidden_size, head_type=predictor_type,
                           norm_name='Identity',
                           **parameters)

    models = nn.ModuleDict(dict(encoder=encoder,
                                projector=projector,
                                bridge=bridge,
                                predictor=predictor))
    models.train()
    models.to(device)
    return models


def embed_(m: nn.Module):
    def run(x):
        encoded = m.encoder(x)
        outs = m.bridge(encoded)
        projected = m.projector(outs['embedding'])
        outs['projected'] = projected
        outs['encoded'] = encoded
        return outs
    return run


@torch.no_grad()
def extract_features(embed, loader, device=None, discrete=False):
    Y, Z, labels = [], [], []
    for data in iter(loader):
        x = data[0].to(device=device)
        labels.append(data[1:])

        e = embed(x)
        Z.append(e['encoded'].cpu())

        y = e['representation']
        if discrete:
            y = y.cpu().flatten(1).numpy()
            y = csr_matrix(y)
            Y.append(y)
        else:
            Y.append(y.cpu().flatten(1).numpy())

    labels = [torch.cat(x).cpu().numpy() for x in zip(*labels)]
    Z = torch.cat(Z).flatten(1)
    if discrete:
        Y = sparse.vstack(Y)
    else:
        Y = np.vstack(Y)
    return Z, Y, labels


def evaluation(models, args, device=None):
    discrete = not args.continuous_code

    @util.eval_ctx(models, debug=args.debug)
    def eval_step(step, attribute_info, embed, logger, eval_stats):
        def get_classifiers():
            z, y, labels = extract_features(embed, args.loaders['sub'], device=device, discrete=discrete)
            z, stats_z = normalize(z, args.norm_feats)
            classifiers_on_z = train_classifier(z.cpu().numpy(), labels, attribute_info,
                                                ridge=args.ridge)

            classifiers_on_y = train_classifier(y, labels, attribute_info,
                                                fit_intercept=not discrete,
                                                ridge=args.ridge)
            return classifiers_on_z, classifiers_on_y, stats_z

        classifiers_on_z, classifiers_on_y, stats_z = get_classifiers()

        def append_results(data, labels, classifiers, split, feature):
            scores = fetch_scores(data, labels, classifiers, attribute_info)
            for i, score in zip(attribute_info, scores):
                getattr(eval_stats, 'score_'+str(i)).append(score)
            eval_stats.step.append(step)
            eval_stats.split.append(split)
            eval_stats.feature.append(feature)
            return np.asarray(scores)

        def eval_on(split='iid'):
            z, y, labels = extract_features(embed, args.loaders[split], device=device, discrete=discrete)
            z, _ = normalize(z, args.norm_feats, *stats_z)
            scores = append_results(z.cpu().numpy(), labels, classifiers_on_z, split, 'z')
            logger.eval.info("step: %u | [z] %s: %s", step, split, scores)
            avg_score_z = np.mean(scores)

            scores = append_results(y, labels, classifiers_on_y, split, 'y')
            logger.eval.info("step: %u | [y] %s: %s", step, split, scores)
            avg_score_y = np.mean(scores)

            return max(avg_score_y, avg_score_z)

        eval_on('iid')
        all_score = eval_on('val')
        return all_score, eval_stats
    return eval_step


def train(exp):
    #############################
    #  Begginning of the train  #
    #############################
    parameters = vars(exp.args)
    args = exp.args

    exp.logger.info('Arguments:\n%s', parameters)
    exp.logger.info('Logging into: %s', args.save_path)

    @util.train_ctx(debug=args.debug)
    def train_ssl_step(models, x1, x2, key_models, log=False):
        def simclr(query, keys):
            query = models.predictor(query).view_as(keys)
            query = query.flatten(1)
            keys = keys.flatten(1)
            B, _ = keys.shape
            mask = torch.eye(B, dtype=bool, device=keys.device)
            queue_mask = torch.zeros(B, queue_total_size,
                                     dtype=bool, device=keys.device)
            mask = torch.cat([mask, torch.zeros_like(mask), queue_mask], dim=-1)

            positive_loss = F.cosine_similarity(query, keys, dim=-1).div(args.tau).mean().neg()

            extended_keys = torch.cat([query, keys, z_queue])
            negative_scores = F.cosine_similarity(query[:, None], extended_keys[None, :], dim=-1).div(args.tau)
            negative_scores = negative_scores.masked_fill(mask, float('-inf'))
            negative_loss = negative_scores.logsumexp(dim=-1).mean()
            return positive_loss + negative_loss

        discrete_key_encoder = embed_(key_models)

        if args.pass_views_together:
            x3 = torch.cat([x1, x2])
            ex3 = embed(x3)
            y3 = ex3['projected']
            y1, y2 = torch.chunk(y3, 2)
            ez3 = discrete_key_encoder(x3)['projected']
            ez1, ez2 = torch.chunk(ez3, 2)
        else:
            ex1 = embed(x1)
            ex2 = embed(x2)
            y1 = ex1['projected']
            y2 = ex2['projected']
            ez1 = discrete_key_encoder(x1)['projected']
            ez2 = discrete_key_encoder(x2)['projected']

        if exp.args.pass_views_together:
            ex = ex3
        else:
            ex = ex1
        metrics = ex['metrics'] if log else None

        loss = 0.5 * (simclr(y1, ez2) + simclr(y2, ez1))
        return dict(loss=loss), dict(keys=torch.cat([ez1, ez2])), metrics

    def reset_queue(encoder):
        start_idx = 0
        if queue_total_size > 0:
            for _ in range(args.queue_size):
                x1, x2 = next(train_ssl_iter)
                x1 = x1.to(exp.device)
                x2 = x2.to(exp.device)
                if args.pass_views_together:
                    x3 = torch.cat([x1, x2])
                    ez1, ez2 = torch.chunk(encoder(x3)['projected'], 2)
                else:
                    ez1, ez2 = encoder(x1)['projected'], encoder(x2)['projected']
                start_idx = update_queue(torch.cat([ez1, ez2]), start_idx)
        return start_idx

    @torch.no_grad()
    def update_queue(x, start_idx):
        if queue_total_size > 0:
            next_idx_queue = start_idx + x.size(0)
            diff = next_idx_queue - queue_total_size
            if diff > 0:
                z_queue.index_copy_(0,
                                    torch.arange(start_idx, queue_total_size, device=exp.device),
                                    x[:-diff])
                z_queue.index_copy_(0,
                                    torch.arange(0, diff, device=exp.device),
                                    x[-diff:])
                start_idx = diff
            else:
                z_queue.index_copy_(0,
                                    torch.arange(start_idx, next_idx_queue, device=exp.device),
                                    x)
                start_idx = next_idx_queue % queue_total_size
        return start_idx

    #################################
    #  Prepare Dataset and Loaders  #
    #################################
    train_full_loader = args.loaders['train']
    train_ssl_loader = args.loaders['train_ssl']
    train_iter = util.infinite_sampler(train_full_loader)
    train_ssl_iter = util.infinite_sampler(train_ssl_loader)
    attribute_info = args.attribute_info

    exp.logger.info('Training on dataset %s', args.dataset)
    exp.logger.info('Lengths:\n'+','.join([f'{k}={len(v.dataset)}' for k, v in args.loaders.items()]))

    ##################################
    #  Define Models and Optimizers  #
    ##################################
    code = encode.Code(args.message_size, args.voc_size,
                       continuous=args.continuous_code)

    models = init_models(code, device=exp.device, channel_tau=args.channel_tau_student, **parameters)
    exp.register(models)
    exp.logger.info('Architecture:\n%s', models)

    ema_models = init_models(code, device=exp.device, channel_tau=args.channel_tau_teacher, **parameters)
    ema_models = copy_ema(ema_models, models)
    exp.register(ema_models, name='ema_models')

    optimizer = init_optimizer(models.parameters(), args.optimizer, lr=args.lr, wd=args.wd)
    exp.register(optimizer)
    exp.logger.info('Optimizer:\n%s', optimizer)

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 800)
        exp.register(scheduler)
        exp.logger.info('Scheduler:\n%s', scheduler)

    ###################
    #  Resume & Init  #
    ###################
    try:
        if args.reload:
            eval_stats = exp.resume()  # Reload from LATEST
    except FileNotFoundError:
        exp.logger.error("Reload was True, but did not found a LATEST model. Starting from scratch...")
        args.reload = False
    if not args.reload:
        eval_stats = util.accumulator()

    ###############
    #  Procedure  #
    ###############
    embed = embed_(m=models)
    eval_step = evaluation(models, args, device=exp.device)

    queue_total_size = 2 * args.train_batch_size * args.queue_size
    key_models = models if queue_total_size == 0 else ema_models

    z_queue = torch.zeros(queue_total_size, args.hidden_size,
                          dtype=torch.float32, device=exp.device)
    # If we are using a memory bank, then use ema models to generate keys
    key_models = models if queue_total_size == 0 else ema_models
    # Reset memory bank
    idx_queue = reset_queue(embed_(m=key_models))
    z_queue = getattr(exp.info, 'z_queue', z_queue)
    idx_queue = getattr(exp.info, 'idx_queue', idx_queue)

    ###################
    #  Training Loop  #
    ###################
    exp.info.best_score = all_score = getattr(exp.info, 'best_score', float('-inf'))
    init_step = getattr(exp.info, 'step', -1) + 1
    for step in range(init_step, 1 + args.max_steps):
        exp.info.step = step
        log = step % args.log_every == 0
        evaluate = (step % args.eval_every == 0 and not args.final_eval_only)\
            or step == args.max_steps
        epoch = step // len(train_full_loader)

        if evaluate:
            all_score, eval_stats = eval_step(step, attribute_info, embed, exp.loggers, eval_stats)
            is_best = False
            if all_score > exp.info.best_score:
                exp.info.best_score = all_score
                is_best = True
            exp.info.z_queue = z_queue
            exp.info.idx_queue = idx_queue
            exp.checkpoint(f'step_{step}.torch', eval_stats, is_best=is_best)

        if step < args.warmup_steps:
            lr_scale = min(1., float(step + 1) / args.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * args.lr

        x1, x2 = next(train_ssl_iter)
        x1 = x1.to(exp.device)
        x2 = x2.to(exp.device)
        loss, aux, metrics = train_ssl_step(models, optimizer, x1, x2, key_models, log=log)
        idx_queue = update_queue(aux['keys'].detach(), idx_queue)
        update_ema(models, ema_models, args.ema)
        if step % len(train_full_loader) == 0:
            print(f'Epoch: {step//len(train_full_loader)}')
            if args.scheduler:
                scheduler.step()

        if log:
            loss['enc_grad'] = nn.utils.clip_grad_norm_(models.encoder.parameters(), float('inf'))
            loss.update(metrics)
            exp.loggers.train.info(
                f'step: {step} | epoch: {epoch} | ' +
                ' | '.join([f'{k}: {float(v):.4f}' for k, v in loss.items()])
            )

    return float(1 - all_score)  # Respond current best validation score (to be minimized)
