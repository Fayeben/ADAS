#!/usr/bin/env python
import os
import os.path as osp
import argparse
import logging
import time
import socket
import warnings

import torch
import torch.nn.functional as F
import torch.nn as nn
from tensorboardX import SummaryWriter

from xmuda.common.solver.build import build_optimizer, build_scheduler
from xmuda.common.utils.checkpoint import CheckpointerV2
from xmuda.common.utils.logger import setup_logger
from xmuda.common.utils.metric_logger import MetricLogger
from xmuda.common.utils.torch_util import set_random_seed
from xmuda.models.build_sampling import build_model_2d, build_model_3d
from xmuda.data.build_sampling import build_dataloader
from xmuda.data.utils.validate import validate
from xmuda.models.losses import entropy_loss

from xmuda.discriminator import BEVDiscriminator_Conv_2

import torch.optim as optim
import torch.backends.cudnn as cudnn
import tqdm
import pickle
import os
import io
import os.path as osp

import io
import cv2

from transformer import LinearHelixTransformerBlock


def get_fc_discriminator(input_channels=64, ndf=64):
    return nn.Sequential(
        nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2, padding=1),
    )



def parse_args():
    parser = argparse.ArgumentParser(description='xMUDA training')
    parser.add_argument(
        '--cfg',
        dest='config_file',
        default='',
        metavar='FILE',
        help='path to config file',
        type=str,
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    return args


def init_metric_logger(metric_list):
    new_metric_list = []
    for metric in metric_list:
        if isinstance(metric, (list, tuple)):
            new_metric_list.extend(metric)
        else:
            new_metric_list.append(metric)
    metric_logger = MetricLogger(delimiter='  ')
    metric_logger.add_meters(new_metric_list)
    return metric_logger


def bce_loss(y_pred, y_label):
    y_truth_tensor = torch.FloatTensor(y_pred.size())
    y_truth_tensor.fill_(y_label)
    y_truth_tensor = y_truth_tensor.to(y_pred.get_device())
    return torch.nn.BCEWithLogitsLoss()(y_pred, y_truth_tensor)


def lr_poly(base_lr, iter, max_iter, power):
    """ Poly_LR scheduler
    """
    return base_lr * ((1 - float(iter) / max_iter) ** power)


def adjust_learning_rate_discriminator(optimizer, i_iter, max_iteration, learning_rate = 2.5e-4, power = 0.9):
    lr = lr_poly(learning_rate, i_iter, max_iteration, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def active_evaluate(Attention, model_2d, model_3d, model_discriminator, target_loader):
    print("======> Active Evaluate <======")
    dataloader_iter_tar = iter(target_loader)
    total_iter_tar = len(dataloader_iter_tar)
    frame_scores = []
    return_scores = []
    model_2d.eval()
    model_3d.eval()
    model_discriminator.eval()
    pbar = tqdm.tqdm(total=total_iter_tar, leave=False, desc='active_evaluate', dynamic_ncols=True)

    for cur_it in range(total_iter_tar):
    # for cur_it in range(2):
        try:
            batch = next(dataloader_iter_tar)
        except StopIteration:
            dataloader_iter_tar = iter(target_loader)
            batch = next(dataloader_iter_tar)
            print('new iter')

        with torch.no_grad():
            # load_data_to_gpu(batch)
            # batch.cuda()
            batch['x'][1] = batch['x'][1].cuda()
            batch['seg_label'] = batch['seg_label'].cuda()
            batch['img'] = batch['img'].cuda()
            forward_args = {
                'mode': 'active_evaluate'
            }
            batch_dict = model_2d(batch)
            batch_dict_3d = model_3d(batch)
            attention, _ = Attention(batch_dict, batch_dict_3d)
            batch_dict_3d = model_discriminator(attention, domm='3d')
            batch_dict_3d = model_discriminator.domainness_evaluate(batch_dict_3d)            
            frame_id = [lidar_path for lidar_path in batch_dict['lidar_path']]
            domainness_evaluate_3d = batch_dict_3d['domainness_evaluate'].cpu()
            domainness_evaluate_sigmoid_3d = batch_dict_3d['domainness_evaluate_sigmoid'].cpu()
            sample_score = []
            for i in range(len(frame_id)):
                frame_score = {
                    'frame_id': frame_id[i],
                    'domainness_evaluate': domainness_evaluate_3d[i].cpu(),
                    'total_score': domainness_evaluate_3d[i].cpu()
                }
                sample_score.append(frame_score)
            frame_scores.append(sample_score)
        pbar.update()
        pbar.refresh()
    pbar.close()

    gather_scores = gather_all_scores(frame_scores)
    for score in gather_scores:
        for f_score in score:
            return_scores += f_score
    return return_scores

def gather_all_scores(frame_scores):
    scores = [frame_scores]
    return scores

def active_sample_source(frame_scores):
    sampled_frame_info = [item for item in frame_scores if item['total_score'] > 0]
    sampled_frame_id = [frame['frame_id'] for frame in sampled_frame_info]
    return sampled_frame_id, sampled_frame_info

def update_source_list(sample_list, target_list, sample_frame_id, iter, save_path):
    target_list = [item for item in target_list if item['lidar_path'] not in sample_frame_id]
    target_list_path = save_path / ('epoch-%d_target_list.pkl' % iter)
    with open(target_list_path, 'wb') as f:
        pickle.dump(target_list, f)
    return target_list, target_list_path

def update_sample_list(sample_list, target_list, sample_frame_id, iteration, save_path, settings="usa2singapore"):
    new_sample_list = [item for item in target_list if item['lidar_path'] in sample_frame_id]
    sample_list = sample_list + new_sample_list
    sample_list_path = osp.join(save_path, (settings + '_attn_iteration-%d_sample_list.pkl' % iteration))
    with open(sample_list_path, 'wb') as f:
        pickle.dump(sample_list, f)
    return sample_list, sample_list_path


def get_dataset_list(dataset_file, oss, sample_interval=10, waymo=False):
    if oss == True:
        from petrel_client.client import Client
        client = Client('~/.petreloss.conf')
    if waymo == False:
        if oss == True:
            target_list = read_s3_pkl("feiben", dataset_file)
        else:
            target_list = read_s3_pkl("feiben", dataset_file)
    else:
        data_path = '../data/waymo/ImageSets/train.txt'
        target_list = []
        sample_sequence_list = [x.strip() for x in open(data_path).readlines()]
        for k in tqdm.tqdm(range(len(sample_sequence_list))):
            sequence_name = os.path.splitext(sample_sequence_list[k])[0]
            if oss == False:
                info_path = Path(dataset_file) / sequence_name / ('%s.pkl' % sequence_name)
                if not Path(info_path).exists():
                    continue
            else:
                info_path = os.path.join(dataset_file, sequence_name, ('%s.pkl' % sequence_name))
                # if not Path(info_path).exists():
                #     continue
            
            if oss == False:
                with open(info_path, 'rb') as f:
                    infos = pickle.load(f)
                    target_list.extend(infos)
            else:
                pkl_bytes = client.get(info_path, update_cache=True)
                infos = pickle.load(io.BytesIO(pkl_bytes))
                target_list.extend(infos)
        if sample_interval > 1:
            sampled_waymo_infos = []
            for k in range(0, len(target_list), sample_interval):
                sampled_waymo_infos.append(target_list[k])
            target_list = sampled_waymo_infos
    return target_list

def train(cfg, output_dir='', run_name=''):
    # ---------------------------------------------------------------------------- #
    # Build models, optimizer, scheduler, checkpointer, etc.
    # ---------------------------------------------------------------------------- #

    LAMBDA_ADV_MAIN = 0.001 
    cudnn.benchmark = True
    cudnn.enabled = True


    logger = logging.getLogger('xmuda.train')

    #set_random_seed(cfg.RNG_SEED)


    # seg maps, i.e. output, level for  DISCRIMINATOR NETWORK
    GPU_ID = 0
    device = GPU_ID
    LEARNING_RATE_D = 1e-4
    d_main = BEVDiscriminator_Conv_2()
    d_main.train()
    d_main.to(device)
    optimizer_d_main = optim.Adam(d_main.parameters(), lr=LEARNING_RATE_D, betas=(0.9, 0.99))
    source_label = 0
    target_label = 1


    # build 2d model
    model_2d, train_metric_2d = build_model_2d(cfg)
    logger.info('Build 2D model:\n{}'.format(str(model_2d)))
    num_params = sum(param.numel() for param in model_2d.parameters())
    print('#Parameters: {:.2e}'.format(num_params))

    # build 3d model
    model_3d, train_metric_3d = build_model_3d(cfg)
    logger.info('Build 3D model:\n{}'.format(str(model_3d)))
    num_params = sum(param.numel() for param in model_3d.parameters())
    print('#Parameters: {:.2e}'.format(num_params))

    model_2d = model_2d.cuda()
    model_3d = model_3d.cuda()


    # build optimizer
    optimizer_2d = build_optimizer(cfg, model_2d)
    optimizer_3d = build_optimizer(cfg, model_3d)

    # build lr scheduler
    scheduler_2d = build_scheduler(cfg, optimizer_2d)
    scheduler_3d = build_scheduler(cfg, optimizer_3d)

    scheduler_discriminator = build_scheduler(cfg, optimizer_d_main)
    # build checkpointer
    # Note that checkpointer will load state_dict of model, optimizer and scheduler.
    checkpointer_2d = CheckpointerV2(model_2d,
                                     optimizer=optimizer_2d,
                                     scheduler=scheduler_2d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_2d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_2d = checkpointer_2d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    checkpointer_3d = CheckpointerV2(model_3d,
                                     optimizer=optimizer_3d,
                                     scheduler=scheduler_3d,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_3d',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_3d = checkpointer_3d.load(cfg.RESUME_PATH, resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    checkpointer_discriminator = CheckpointerV2(d_main,
                                     optimizer=optimizer_d_main,
                                     scheduler=scheduler_discriminator,
                                     save_dir=output_dir,
                                     logger=logger,
                                     postfix='_discriminator',
                                     max_to_keep=cfg.TRAIN.MAX_TO_KEEP)
    checkpoint_data_discriminator = checkpointer_discriminator.load("", resume=cfg.AUTO_RESUME, resume_states=cfg.RESUME_STATES)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build tensorboard logger (optionally by comment)
    if output_dir:
        tb_dir = osp.join(output_dir, 'tb.{:s}'.format(run_name))
        summary_writer = SummaryWriter(tb_dir)
    else:
        summary_writer = None

    # ---------------------------------------------------------------------------- #
    # Train
    # ---------------------------------------------------------------------------- #
    max_iteration = cfg.SCHEDULER.MAX_ITERATION
    start_iteration = checkpoint_data_2d.get('iteration', 0)

    # build data loader
    # Reset the random seed again in case the initialization of models changes the random state.
    #set_random_seed(cfg.RNG_SEED)
    train_dataloader_src = build_dataloader(cfg, mode='train', domain='source', start_iteration=start_iteration)
    train_dataloader_trg = build_dataloader(cfg, mode='train', domain='target', start_iteration=start_iteration)
    val_period = cfg.VAL.PERIOD
    val_dataloader = build_dataloader(cfg, mode='val', domain='target') if val_period > 0 else None

    best_metric_name = 'best_{}'.format(cfg.VAL.METRIC)
    best_metric = {
        '2d': checkpoint_data_2d.get(best_metric_name, None),
        '3d': checkpoint_data_3d.get(best_metric_name, None)
    }
    best_metric_iter = {'2d': -1, '3d': -1}
    logger.info('Start training from iteration {}'.format(start_iteration))

    # add metrics
    train_metric_logger = init_metric_logger([train_metric_2d, train_metric_3d])
    val_metric_logger = MetricLogger(delimiter='  ')

    def setup_train():
        # set training mode
        model_2d.train()
        model_3d.train()
        # reset metric
        train_metric_logger.reset()

    def setup_validate():
        # set evaluate mode
        model_2d.eval()
        model_3d.eval()
        # reset metric
        val_metric_logger.reset()

    if cfg.TRAIN.CLASS_WEIGHTS:
        class_weights = torch.tensor(cfg.TRAIN.CLASS_WEIGHTS).cuda()
    else:
        class_weights = None

    setup_train()
    end = time.time()
    train_iter_src = enumerate(train_dataloader_src)
    train_iter_trg = enumerate(train_dataloader_trg)
    sample_list_source = []
    for iteration in range(start_iteration, max_iteration):

        # fetch data_batches for source & target
        _, data_batch_src = train_iter_src.__next__()
        _, data_batch_trg = train_iter_trg.__next__()
        # print(data_batch_src.keys())
        data_time = time.time() - end
        # copy data from cpu to gpu
        if 'SCN' in cfg.DATASET_SOURCE.TYPE and 'SCN' in cfg.DATASET_TARGET.TYPE:
            # source
            data_batch_src['x'][1] = data_batch_src['x'][1].cuda()
            data_batch_src['seg_label'] = data_batch_src['seg_label'].cuda()
            data_batch_src['img'] = data_batch_src['img'].cuda()
            # target
            data_batch_trg['x'][1] = data_batch_trg['x'][1].cuda()
            data_batch_trg['seg_label'] = data_batch_trg['seg_label'].cuda()
            data_batch_trg['img'] = data_batch_trg['img'].cuda()
            if cfg.TRAIN.XMUDA.lambda_pl > 0:
                data_batch_trg['pseudo_label_2d'] = data_batch_trg['pseudo_label_2d'].cuda()
                data_batch_trg['pseudo_label_3d'] = data_batch_trg['pseudo_label_3d'].cuda()
        else:
            raise NotImplementedError('Only SCN is supported for now.')

        # reset optimizers
        optimizer_2d.zero_grad()
        optimizer_3d.zero_grad()
        optimizer_d_main.zero_grad()

        adjust_learning_rate_discriminator(optimizer_d_main, i_iter = iteration, max_iteration = max_iteration)
        # only train segnet. Don't accumulate grads in disciminators
        #for param in d_main.parameters():
        #    param.requires_grad = False

        # ---------------------------------------------------------------------------- #
        # Train on source
        # ---------------------------------------------------------------------------- #

        preds_2d = model_2d(data_batch_src)
        preds_3d = model_3d(data_batch_src)

        # segmentation loss: cross entropy
        seg_loss_src_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch_src['seg_label'], weight=class_weights)
        seg_loss_src_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch_src['seg_label'], weight=class_weights)
        train_metric_logger.update(seg_loss_src_2d=seg_loss_src_2d, seg_loss_src_3d=seg_loss_src_3d)
        loss_2d = seg_loss_src_2d
        loss_3d = seg_loss_src_3d

        if cfg.TRAIN.XMUDA.lambda_xm_src > 0:
            # cross-modal loss: KL divergence
            seg_logit_2d = preds_2d['seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            xm_loss_src_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                                      F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            xm_loss_src_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
                                      F.softmax(preds_2d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            train_metric_logger.update(xm_loss_src_2d=xm_loss_src_2d,
                                       xm_loss_src_3d=xm_loss_src_3d)
            loss_2d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_2d
            loss_3d += cfg.TRAIN.XMUDA.lambda_xm_src * xm_loss_src_3d

        # update metric (e.g. IoU)
        with torch.no_grad():
            train_metric_2d.update_dict(preds_2d, data_batch_src)
            train_metric_3d.update_dict(preds_3d, data_batch_src)

        loss_2d.backward()
        loss_3d.backward()



        # ---------------------------------------------------------------------------- #
        # Train on target
        # ---------------------------------------------------------------------------- #

        preds_2d = model_2d(data_batch_trg)
        preds_3d = model_3d(data_batch_trg)

        loss_2d = []
        loss_3d = []
        if cfg.TRAIN.XMUDA.lambda_xm_trg > 0:
            # cross-modal loss: KL divergence
            seg_logit_2d = preds_2d['seg_logit2'] if cfg.MODEL_2D.DUAL_HEAD else preds_2d['seg_logit']
            seg_logit_3d = preds_3d['seg_logit2'] if cfg.MODEL_3D.DUAL_HEAD else preds_3d['seg_logit']
            xm_loss_trg_2d = F.kl_div(F.log_softmax(seg_logit_2d, dim=1),
                                      F.softmax(preds_3d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            xm_loss_trg_3d = F.kl_div(F.log_softmax(seg_logit_3d, dim=1),
                                      F.softmax(preds_2d['seg_logit'].detach(), dim=1),
                                      reduction='none').sum(1).mean()
            train_metric_logger.update(xm_loss_trg_2d=xm_loss_trg_2d,
                                       xm_loss_trg_3d=xm_loss_trg_3d)
            loss_2d.append(cfg.TRAIN.XMUDA.lambda_xm_trg * xm_loss_trg_2d)
            loss_3d.append(cfg.TRAIN.XMUDA.lambda_xm_trg * xm_loss_trg_3d)
        if cfg.TRAIN.XMUDA.lambda_pl > 0:
            # uni-modal self-training loss with pseudo labels
            pl_loss_trg_2d = F.cross_entropy(preds_2d['seg_logit'], data_batch_trg['pseudo_label_2d'])
            pl_loss_trg_3d = F.cross_entropy(preds_3d['seg_logit'], data_batch_trg['pseudo_label_3d'])
            train_metric_logger.update(pl_loss_trg_2d=pl_loss_trg_2d,
                                       pl_loss_trg_3d=pl_loss_trg_3d)
            loss_2d.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_2d)
            loss_3d.append(cfg.TRAIN.XMUDA.lambda_pl * pl_loss_trg_3d)
        if cfg.TRAIN.XMUDA.lambda_minent > 0:
            # MinEnt
            minent_loss_trg_2d = entropy_loss(F.softmax(preds_2d['seg_logit'], dim=1))
            minent_loss_trg_3d = entropy_loss(F.softmax(preds_3d['seg_logit'], dim=1))
            train_metric_logger.update(minent_loss_trg_2d=minent_loss_trg_2d,
                                       minent_loss_trg_3d=minent_loss_trg_3d)
            loss_2d.append(cfg.TRAIN.XMUDA.lambda_minent * minent_loss_trg_2d)
            loss_3d.append(cfg.TRAIN.XMUDA.lambda_minent * minent_loss_trg_3d)

        sum(loss_2d).backward()
        sum(loss_3d).backward()

        # TODO: LAMBDA_ADV_MAIN等参数的设置
        # adversarial training to fool the discriminator with target samples   preds_2d = model_2d(data_batch_trg)
        #for param in d_main.parameters():
        #    param.requires_grad = True    #TODO
        # train with target samples to fool the discriminator
        Attention = LinearHelixTransformerBlock(dim=1, num_heads=1).cuda()
        attention, _ = Attention(preds_2d, preds_3d)
        d_out_main = d_main(attention, domm='3d')
        loss_adv_trg_main = bce_loss(d_out_main['domainness'], source_label)
        loss_2d = LAMBDA_ADV_MAIN * loss_adv_trg_main  # cfg.TRAIN.LAMBDA_ADV_MAIN
        # backward
        loss_2d.backward()
        train_metric_logger.update(loss_adv_trg = loss_adv_trg_main)

        optimizer_d_main.zero_grad()
        ## enable training mode on discriminator networks
        for param in d_main.parameters():
            param.requires_grad = True

        #Train discriminator networks with target
        d_out_main = d_main(attention, domm='3d')
        loss_d_main = d_main.get_discriminator_loss(d_out_main,source=False)
        loss_d_main.backward()

        # Train discriminator networks with source samples
        preds_2d = model_2d(data_batch_src)
        preds_3d = model_3d(data_batch_src)
        attention, _ = Attention(preds_2d, preds_3d)
        d_out_main = d_main(attention, domm='3d')
        loss_d_main = d_main.get_discriminator_loss(d_out_main,source=True)
        loss_d_main.backward()
        train_metric_logger.update(loss_d_trg_src = loss_d_main)


        optimizer_2d.step()
        optimizer_3d.step()
        optimizer_d_main.step()

        # ---------------------------------------------------------------------------- #
        # activate sample
        # ---------------------------------------------------------------------------- #
        source_list = get_dataset_list(osp.join(cfg.DATASET_SOURCE.NuScenesSCN.preprocess_dir, cfg.DATASET_TARGET.TRAIN[0] + '.pkl'), oss=True)

        if iteration % 1000 == 0 and iteration != start_iteration:
            sample_save_path = os.path.join(cfg.output_dir,'save_pkl_2d_3d_attn_target')
            frame_score = active_evaluate(Attention, model_2d, model_3d, d_main, train_dataloader_trg)
            sampled_frame_id_source, _ = active_sample_source(frame_score)
            sample_list_source, info_path_source = update_sample_list(sample_list_source, source_list, sampled_frame_id_source,
                                                                                            iteration, sample_save_path, settings=cfg.DATASET_SOURCE.SETTING)

        batch_time = time.time() - end
        train_metric_logger.update(time=batch_time, data=data_time)

        # log
        cur_iter = iteration + 1
        if cur_iter == 1 or (cfg.TRAIN.LOG_PERIOD > 0 and cur_iter % cfg.TRAIN.LOG_PERIOD == 0):
            logger.info(
                train_metric_logger.delimiter.join(
                    [
                        'iter: {iter:4d}',
                        '{meters}',
                        'lr: {lr:.2e}',
                        'max mem: {memory:.0f}',
                    ]
                ).format(
                    iter=cur_iter,
                    meters=str(train_metric_logger),
                    lr=optimizer_2d.param_groups[0]['lr'],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )

        # summary
        if summary_writer is not None and cfg.TRAIN.SUMMARY_PERIOD > 0 and cur_iter % cfg.TRAIN.SUMMARY_PERIOD == 0:
            keywords = ('loss', 'acc', 'iou')
            for name, meter in train_metric_logger.meters.items():
                if all(k not in name for k in keywords):
                    continue
                summary_writer.add_scalar('train/' + name, meter.avg, global_step=cur_iter)

        # checkpoint
        if (ckpt_period > 0 and cur_iter % ckpt_period == 0) or cur_iter == max_iteration:
            checkpoint_data_2d['iteration'] = cur_iter
            checkpoint_data_2d[best_metric_name] = best_metric['2d']
            checkpointer_2d.save('model_2d_{:06d}'.format(cur_iter), **checkpoint_data_2d)
            checkpoint_data_3d['iteration'] = cur_iter
            checkpoint_data_3d[best_metric_name] = best_metric['3d']
            checkpointer_3d.save('model_3d_{:06d}'.format(cur_iter), **checkpoint_data_3d)
            checkpoint_data_discriminator['iteration'] = cur_iter
            checkpointer_discriminator.save('model_discriminator_{:06d}'.format(cur_iter), **checkpoint_data_discriminator)
        # ---------------------------------------------------------------------------- #
        # validate for one epoch
        # ---------------------------------------------------------------------------- #
        if val_period > 0 and (cur_iter % val_period == 0 or cur_iter == max_iteration):
            start_time_val = time.time()
            setup_validate()

            validate(cfg,
                     model_2d,
                     model_3d,
                     val_dataloader,
                     val_metric_logger)

            epoch_time_val = time.time() - start_time_val
            logger.info('Iteration[{}]-Val {}  total_time: {:.2f}s'.format(
                cur_iter, val_metric_logger.summary_str, epoch_time_val))

            # summary
            if summary_writer is not None:
                keywords = ('loss', 'acc', 'iou')
                for name, meter in val_metric_logger.meters.items():
                    if all(k not in name for k in keywords):
                        continue
                    summary_writer.add_scalar('val/' + name, meter.avg, global_step=cur_iter)

            # best validation
            for modality in ['2d', '3d']:
                cur_metric_name = cfg.VAL.METRIC + '_' + modality
                if cur_metric_name in val_metric_logger.meters:
                    cur_metric = val_metric_logger.meters[cur_metric_name].global_avg
                    if best_metric[modality] is None or best_metric[modality] < cur_metric:
                        best_metric[modality] = cur_metric
                        best_metric_iter[modality] = cur_iter

            # restore training
            setup_train()

        scheduler_2d.step()
        scheduler_3d.step()
        end = time.time()

    for modality in ['2d', '3d']:
        logger.info('Best val-{}-{} = {:.2f} at iteration {}'.format(modality.upper(),
                                                                     cfg.VAL.METRIC,
                                                                     best_metric[modality] * 100,
                                                                     best_metric_iter[modality]))


def main():
    args = parse_args()

    # load the configuration
    # import on-the-fly to avoid overwriting cfg
    from xmuda.common.config import purge_cfg
    from xmuda.config.xmuda import cfg
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    purge_cfg(cfg)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    # replace '@' with config path
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        output_dir = output_dir.replace('@', config_path.replace('configs/', ''))
        if osp.isdir(output_dir):
            warnings.warn('Output directory exists.')
        os.makedirs(output_dir, exist_ok=True)

    # run name
    timestamp = time.strftime('%m-%d_%H-%M-%S')
    hostname = socket.gethostname()
    run_name = '{:s}.{:s}'.format(timestamp, hostname)

    logger = setup_logger('xmuda', output_dir, comment='train.{:s}'.format(run_name))
    logger.info('{:d} GPUs available'.format(torch.cuda.device_count()))
    logger.info(args)

    logger.info('Loaded configuration file {:s}'.format(args.config_file))
    logger.info('Running with config:\n{}'.format(cfg))

    # check that 2D and 3D model use either both single head or both dual head
    assert cfg.MODEL_2D.DUAL_HEAD == cfg.MODEL_3D.DUAL_HEAD
    # check if there is at least one loss on target set
    print(cfg.TRAIN.XMUDA.lambda_xm_src, cfg.TRAIN.XMUDA.lambda_xm_trg, cfg.TRAIN.XMUDA.lambda_pl, cfg.TRAIN.XMUDA.lambda_minent)
    assert cfg.TRAIN.XMUDA.lambda_xm_src > 0 or cfg.TRAIN.XMUDA.lambda_xm_trg > 0 or cfg.TRAIN.XMUDA.lambda_pl > 0 or \
           cfg.TRAIN.XMUDA.lambda_minent > 0
    train(cfg, output_dir, run_name)


if __name__ == '__main__':
    main()
