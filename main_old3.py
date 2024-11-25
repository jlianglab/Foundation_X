# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------

## A CHANGE @@@@@

import os 

# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '28000'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from tqdm import tqdm
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os, sys
import numpy as np

import torch
import torch.nn
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from util.get_param_dicts import get_param_dict
from util.logger import setup_logger
from util.slconfig import DictAction, SLConfig
from util.utils import ModelEma, BestMetricHolder
import util.misc as utils
import copy
import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test, test_NAD
from sklearn.metrics import roc_auc_score, accuracy_score

# Segmentation from Jiaxuan
from utils_segmentation import load_popar_weight, AverageMeter, save_model, dice_score, mean_dice_coef, torch_dice_coef_loss, exp_lr_scheduler_with_warmup, step_decay, load_swin_pretrained
from datasets_medical import build_transform_segmentation, dataloader_return, PXSDataset, MontgomeryDataset, JSRTClavicleDataset, JSRTHeartDataset,JSRTLungDataset, VinDrRibCXRDataset, ChestXDetDataset, JSRTLungDataset, VindrCXRHeartDataset
from timm.utils import NativeScaler, ModelEma
from models.load_weights_model import load_weights
import math

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer, create_optimizer_v2, optimizer_kwargs

import pandas as pd
import csv

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)


device = torch.device("cuda")


# if 'SLURM_NPROCS' in os.environ:
#     args.world_size = int(os.environ['SLURM_NPROCS'])
# else:
#     args.world_size = 1  # Default value for non-cluster environment

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--config_file', '-c', type=str, required=True)
    parser.add_argument('--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')

    parser.add_argument('--serverC', type=str, default='SOL')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='/scratch/jliang12/data/coco-2017/')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--fix_size', action='store_true') 

    parser.add_argument('--segmentation_dataset', default='jsrt_lung')  
    parser.add_argument('--classification_dataset', default='imagenet')
    parser.add_argument('--imgsize', type=int, default=224) 
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--numClasses', type=int, default=1000, help='segmentation or classification class number') 
    parser.add_argument('--backbonemodel', default='Swin-L', help='Swin-T, Swin-B, Swin-L') 

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default=None, help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint') 
    parser.add_argument('--backbone_dir', default=None, type=str, help='loading backbone weights') #
    parser.add_argument('--init', default=None, type=str, help='imagenet22k | ark')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--total_epochs', default=500, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--tsneOut', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    parser.add_argument('--taskcomponent', default='detection', help='classification | segmentation | detection')
    parser.add_argument('--cyclictask', default='heart', help='heart | leftlung | rightlung | heart_leftlung | leftlung_rightlung | heart_leftlung_rightlung')

    parser.add_argument("--modelEMA", type=str, default=None, help="use EMA Model || True_Epoch | True_Iteration")


    # Optimizer parameters
    parser.add_argument('--opt', default='momentum', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, metavar='LR_Backbone',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--lr_segmentor', type=float, default=1e-2, metavar='LR_Segmentor',
                        help='learning rate (default: 1e-5)')

    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    # parser.add_option('--patience-epochs', type=int, default=10, metavar='N',
    #                     help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.5, metavar='RATE',
                        help='LR decay rate (default: 0.1)')




    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',
                        help="Train with mixed precision")
    
    return parser


def reinitialize_zero_weights(m):
    for name, param in m.named_parameters():
        if '.weight' in name and torch.sum(param.data) == 0:
            nn.init.xavier_uniform_(param.data)
    return m


def ema_update_teacher(model, teacher, momentum_schedule, it):
    with torch.no_grad():
        if it < 10:
            m = momentum_schedule[it]  # momentum parameter
        else:
            m = momentum_schedule[9]
        # m = momentum_schedule[it]
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def step_decay_cosine(step, learning_rate, num_epochs, warmup_epochs=5):
    lr = learning_rate
    progress = (step - warmup_epochs) / float(num_epochs - warmup_epochs)
    progress = np.clip(progress, 0.0, 1.0)
    #decay_type == 'cosine':
    lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
    if warmup_epochs:
      # lr = lr * np.minimum(1., step / warmup_epochs)
      lr = lr
    return lr


def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors


def train_one_epoch_SEGMENTATION(model, train_loader, optimizer, loss_scaler, epoch, head_number=None, log_writter_SEGMENTATION=None, model_ema=None, momen=None, coff=None, criterionMSE=None):
    model.train(True)
    # model_ema.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    criterion = torch_dice_coef_loss
    end = time.time()

    for idx, (img,mask) in enumerate(train_loader):

        if args.debug:
            if idx == 250:
                print("Segmentation DEBUG BREAK!"*5)
                break

        data_time.update(time.time() - end)
        bsz = img.shape[0]

        img = img.cuda(non_blocking=True) 
        mask = mask.cuda(non_blocking=True) 

        # with torch.cuda.amp.autocast():
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            out_SegmentationHead, features, features_cons = model.module.backbone[0].extra_features_seg(img, head_n=head_number)
        else:
            out_SegmentationHead, features, features_cons = model.backbone[0].extra_features_seg(img, head_n=head_number)

        outputs = torch.sigmoid( out_SegmentationHead )

        del out_SegmentationHead, features

        loss = criterion(mask, outputs)

        if model_ema is not None:
            # with torch.cuda.amp.autocast():
            # with torch.no_grad():
            _, _, features_cons_ema = model_ema.backbone[0].extra_features_seg(img, head_n=head_number)
            loss_cons = criterionMSE(features_cons, features_cons_ema)
            loss = (1-coff)*loss + coff*loss_cons


        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), file=log_writter_SEGMENTATION)
            sys.exit(1)
            # update metric
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=None,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if (idx + 1) % 100 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'lr {lr}\t'
                  'Total loss {ttloss.val:.5f} ({ttloss.avg:.5f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, lr=optimizer.param_groups[0]['lr'], ttloss=losses), file=log_writter_SEGMENTATION)
            log_writter_SEGMENTATION.flush()
            # if conf.debug_mode:
            #     break

    if args.modelEMA == "True_Epoch": # Epoch based EMA update # ACTIVE
        ema_update_teacher(model, model_ema, momen, epoch)

    return losses.avg


def evaluation_SEGMENTATION(model, val_loader, epoch, head_number=None, log_writter_SEGMENTATION=None):
    model.eval()
    losses = AverageMeter()
    criterion = torch_dice_coef_loss

    with torch.no_grad():
        for idx, (img, mask) in enumerate(val_loader):
            if args.debug:
                if idx == 100:
                    print("Segmentation Test Break!!"*5)
                    break
            bsz = img.shape[0]

            # img = img.double().cuda(non_blocking=True)
            # mask = mask.double().cuda(non_blocking=True)
            img = img.cuda(non_blocking=True) 
            mask = mask.cuda(non_blocking=True) 

            # if conf.arch == "swin_upernet":
            #     enco_out = model.extract_feat(img)
            #     outputs = model.decode_head.forward(enco_out)
            #     outputs = F.interpolate(outputs, size=conf.img_size, mode='bilinear')
            # else:
            # outputs = torch.sigmoid( model(img) ) # out_features, out_classifierHead, out_SegmentationHead = model.backbone(img)

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                # out_features, out_classifierHead, out_SegmentationHead = model.module.backbone(img)
                out_SegmentationHead, _, _ = model.module.backbone[0].extra_features_seg(img, head_n=head_number)
            else:
                # out_features, out_classifierHead, out_SegmentationHead = model.backbone(img)
                out_SegmentationHead, _, _ = model.backbone[0].extra_features_seg(img, head_n=head_number)
            outputs = torch.sigmoid( out_SegmentationHead )
            # del out_features, out_classifierHead, out_SegmentationHead

            loss = criterion(mask, outputs)

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), file=log_writter_SEGMENTATION)
                sys.exit(1)
                # update metric
            losses.update(loss.item(), bsz)


            torch.cuda.synchronize()


            if (idx + 1) % 100 == 0:
                print('Evaluation: [{0}][{1}/{2}]\t'
                      'Total loss {ttloss.val:.5f} ({ttloss.avg:.5f})'.format(
                    epoch, idx + 1, len(val_loader), ttloss=losses), file=log_writter_SEGMENTATION)
                log_writter_SEGMENTATION.flush()
                # if conf.debug_mode:
                #     break
    return losses.avg

def test_SEGMENTATION(model, test_loader, head_number=None, log_writter_SEGMENTATION=None):

    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
    # model.load_state_dict(checkpoint_model)

    # model = model.cuda()
    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    #     model = model.cuda()
    #     cudnn.benchmark = True

    features_segmentationList = []
    model.eval()
    with torch.no_grad():
        test_p = None
        test_y = None
        for idx, (img, mask) in enumerate(test_loader):

            if args.debug:
                if idx == 100:
                    print("Segmentation Test Break!!"*5)
                    break
            bsz = img.shape[0]
            with torch.cuda.amp.autocast():
                # img = img.double().cuda(non_blocking=True)
                # mask = mask.cuda(non_blocking=True)
                img = img.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

                # if conf.arch == "swin_upernet":
                #     enco_out = model.extract_feat(img)
                #     outputs = model.decode_head.forward(enco_out)
                #     outputs = F.interpolate(outputs, size=conf.img_size, mode='bilinear')
                # else:
                # outputs = torch.sigmoid(model(img))

                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    # out_features, out_classifierHead, out_SegmentationHead = model.module.backbone(img)
                    out_SegmentationHead, _, _ = model.module.backbone[0].extra_features_seg(img, head_n=head_number)
                else:
                    # out_features, out_classifierHead, out_SegmentationHead = model.backbone(img)
                    out_SegmentationHead, _, _ = model.backbone[0].extra_features_seg(img, head_n=head_number)

                if args.tsneOut:
                    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                        _, features = model.module.backbone[0].forward_raw(img)
                    else:
                        _, features = model.backbone[0].forward_raw(img)

                    features = features[-1]
                    # print("[CHECK] features0", len(features))
                    for iii in range(0, len(features)):
                        outs_detached = features[iii]
                        outs_detached = outs_detached.cpu().detach().numpy()
                        outs_detached = outs_detached.reshape(outs_detached.shape[0], outs_detached.shape[1]*outs_detached.shape[2])
                        # embeddings_2d_segmentation = tsne.fit_transform(outs_detached)
                        features_segmentationList.append(outs_detached)

                outputs = torch.sigmoid( out_SegmentationHead )

                outputs = outputs.cpu().detach()
                mask = mask.cpu().detach()

                # # FOR SAVING IMAGES
                # if conf.dataset =="vindrribcxr":
                #     for b in range(bsz):
                #         save_image(img[b].cpu().numpy().transpose(1, 2, 0), conf.model_path+"/{}_{}_input".format("test", b))
                #         for i in range(conf.num_classes):
                #             save_image(mask[b].cpu().numpy()[i],
                #                        conf.model_path + "/{}_iter_{}_batch_{}_mask_{}".format("test",idx, b,i))
                #             save_image(outputs[b].cpu().numpy()[i],
                #                        conf.model_path + "/{}_iter_{}_batch_{}_pred_{}".format("test",idx, b,i))

                # else:
                #     for b in range(bsz):
                #         save_image(img[b].cpu().numpy().transpose(1, 2, 0), conf.model_path+"/{}_{}_input".format("test", b))
                #         save_image(mask[b].cpu().squeeze(0).numpy(), conf.model_path+"/{}_{}_mask".format("test", b))
                #         save_image(outputs[b].cpu().squeeze(0).numpy(), conf.model_path+"/{}_{}_pred".format("test", b))


                if test_p is None and test_y is None:
                    test_p = outputs
                    test_y = mask
                else:
                    test_p = torch.cat((test_p, outputs), 0)
                    test_y = torch.cat((test_y, mask), 0)
                torch.cuda.empty_cache()
                if (idx + 1) % 100 == 0:
                    print("Testing Step[{}/{}] ".format(idx + 1, len(test_loader)), file=log_writter_SEGMENTATION)
                    log_writter_SEGMENTATION.flush()
                    # if conf.debug_mode:
                    #     break

        # print("Done testing iteration!", file=log_writter_SEGMENTATION)
        log_writter_SEGMENTATION.flush()

    test_p = test_p.numpy()
    test_y = test_y.numpy()
    test_y = test_y.reshape(test_p.shape)
    return test_y, test_p, features_segmentationList



class AverageMeter_CLASSIFICATION(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter_CLASSIFICATION(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy_CLASSIFICATION(output, target, topk=(1,)): # used for ImageNet
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def metric_AUROC(target, output, nb_classes=14): # used for NIH14
    outAUROC = []

    target = target.cpu().numpy()
    output = output.cpu().numpy()

    for i in range(nb_classes):
        outAUROC.append(roc_auc_score(target[:, i], output[:, i]))

    return outAUROC

def train_CLASSIFICATION(train_loader, model, criterion, optimizer, epoch, args, log_writter_CLASSIFICATION):
    batch_time = AverageMeter_CLASSIFICATION('Time', ':6.3f')
    data_time = AverageMeter_CLASSIFICATION('Data', ':6.3f')
    losses = AverageMeter_CLASSIFICATION('Loss', ':.4e')
    top1 = AverageMeter_CLASSIFICATION('Acc@1', ':6.2f')
    top5 = AverageMeter_CLASSIFICATION('Acc@5', ':6.2f')
    progress = ProgressMeter_CLASSIFICATION(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    # model.train(True)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # if i == 500: # Debug
        #     break

        # measure data loading time
        data_time.update(time.time() - end)

        # if args.gpu is not None:
        #     images = images.cuda(args.gpu, non_blocking=True)
        # if torch.cuda.is_available():
        #     target = target.cuda(args.gpu, non_blocking=True)

        images, target = images.float().to(device), target.float().to(device) # NIH14 # int for CheXpert

        # compute output
        # output = model(images)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # _, out_classifierHead, _ = model.module.backbone(images)
            out_classifierHead = model.module.backbone[0].extra_features(images)
        else:
            # _, out_classifierHead, _ = model.backbone(images)
            out_classifierHead = model.backbone[0].extra_features(images)
        output = out_classifierHead

        # print(output.shape, target.shape)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.classification_dataset == 'imagenet':
            # measure accuracy and record loss
            acc1, acc5 = accuracy_CLASSIFICATION(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
        else:
            losses.update(loss.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.classification_dataset == 'imagenet':
            if i % 100 == 0:
                progress.display(i)
                print( "Epoch:", epoch, "|", "["+str(i)+"/"+str(len(train_loader))+"]", "Loss {:.4e}".format(loss.item()), "Acc@1 {:.4f}".format(acc1[0]), " Acc@5 {:.4f}".format(acc5[0]), file=log_writter_CLASSIFICATION)
                log_writter_CLASSIFICATION.flush()
        else:
            if i % 50 == 0:
                print( "Epoch:", epoch, "|", "["+str(i)+"/"+str(len(train_loader))+"]", "Loss {:.5f} ({:.5f}) LR {:.6f}".format(loss.item(),losses.avg,optimizer.state_dict()['param_groups'][0]['lr']))
                print( "Epoch:", epoch, "|", "["+str(i)+"/"+str(len(train_loader))+"]", "Loss {:.5f} ({:.5f})".format(loss.item(),losses.avg), file=log_writter_CLASSIFICATION)
                log_writter_CLASSIFICATION.flush()
    return losses.avg


def evaluate_CLASSIFICATION(val_loader, model, criterion, args, log_writter_CLASSIFICATION):
    batch_time = AverageMeter_CLASSIFICATION('Time', ':6.3f')
    losses = AverageMeter_CLASSIFICATION('Loss', ':.4e')
    top1 = AverageMeter_CLASSIFICATION('Acc@1', ':6.2f')
    top5 = AverageMeter_CLASSIFICATION('Acc@5', ':6.2f')
    progress = ProgressMeter_CLASSIFICATION(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    y_test = torch.FloatTensor().cuda()
    p_test = torch.FloatTensor().cuda()
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # if args.gpu is not None:
            #     images = images.cuda(args.gpu, non_blocking=True)
            # if torch.cuda.is_available():
            #     target = target.cuda(args.gpu, non_blocking=True)

            images, target = images.float().to(device), target.float().to(device) # NIH14 # int for CheXpert

            # compute output
            # output = model(images) ## was active
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                # _, out_classifierHead, _ = model.module.backbone(images)
                out_classifierHead = model.module.backbone[0].extra_features(images)
            else:
                # _, out_classifierHead, _ = model.backbone(images)
                out_classifierHead = model.backbone[0].extra_features(images)
            output = out_classifierHead

            # output = torch.sigmoid( output )

            loss = criterion(output, target)

            if args.classification_dataset == 'imagenet':
                # measure accuracy and record loss
                acc1, acc5 = accuracy_CLASSIFICATION(output, target, topk=(1, 5))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0], images.size(0))
                top5.update(acc5[0], images.size(0))
            else:
                losses.update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            y_test = torch.cat((y_test, target), 0)
            outRes = torch.sigmoid(output)

            # for NIH dataset
            if len(images.size()) == 4:
              bs, c, h, w = images.size()
              n_crops = 1
            elif len(images.size()) == 5:
              bs, n_crops, c, h, w = images.size()
            outMean = outRes.view(bs, n_crops, -1).mean(1) # for NIH dataset
            p_test = torch.cat((p_test, outMean.data), 0) # for NIH dataset


            if args.classification_dataset == 'imagenet':
                if i % 50 == 0:
                    progress.display(i)
            else:
                if i % 10 == 0:
                    print( " - Test:", "["+str(i)+"/"+str(len(val_loader))+"]", "Loss {:.5f}".format(losses.avg))
                    print( " - Test:", "["+str(i)+"/"+str(len(val_loader))+"]", "Loss {:.5f}".format(losses.avg), file=log_writter_CLASSIFICATION)
                    log_writter_CLASSIFICATION.flush()

        if args.classification_dataset == 'imagenet':
            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5), file=log_writter_CLASSIFICATION)
            log_writter_CLASSIFICATION.flush()
            return top1.avg
        else:
            individual_results = metric_AUROC(y_test, p_test, args.numClasses) # for NIH dataset
            individual_results = np.array(individual_results).mean() # for NIH dataset
            print("Validation/Test AUC =", individual_results)
            print("Validation/Test AUC =", individual_results, file=log_writter_CLASSIFICATION)
            return losses.avg

def test_CLASSIFICATION(data_loader_test, model, args):

  model.eval()

  y_test = torch.FloatTensor().cuda()
  p_test = torch.FloatTensor().cuda()

  with torch.no_grad():
    for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
      targets = targets.cuda()
      y_test = torch.cat((y_test, targets), 0)

      if len(samples.size()) == 4:
        bs, c, h, w = samples.size()
        n_crops = 1
      elif len(samples.size()) == 5:
        bs, n_crops, c, h, w = samples.size()

      varInput = torch.autograd.Variable(samples.view(-1, c, h, w).cuda())

      
      if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        # _, out_classifierHead, _ = model.module.backbone(images)
        out_classifierHead = model.module.backbone[0].extra_features(varInput)
      else:
        # _, out_classifierHead, _ = model.backbone(images)
        out_classifierHead = model.backbone[0].extra_features(varInput)
      out = out_classifierHead
      # out = model(varInput)

      if args.classification_dataset == "RSNAPneumonia":
        out = torch.softmax(out,dim = 1)
      else:
        out = torch.sigmoid(out)
      outMean = out.view(bs, n_crops, -1).mean(1)
      p_test = torch.cat((p_test, outMean.data), 0)

  return y_test, p_test


def adjust_learning_rate_CLASSIFICATION(optimizer, epoch, start_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = start_lr * (0.1 ** (epoch // 30))
    return lr

def save_checkpoint_CLASSIFICATION(state, is_best, filename='model'):
    torch.save(state, os.path.join(filename, "checkpoint.pth.tar"))
    if is_best:
        shutil.copyfile(os.path.join(filename, "checkpoint.pth.tar"),
                        os.path.join(filename, "model_best.pth.tar"))




# class ModifiedDetector(nn.Module):
#     def __init__(self, backboneModel ):
#         super().__init__()
#         self.net = backboneModel
#         self.upernet = None
#     def forward(self, x):
#         dataI = x
#         x = self.net(return_loss=False, rescale=True, **x)
        
#         dataI = dataI['img'][0].cuda()
#         backbone_features = self.net.backbone(dataI)
#         return x, backbone_features
















def main(args):
    utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, "config_cfg.py")
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # update some new args temporally
    if not getattr(args, 'use_ema', None):
        args.use_ema = False
    if not getattr(args, 'debug', None):
        args.debug = False

    # setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("Command: "+' '.join(sys.argv))
    if args.rank == 0:
        save_json_path = os.path.join(args.output_dir, "config_args_all.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(save_json_path))
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')


    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)





    ### ----------------- BUILD MODEL ----------------- ###
    if args.taskcomponent == "detect_segmentation_cyclic" or args.taskcomponent == "detect_segmentation_cyclic_v2":
        args.num_classes = 3+1
        args.dn_labelbook_size = 5
    if args.taskcomponent in ["detect_segmentation_cyclic_v3", "detect_segmentation_cyclic_v4", "detect_vindrcxr_heart_segTest", "detect_vindrcxr_heart", "detect_vindrcxr_leftlung", "detect_vindrcxr_rightlung"]: # 
        # args.num_classes = 1+1
        # args.dn_labelbook_size = 3
        args.num_classes = 3+1
        args.dn_labelbook_size = 5
    if args.taskcomponent in ["segmentation"]:
        args.num_classes = 3+1
        args.dn_labelbook_size = 5


    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

    ## EMA
    if args.modelEMA == "True_Epoch" or args.modelEMA == "True_Iteration":
        # model_ema, criterion_ema, postprocessors_ema = build_model_main(args)
        model_ema = copy.deepcopy(model)
        criterion_ema = copy.deepcopy(criterion)
        postprocessors_ema = copy.deepcopy(postprocessors)
        criterionMSE = torch.nn.MSELoss()
        print("[Model Info] Using EMA CPU Model as Teacher Model.")
        # if args.distributed:
            # model_ema = torch.nn.DataParallel(model_ema) ## model ark6 weights loading issue
            # model_ema = torch.nn.parallel.DistributedDataParallel(model_ema, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params) ## didn't work
        model_ema = model_ema.cuda() ## working
        for p in model_ema.parameters():
            p.requires_grad = False
    else:
        model_ema = None
        criterion_ema = None
        postprocessors_ema = None

    # ema
    if args.use_ema:
        ema_m = ModelEma(model, args.ema_decay)
    else:
        ema_m = None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    # logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))




    ### ----------------- Optimizer ----------------- ###
    if args.taskcomponent in ["detect_segmentation_cyclic", "detect_segmentation_cyclic_v2", "detect_segmentation_cyclic_v3", "detect_segmentation_cyclic_v4", "segmentation", 'detect_vindrcxr_heart_segTest', 'detect_vindrcxr_heart', 'detect_vindrcxr_leftlung', 'detect_vindrcxr_rightlung']:
        loss_scaler = NativeScaler()
        # args.lr_backbone = args.lr # 0.0001
        param_dicts = get_param_dict(args, model_without_ddp)
        if args.opt == "adamw":
            # optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
            optimizer = torch.optim.AdamW([
                {'params': [param for name, param in model.named_parameters() if 'backbone' in name and 'segmentation_' not in name], 'lr': args.lr_backbone},##Backbone
                {'params': [param for name, param in model.named_parameters() if 'segmentation_' in name], 'lr': args.lr_segmentor},##Segmentor
                {'params': [param for name, param in model.named_parameters() if 'transformer' in name], 'lr': args.lr},##Localizer
            ], lr=args.lr, weight_decay=args.weight_decay)

            args.lr_drop = 15
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
            print("[Info.] Lr Scheduler:", "StepLR", "Drop:", args.lr_drop)
        elif args.opt == "sgd":
            # optimizer = torch.optim.SGD(param_dicts, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=False)
            ### Backbone -- Segmentor -- Localizer
            optimizer = torch.optim.SGD([
                {'params': [param for name, param in model.named_parameters() if 'backbone' in name and 'segmentation_' not in name], 'lr': args.lr_backbone},##Backbone
                {'params': [param for name, param in model.named_parameters() if 'segmentation_' in name], 'lr': args.lr_segmentor},##Segmentor
                {'params': [param for name, param in model.named_parameters() if 'transformer' in name], 'lr': args.lr},##Localizer
            ], lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            args.lr_drop = 20

        # args.num_classes = 3
        # args.dn_labelbook_size = 4

    # if args.taskcomponent == "segmentation" or args.taskcomponent == "segmentation_cyclic":
    if args.taskcomponent == "segmentation_cyclic":
        loss_scaler = NativeScaler()
        if args.opt == "adamw":
            # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer = torch.optim.AdamW([
                {'params': [param for name, param in model.named_parameters() if 'backbone' in name and 'segmentation_' not in name], 'lr': args.lr_backbone},##Backbone
                {'params': [param for name, param in model.named_parameters() if 'segmentation_' in name], 'lr': args.lr_segmentor},##Segmentor
                {'params': [param for name, param in model.named_parameters() if 'transformer' in name], 'lr': args.lr},##Localizer
            ], lr=args.lr, weight_decay=args.weight_decay)

            args.lr_drop = 15
        elif args.opt == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=False)

    if args.taskcomponent in ['detection', 'detection_baseline']:
        # args.lr_backbone = args.lr
        # args.lr_backbone = 1e-5 # 0.0001
        param_dicts = get_param_dict(args, model_without_ddp)

        if args.opt == "adamw":
            # optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
            optimizer = torch.optim.AdamW([
                {'params': [param for name, param in model.named_parameters() if 'backbone' in name and 'segmentation_' not in name], 'lr': args.lr_backbone},##Backbone
                {'params': [param for name, param in model.named_parameters() if 'segmentation_' in name], 'lr': args.lr_segmentor},##Segmentor
                {'params': [param for name, param in model.named_parameters() if 'transformer' in name], 'lr': args.lr},##Localizer
            ], lr=args.lr, weight_decay=args.weight_decay)
            args.lr_drop = 20
        elif args.opt == "sgd":
            optimizer = torch.optim.SGD(param_dicts, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=False)
            args.lr_drop = 20

    if args.taskcomponent == 'classification':
        if args.opt == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        elif args.opt == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)

        if args.classification_dataset == 'imagenet':
            optimizer = create_optimizer(args, model)
            criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        if args.classification_dataset == 'ChestXray14':
            optimizer = create_optimizer(args, model)
            lr_scheduler, _ = create_scheduler(args, optimizer)
            loss_scaler = NativeScaler()
            criterion = torch.nn.BCEWithLogitsLoss()

    print("[Information.] Optimizer:", args.opt)


    # if args.resume:
    #     model_without_ddp = load_weights(model_without_ddp, args)
    #     if args.use_ema:
    #         if 'ema_model' in checkpoint:
    #             ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
    #         else:
    #             del ema_m
    #             ema_m = ModelEma(model, args.ema_decay)                

    #     if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #         args.start_epoch = checkpoint['epoch'] + 1

    if (not args.resume) and args.pretrain_model_path:
        checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')['model']
        from collections import OrderedDict
        _ignorekeywordlist = args.finetune_ignore if args.finetune_ignore else []
        ignorelist = []

        def check_keep(keyname, ignorekeywordlist):
            for keyword in ignorekeywordlist:
                if keyword in keyname:
                    ignorelist.append(keyname)
                    return False
            return True
        logger.info("Ignore keys: {}".format(json.dumps(ignorelist, indent=2)))
        _tmp_st = OrderedDict({k:v for k, v in utils.clean_state_dict(checkpoint).items() if check_keep(k, _ignorekeywordlist)})
        _load_output = model_without_ddp.load_state_dict(_tmp_st, strict=False)
        logger.info(str(_load_output))
        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)  

    with open(args.output_dir + '/model_param.txt', 'w') as file:
        for name, param in model.named_parameters():
            file.write(name + '\n')
    file.close()
    del file




    if args.taskcomponent in ['detection', 'detection_baseline']:
        logs_path = os.path.join(args.output_dir, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if os.path.exists(os.path.join(logs_path, "log.txt")):
            log_writter_DETECTION = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            log_writter_DETECTION = open(os.path.join(logs_path, "log.txt"), 'w')

        print()
        print("-------------")
        print("[Information]  TASK:", args.taskcomponent)
        print("[Information]  Backbone:", args.backbonemodel) 
        print("[Information]  Backbone_INIT:", args.init)
        print("[Information]  Backbone Weights:", args.backbone_dir)
        print("[Information]  Dataset:", args.dataset_file)
        print("[Information]  Total Epoch:", args.total_epochs)
        print("[Information]  Batch Size:", args.batch_size)
        print("[Information]  Learning Rate:", args.lr)
        print("[Information]  Learning Rate Backbone:", args.lr_backbone)
        print("[Information]  Num Workers:", args.num_workers)
        print("[Information]  Optimizer:", args.opt)
        print("[Information]  Output Dir:", args.output_dir)
        print("-------------")
        print()

        print("-------------", file=log_writter_DETECTION)
        print("[Information]  TASK:", args.taskcomponent, file=log_writter_DETECTION)
        print("[Information]  Backbone:", args.backbonemodel, file=log_writter_DETECTION)
        print("[Information]  Backbone_INIT:", args.init, file=log_writter_DETECTION)
        print("[Information]  Backbone Weights:", args.backbone_dir, file=log_writter_DETECTION)
        print("[Information]  Dataset:", args.dataset_file, file=log_writter_DETECTION)
        print("[Information]  Total Epoch:", args.total_epochs, file=log_writter_DETECTION)
        print("[Information]  Batch Size:", args.batch_size, file=log_writter_DETECTION)
        print("[Information]  Learning Rate:", args.lr, file=log_writter_DETECTION)
        print("[Information]  Learning Rate Backbone:", args.lr_backbone, file=log_writter_DETECTION)
        print("[Information]  Num Workers:", args.num_workers, file=log_writter_DETECTION)
        print("[Information]  Optimizer:", args.opt, file=log_writter_DETECTION)
        print("[Information]  Output Dir:", args.output_dir, file=log_writter_DETECTION)
        print("-------------", file=log_writter_DETECTION)

        
        if args.backbone_dir is not None:
            model = load_weights(model, args)

        data_loader_train, data_loader_val, sampler_train, dataset_val = dataloader_return(args)
        base_ds = get_coco_api_from_dataset(dataset_val)

        if args.onecyclelr:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.total_epochs, pct_start=0.2)
        elif args.multi_step_lr:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
        else:
            print("[Info.] Lr Scheduler:", "StepLR", "Drop:", args.lr_drop)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


        if args.dataset_file == "coco_panoptic":
            # We also evaluate AP during panoptic training, on original coco DS
            coco_val = datasets.coco.build("val", args)
            base_ds = get_coco_api_from_dataset(coco_val)
        else:
            base_ds = get_coco_api_from_dataset(dataset_val)
        if args.frozen_weights is not None:
            checkpoint = torch.load(args.frozen_weights, map_location='cpu')
            model_without_ddp.detr.load_state_dict(checkpoint['model'])
        output_dir = Path(args.output_dir)
        if os.path.exists(os.path.join(args.output_dir, 'checkpoint.pth')):
            args.resume = os.path.join(args.output_dir, 'checkpoint.pth')


    if args.taskcomponent == 'detection' and args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        print("[Info.] Detection Evaluation on:", args.dataset_file)
        test_stats, coco_evaluator, _ = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")

        log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        return

    ## Added by Nahid
    if args.taskcomponent == 'detection' and args.test:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats = test_NAD(model, criterion, postprocessors,
                                              data_loader_test, base_ds, device, args.output_dir, wo_class_error=wo_class_error, args=args)
        print("Testing Done...")
        # log_stats = {**{f'test_{k}': v for k, v in test_stats.items()} }
        # if args.output_dir and utils.is_main_process():
        #     with (output_dir / "log.txt").open("a") as f:
        #         f.write(json.dumps(log_stats) + "\n")
        return


    if args.taskcomponent in ['detect_vindrcxr_heart', "detect_vindrcxr_leftlung", "detect_vindrcxr_rightlung"]:
        logs_path = os.path.join(args.output_dir, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if os.path.exists(os.path.join(logs_path, "log.txt")):
            log_writter_DETECTION = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            log_writter_DETECTION = open(os.path.join(logs_path, "log.txt"), 'w')

        logs_path = os.path.join(args.output_dir, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if os.path.exists(os.path.join(logs_path, "log_seg.txt")):
            log_writter_SEGMENTATION = open(os.path.join(logs_path, "log_seg.txt"), 'a')
        else:
            log_writter_SEGMENTATION = open(os.path.join(logs_path, "log_seg.txt"), 'w')

        print()
        print("-------------")
        print("[Information]  TASK:", args.taskcomponent)
        print("[Information]  Backbone:", args.backbonemodel) 
        print("[Information]  Backbone_INIT:", args.init)
        print("[Information]  Backbone Weights:", args.backbone_dir)
        print("[Information]  Dataset:", args.dataset_file)
        print("[Information]  Total Epoch:", args.total_epochs)
        print("[Information]  Batch Size:", args.batch_size)
        print("[Information]  Learning Rate:", args.lr)
        print("[Information]  Learning Rate Backbone:", args.lr_backbone)
        print("[Information]  Num Workers:", args.num_workers)
        print("[Information]  Optimizer:", args.opt)
        print("[Information]  Output Dir:", args.output_dir)
        print("-------------")
        print()

        print("-------------", file=log_writter_DETECTION)
        print("[Information]  TASK:", args.taskcomponent, file=log_writter_DETECTION)
        print("[Information]  Backbone:", args.backbonemodel, file=log_writter_DETECTION)
        print("[Information]  Backbone_INIT:", args.init, file=log_writter_DETECTION)
        print("[Information]  Backbone Weights:", args.backbone_dir, file=log_writter_DETECTION)
        print("[Information]  Dataset:", args.dataset_file, file=log_writter_DETECTION)
        print("[Information]  Total Epoch:", args.total_epochs, file=log_writter_DETECTION)
        print("[Information]  Batch Size:", args.batch_size, file=log_writter_DETECTION)
        print("[Information]  Learning Rate:", args.lr, file=log_writter_DETECTION)
        print("[Information]  Learning Rate Backbone:", args.lr_backbone, file=log_writter_DETECTION)
        print("[Information]  Num Workers:", args.num_workers, file=log_writter_DETECTION)
        print("[Information]  Optimizer:", args.opt, file=log_writter_DETECTION)
        print("[Information]  Output Dir:", args.output_dir, file=log_writter_DETECTION)
        print("-------------", file=log_writter_DETECTION)

        
        if args.backbone_dir is not None:
            model = load_weights(model, args)
        elif args.resume is not None:
            model = load_weights(model, args)

        data_loader_train, data_loader_val, sampler_train, dataset_val  = dataloader_return(args)
        base_ds = get_coco_api_from_dataset(dataset_val)



    if args.taskcomponent == 'detect_vindrcxr_heart_segTest': # detect_vindrcxr_heart
        logs_path = os.path.join(args.output_dir, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if os.path.exists(os.path.join(logs_path, "log.txt")):
            log_writter_DETECTION = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            log_writter_DETECTION = open(os.path.join(logs_path, "log.txt"), 'w')

        logs_path = os.path.join(args.output_dir, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if os.path.exists(os.path.join(logs_path, "log_seg.txt")):
            log_writter_SEGMENTATION = open(os.path.join(logs_path, "log_seg.txt"), 'a')
        else:
            log_writter_SEGMENTATION = open(os.path.join(logs_path, "log_seg.txt"), 'w')

        print()
        print("-------------")
        print("[Information]  TASK:", "Detection - detect_vindrcxr_heart_segTest")
        print("[Information]  Backbone:", args.backbonemodel) 
        print("[Information]  Backbone_INIT:", args.init)
        print("[Information]  Backbone Weights:", args.backbone_dir)
        print("[Information]  Dataset:", args.dataset_file)
        print("[Information]  Total Epoch:", args.total_epochs)
        print("[Information]  Batch Size:", args.batch_size)
        print("[Information]  Learning Rate:", args.lr)
        print("[Information]  Learning Rate Backbone:", args.lr_backbone)
        print("[Information]  Num Workers:", args.num_workers)
        print("[Information]  Optimizer:", args.opt)
        print("[Information]  Output Dir:", args.output_dir)
        print("-------------")
        print()

        print("-------------", file=log_writter_DETECTION)
        print("[Information]  TASK:", "Detection", file=log_writter_DETECTION)
        print("[Information]  Backbone:", args.backbonemodel, file=log_writter_DETECTION)
        print("[Information]  Backbone_INIT:", args.init, file=log_writter_DETECTION)
        print("[Information]  Backbone Weights:", args.backbone_dir, file=log_writter_DETECTION)
        print("[Information]  Dataset:", args.dataset_file, file=log_writter_DETECTION)
        print("[Information]  Total Epoch:", args.total_epochs, file=log_writter_DETECTION)
        print("[Information]  Batch Size:", args.batch_size, file=log_writter_DETECTION)
        print("[Information]  Learning Rate:", args.lr, file=log_writter_DETECTION)
        print("[Information]  Learning Rate Backbone:", args.lr_backbone, file=log_writter_DETECTION)
        print("[Information]  Num Workers:", args.num_workers, file=log_writter_DETECTION)
        print("[Information]  Optimizer:", args.opt, file=log_writter_DETECTION)
        print("[Information]  Output Dir:", args.output_dir, file=log_writter_DETECTION)
        print("-------------", file=log_writter_DETECTION)

        
        if args.backbone_dir is not None:
            model = load_weights(model, args)
        elif args.resume is not None:
            model = load_weights(model, args)

        data_loader_train, data_loader_val, sampler_train, dataset_val, seg_val_loader  = dataloader_return(args)
        base_ds = get_coco_api_from_dataset(dataset_val)



    if args.taskcomponent == 'segmentation' or args.taskcomponent == 'segmentation_cyclic':
        if args.backbone_dir is not None:
            model = load_weights(model, args)

        model_path_SEGMENTATION = args.output_dir # /data/jliang12/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/SOMETHING
        if not os.path.exists(model_path_SEGMENTATION):
            os.makedirs(model_path_SEGMENTATION)

        logs_path = os.path.join(model_path_SEGMENTATION, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if os.path.exists(os.path.join(logs_path, "log.txt")):
            log_writter_SEGMENTATION = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            log_writter_SEGMENTATION = open(os.path.join(logs_path, "log.txt"), 'w')

        args.total_epochs = args.total_epochs
        img_size = args.imgsize # 448 worked
        batch_size = args.batch_size
        num_workers = args.num_workers
        best_val_loss_SEGMENTATION = 100000
        patience_SEGMENTATION = 50
        patience_counter_SEGMENTATION = 0

        print()
        print("-------------")
        print("[Information]  TASK:", "Segmentation")
        print("[Information]  Backbone:", args.backbonemodel) 
        print("[Information]  Backbone_INIT:", args.init)
        print("[Information]  Backbone Weights:", args.backbone_dir)
        print("[Information]  Dataset:", args.segmentation_dataset)
        print("[Information]  Total Epoch:", args.total_epochs)
        print("[Information]  Image Size:", img_size)
        print("[Information]  Batch Size:", batch_size)
        print("[Information]  Learning Rate:", args.lr)
        print("[Information]  Num Workers:", num_workers)
        print("[Information]  Optimizer:", args.opt)
        print("[Information]  Patience:", patience_SEGMENTATION)
        print("[Information]  Output Dir:", args.output_dir)
        print("-------------")
        print() # log_writter_SEGMENTATION

        print("-------------", file=log_writter_SEGMENTATION)
        print("[Information]  TASK:", "Segmentation", file=log_writter_SEGMENTATION)
        print("[Information]  Backbone:", args.backbonemodel, file=log_writter_SEGMENTATION)
        print("[Information]  Backbone_INIT:", args.init, file=log_writter_SEGMENTATION)
        print("[Information]  Backbone Weights:", args.backbone_dir, file=log_writter_SEGMENTATION)
        print("[Information]  Dataset:", args.segmentation_dataset, file=log_writter_SEGMENTATION)
        print("[Information]  Total Epoch:", args.total_epochs, file=log_writter_SEGMENTATION)
        print("[Information]  Image Size:", img_size, file=log_writter_SEGMENTATION)
        print("[Information]  Batch Size:", batch_size, file=log_writter_SEGMENTATION)
        print("[Information]  Learning Rate:", args.lr, file=log_writter_SEGMENTATION)
        print("[Information]  Num Workers:", num_workers, file=log_writter_SEGMENTATION)
        print("[Information]  Optimizer:", args.opt, file=log_writter_SEGMENTATION)
        print("[Information]  Patience:", patience_SEGMENTATION, file=log_writter_SEGMENTATION)
        print("[Information]  Output Dir:", args.output_dir, file=log_writter_SEGMENTATION)
        print("-------------", file=log_writter_SEGMENTATION)


        if args.segmentation_dataset == 'jsrt_lung':
            train_loader, val_loader, test_loader = dataloader_return(args)
        elif args.segmentation_dataset == 'jsrt_clavicle':
            train_loader, val_loader, test_loader = dataloader_return(args)
        elif args.segmentation_dataset == 'jsrt_heart':
            train_loader, val_loader, test_loader = dataloader_return(args)
        elif args.segmentation_dataset == 'chestxdetdataset': # Disease Segmentation
            train_loader, test_loader = dataloader_return(args)
        elif args.segmentation_dataset == 'jsrt_lung_heart_clavicle': # For cyclic
            train_loader_jsrtLung, val_loader_jsrtLung, test_loader_jsrtLung, train_loader_jsrtClavicle, val_loader_jsrtClavicle, test_loader_jsrtClavicle, train_loader_jsrtHeart, val_loader_jsrtHeart, test_loader_jsrtHeart = dataloader_return(args)
        elif args.segmentation_dataset == 'vindrcxr_lung':
            train_loader, val_loader = dataloader_return(args)
        elif args.segmentation_dataset == 'vindrcxr_heart':
            train_loader, val_loader = dataloader_return(args)
        elif args.segmentation_dataset == 'vindrcxr_lung_heart':
            train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLung, val_loader_vindrcxrtLung = dataloader_return(args)
        loss_scaler = NativeScaler()

        if args.test:
            print()
            print("[CHECK-Testing] Segmentation Model.")
            checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')
            checkpoint_model = checkpoint['model']
            # checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
            model.load_state_dict(checkpoint_model)
            # print()
            print("[MODEL INFO.] Pretrained model loaded for Segmentation...")

            test_y, test_p = test_SEGMENTATION( model, test_loader, log_writter_SEGMENTATION=log_writter_SEGMENTATION )
            print("[INFO] Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
            print("Mean Dice = {:.4f}".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter_SEGMENTATION)

            print("[INFO] Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)))
            print("Mean Dice = {:.4f}".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)))

            print("Done testing iteration!", file=log_writter_SEGMENTATION)
            log_writter_SEGMENTATION.flush()

            exit(0)



    if args.taskcomponent == 'classification':
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if args.backbone_dir is not None:
            model = load_weights(model, args)

        model_path_CLASSIFICATION = args.output_dir

        logs_path = os.path.join(model_path_CLASSIFICATION, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if os.path.exists(os.path.join(logs_path, "log.txt")):
            log_writter_CLASSIFICATION = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            log_writter_CLASSIFICATION = open(os.path.join(logs_path, "log.txt"), 'w')

        print()
        print("-------------")
        print("[Information]  TASK:", "Classification")
        print("[Information]  Backbone:", args.backbonemodel)
        print("[Information]  Backbone_INIT:", args.init)
        print("[Information]  Backbone Weights:", args.backbone_dir)
        print("[Information]  Dataset:", args.classification_dataset)
        print("[Information]  Total Epoch:", args.total_epochs)
        print("[Information]  Image Size:", img_size)
        print("[Information]  Batch Size:", batch_size)
        print("[Information]  Learning Rate:", args.lr)
        print("[Information]  Num Workers:", num_workers)
        print("[Information]  Optimizer:", "SGD")
        print("[Information]  Output Dir:", args.output_dir)
        print("-------------")
        print() # file=log_writter_SEGMENTATION

        print("-------------", file=log_writter_CLASSIFICATION)
        print("[Information]  TASK:", "Classification", file=log_writter_CLASSIFICATION)
        print("[Information]  Backbone:", args.backbonemodel, file=log_writter_CLASSIFICATION)
        print("[Information]  Backbone_INIT:", args.init, file=log_writter_CLASSIFICATION)
        print("[Information]  Backbone Weights:", args.backbone_dir, file=log_writter_CLASSIFICATION)
        print("[Information]  Dataset:", args.classification_dataset, file=log_writter_CLASSIFICATION)
        print("[Information]  Total Epoch:", args.total_epochs, file=log_writter_CLASSIFICATION)
        print("[Information]  Image Size:", img_size, file=log_writter_CLASSIFICATION)
        print("[Information]  Batch Size:", batch_size, file=log_writter_CLASSIFICATION)
        print("[Information]  Learning Rate:", args.lr, file=log_writter_CLASSIFICATION)
        print("[Information]  Num Workers:", num_workers, file=log_writter_CLASSIFICATION)
        print("[Information]  Optimizer:", "SGD", file=log_writter_CLASSIFICATION)
        print("[Information]  Output Dir:", args.output_dir, file=log_writter_CLASSIFICATION)
        print("-------------", file=log_writter_CLASSIFICATION)


        if args.classification_dataset == "imagenet":
            args.total_epochs = args.total_epochs
            img_size = args.imgsize # 448 worked
            batch_size = args.batch_size # 128
            num_workers = args.num_workers # 16 | 32
            best_acc1_CLASSIFICATION = 100000

            train_loader, val_loader = dataloader_return(args)
            print("[INFO.] Classification Data loaded...")
            if args.test:
                print()
                print("Classification Validation Started...")
                acc1 = evaluate_CLASSIFICATION(val_loader, model, criterion, args, log_writter_CLASSIFICATION)
                log_writter_CLASSIFICATION.flush()
                exit(0)


        elif args.classification_dataset == "ChestXray14":
            args.total_epochs = 200
            img_size = args.imgsize # 448 worked
            batch_size = args.batch_size # 128
            num_workers = args.num_workers # 16 | 32
            # best_acc1_CLASSIFICATION = 100000
            best_val_CLASSIFICATION = 10000

            patience_counter_CLASSIFICATION = 0
            patience_CLASSIFICATION = 35

            train_loader, val_loader, test_loader = dataloader_return(args)

            # Only Testing Classification for Medical Images
            if args.test:
                checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')
                checkpoint_model = checkpoint['model']
                model.load_state_dict(checkpoint_model, strict=False)
                print("[MODEL INFO.] Pretrained model loaded for Classification...")

                y_test, p_test = test_CLASSIFICATION(test_loader, model, args)
                individual_results = metric_AUROC(y_test, p_test, args.numClasses)
                print(">>{}: AUC = {}".format(" TEST ", np.array2string(np.array(individual_results), precision=4, separator=',')))
                individual_results = np.array(individual_results).mean()
                print("Validation/Test mean AUC =", individual_results)
                exit(0)


    if args.taskcomponent == "detect_segmentation_cyclic":
        if args.backbone_dir is not None:
            model = load_weights(model, args)

        model_path_SEGMENTATION = args.output_dir
        if not os.path.exists(model_path_SEGMENTATION):
            os.makedirs(model_path_SEGMENTATION)
        logs_path = os.path.join(model_path_SEGMENTATION, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if os.path.exists(os.path.join(logs_path, "log.txt")):
            log_writter_SEGMENTATION = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            log_writter_SEGMENTATION = open(os.path.join(logs_path, "log.txt"), 'w')


        args.total_epochs = args.total_epochs
        img_size = args.imgsize # 448 worked
        batch_size = args.batch_size
        num_workers = args.num_workers
        best_val_loss_SEGMENTATION = 100000
        patience_SEGMENTATION = 50
        patience_counter_SEGMENTATION = 0

        print()
        print("-------------")
        print("[Information]  TASK:", "Localization_Segmentation_Cyclic")
        print("[Information]  Backbone:", args.backbonemodel) 
        print("[Information]  Backbone_INIT:", args.init)
        print("[Information]  Backbone Weights:", args.backbone_dir)
        print("[Information]  Dataset:", args.segmentation_dataset)
        print("[Information]  Total Epoch:", args.total_epochs)
        print("[Information]  Image Size:", img_size)
        print("[Information]  Batch Size:", batch_size)
        print("[Information]  Learning Rate:", args.lr)
        print("[Information]  Learning Rate Backbone:", args.lr_backbone)
        print("[Information]  Num Workers:", num_workers)
        print("[Information]  Optimizer:", args.opt)
        # print("[Information]  Patience:", patience_SEGMENTATION)
        print("[Information]  Output Dir:", args.output_dir)
        print("-------------")
        print() # log_writter_SEGMENTATION

        print("-------------", file=log_writter_SEGMENTATION)
        print("[Information]  TASK:", "Segmentation", file=log_writter_SEGMENTATION)
        print("[Information]  Backbone:", args.backbonemodel, file=log_writter_SEGMENTATION)
        print("[Information]  Backbone_INIT:", args.init, file=log_writter_SEGMENTATION)
        print("[Information]  Backbone Weights:", args.backbone_dir, file=log_writter_SEGMENTATION)
        print("[Information]  Dataset:", args.segmentation_dataset, file=log_writter_SEGMENTATION)
        print("[Information]  Total Epoch:", args.total_epochs, file=log_writter_SEGMENTATION)
        print("[Information]  Image Size:", img_size, file=log_writter_SEGMENTATION)
        print("[Information]  Batch Size:", batch_size, file=log_writter_SEGMENTATION)
        print("[Information]  Learning Rate:", args.lr, file=log_writter_SEGMENTATION)
        print("[Information]  Learning Rate Backbone:", args.lr_backbone, file=log_writter_SEGMENTATION)
        print("[Information]  Num Workers:", num_workers, file=log_writter_SEGMENTATION)
        print("[Information]  Optimizer:", args.opt, file=log_writter_SEGMENTATION)
        # print("[Information]  Patience:", patience_SEGMENTATION, file=log_writter_SEGMENTATION)
        print("[Information]  Output Dir:", args.output_dir, file=log_writter_SEGMENTATION)
        print("-------------", file=log_writter_SEGMENTATION)

        loss_scaler = NativeScaler()
        # train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLung, val_loader_vindrcxrtLung, data_loader_train_Heart, data_loader_val_Heart, data_loader_train_Lung, data_loader_val_Lung, sampler_train, dataset_val = dataloader_return(args)
        train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLung, val_loader_vindrcxrtLung, data_loader_train, data_loader_val, sampler_train, dataset_val = dataloader_return(args)
        base_ds = get_coco_api_from_dataset(dataset_val)



    if args.taskcomponent == "detect_segmentation_cyclic_v2": # detect_segmentation_cyclic_v3
        if args.backbone_dir is not None:
            model = load_weights(model, args)

        model_path_SEGMENTATION = args.output_dir
        if not os.path.exists(model_path_SEGMENTATION):
            os.makedirs(model_path_SEGMENTATION)
        logs_path = os.path.join(model_path_SEGMENTATION, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if os.path.exists(os.path.join(logs_path, "log.txt")):
            log_writter_SEGMENTATION = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            log_writter_SEGMENTATION = open(os.path.join(logs_path, "log.txt"), 'w')


        args.total_epochs = args.total_epochs
        img_size = args.imgsize # 448 worked
        batch_size = args.batch_size
        num_workers = args.num_workers
        best_val_loss_SEGMENTATION = 100000
        patience_SEGMENTATION = 50
        patience_counter_SEGMENTATION = 0

        print()
        print("-------------")
        print("[Information]  TASK:", "detect_segmentation_cyclic_v2")
        print("[Information]  Backbone:", args.backbonemodel) 
        print("[Information]  Backbone_INIT:", args.init)
        print("[Information]  Backbone Weights:", args.backbone_dir)
        print("[Information]  Dataset:", args.segmentation_dataset)
        print("[Information]  Total Epoch:", args.total_epochs)
        print("[Information]  Image Size:", img_size)
        print("[Information]  Batch Size:", batch_size)
        print("[Information]  Learning Rate:", args.lr)
        print("[Information]  Learning Rate Backbone:", args.lr_backbone)
        print("[Information]  Num Workers:", num_workers)
        print("[Information]  Optimizer:", args.opt)
        # print("[Information]  Patience:", patience_SEGMENTATION)
        print("[Information]  Output Dir:", args.output_dir)
        print("-------------")
        print() # log_writter_SEGMENTATION

        print("-------------", file=log_writter_SEGMENTATION)
        print("[Information]  TASK:", "Segmentation", file=log_writter_SEGMENTATION)
        print("[Information]  Backbone:", args.backbonemodel, file=log_writter_SEGMENTATION)
        print("[Information]  Backbone_INIT:", args.init, file=log_writter_SEGMENTATION)
        print("[Information]  Backbone Weights:", args.backbone_dir, file=log_writter_SEGMENTATION)
        print("[Information]  Dataset:", args.segmentation_dataset, file=log_writter_SEGMENTATION)
        print("[Information]  Total Epoch:", args.total_epochs, file=log_writter_SEGMENTATION)
        print("[Information]  Image Size:", img_size, file=log_writter_SEGMENTATION)
        print("[Information]  Batch Size:", batch_size, file=log_writter_SEGMENTATION)
        print("[Information]  Learning Rate:", args.lr, file=log_writter_SEGMENTATION)
        print("[Information]  Learning Rate Backbone:", args.lr_backbone, file=log_writter_SEGMENTATION)
        print("[Information]  Num Workers:", num_workers, file=log_writter_SEGMENTATION)
        print("[Information]  Optimizer:", args.opt, file=log_writter_SEGMENTATION)
        # print("[Information]  Patience:", patience_SEGMENTATION, file=log_writter_SEGMENTATION)
        print("[Information]  Output Dir:", args.output_dir, file=log_writter_SEGMENTATION)
        print("-------------", file=log_writter_SEGMENTATION)

        loss_scaler = NativeScaler()
        train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLeftLung, val_loader_vindrcxrtLeftLung, train_loader_vindrcxrRightLung, val_loader_vindrcxrtRightLung, data_loader_train, data_loader_val, sampler_train, dataset_val = dataloader_return(args)
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.taskcomponent == "detect_segmentation_cyclic_v3":
        if args.backbone_dir is not None: # model_ema
            model = load_weights(model, args)

        model_path_SEGMENTATION = args.output_dir
        if not os.path.exists(model_path_SEGMENTATION):
            os.makedirs(model_path_SEGMENTATION)
        logs_path = os.path.join(model_path_SEGMENTATION, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if os.path.exists(os.path.join(logs_path, "log.txt")):
            log_writter_SEGMENTATION = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            log_writter_SEGMENTATION = open(os.path.join(logs_path, "log.txt"), 'w')


        args.total_epochs = args.total_epochs
        img_size = args.imgsize # 448 worked
        batch_size = args.batch_size
        num_workers = args.num_workers
        best_val_loss_SEGMENTATION = 100000
        patience_SEGMENTATION = 50
        patience_counter_SEGMENTATION = 0

        print()
        print("-------------")
        print("[Information]  TASK:", args.taskcomponent)
        print("[Information]  Backbone:", args.backbonemodel) 
        print("[Information]  Backbone_INIT:", args.init)
        print("[Information]  Backbone Weights:", args.backbone_dir)
        print("[Information]  Dataset:", args.segmentation_dataset)
        print("[Information]  Total Epoch:", args.total_epochs)
        print("[Information]  Image Size:", img_size)
        print("[Information]  Batch Size:", batch_size)
        print("[Information]  Learning Rate:", args.lr)
        print("[Information]  Learning Rate Backbone:", args.lr_backbone)
        print("[Information]  Num Workers:", num_workers)
        print("[Information]  Optimizer:", args.opt)
        # print("[Information]  Patience:", patience_SEGMENTATION)
        print("[Information]  Output Dir:", args.output_dir)
        print("-------------")
        print() # log_writter_SEGMENTATION

        print("-------------", file=log_writter_SEGMENTATION)
        print("[Information]  TASK:", "Segmentation", file=log_writter_SEGMENTATION)
        print("[Information]  Backbone:", args.backbonemodel, file=log_writter_SEGMENTATION)
        print("[Information]  Backbone_INIT:", args.init, file=log_writter_SEGMENTATION)
        print("[Information]  Backbone Weights:", args.backbone_dir, file=log_writter_SEGMENTATION)
        print("[Information]  Dataset:", args.segmentation_dataset, file=log_writter_SEGMENTATION)
        print("[Information]  Total Epoch:", args.total_epochs, file=log_writter_SEGMENTATION)
        print("[Information]  Image Size:", img_size, file=log_writter_SEGMENTATION)
        print("[Information]  Batch Size:", batch_size, file=log_writter_SEGMENTATION)
        print("[Information]  Learning Rate:", args.lr, file=log_writter_SEGMENTATION)
        print("[Information]  Learning Rate Backbone:", args.lr_backbone, file=log_writter_SEGMENTATION)
        print("[Information]  Num Workers:", num_workers, file=log_writter_SEGMENTATION)
        print("[Information]  Optimizer:", args.opt, file=log_writter_SEGMENTATION)
        # print("[Information]  Patience:", patience_SEGMENTATION, file=log_writter_SEGMENTATION)
        print("[Information]  Output Dir:", args.output_dir, file=log_writter_SEGMENTATION)
        print("-------------", file=log_writter_SEGMENTATION)

        loss_scaler = NativeScaler()
        # train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLeftLung, val_loader_vindrcxrtLeftLung, train_loader_vindrcxrRightLung, val_loader_vindrcxrtRightLung, data_loader_trainHeart, data_loader_valHeart, sampler_trainHeart, dataset_valHeart, data_loader_trainLeftLung, data_loader_valLeftLung, sampler_trainLeftLung, dataset_valLeftLung, data_loader_trainRightLung, data_loader_valRightLung, sampler_trainRightLung, dataset_valRightLung = dataloader_return(args)
        # base_ds_Heart = get_coco_api_from_dataset(dataset_valHeart)
        # base_ds_LeftLung = get_coco_api_from_dataset(dataset_valLeftLung)
        # base_ds_RightLung = get_coco_api_from_dataset(dataset_valRightLung)
        # del dataset_valHeart, dataset_valLeftLung, dataset_valRightLung

        train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLeftLung, val_loader_vindrcxrtLeftLung, train_loader_vindrcxrRightLung, val_loader_vindrcxrtRightLung, \
            data_loader_train, sampler_train, data_loader_val, dataset_val, \
            data_loader_valHeart, dataset_valHeart, dataset_val, \
            data_loader_valLeftLung, dataset_valLeftLung, \
            data_loader_valRightLung, dataset_valRightLung = dataloader_return(args)
        base_ds = get_coco_api_from_dataset(dataset_val)
        base_ds_Heart = get_coco_api_from_dataset(dataset_valHeart)
        base_ds_LeftLung = get_coco_api_from_dataset(dataset_valLeftLung)
        base_ds_RightLung = get_coco_api_from_dataset(dataset_valRightLung)

        # # args.taskcomponent = "detect_segmentation_cyclic_v2"
        # train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLeftLung, val_loader_vindrcxrtLeftLung, train_loader_vindrcxrRightLung, val_loader_vindrcxrtRightLung, data_loader_train, data_loader_val, sampler_train, dataset_val = dataloader_return(args)
        # base_ds = get_coco_api_from_dataset(dataset_val)
        # # args.taskcomponent = "detect_segmentation_cyclic_v3"


    if args.taskcomponent == "detect_segmentation_cyclic_v4": ## With EMA -- Teacher-Student Model
        if args.backbone_dir is not None: # 
            model = load_weights(model, args)
            # model = reinitialize_zero_weights(model)
        # if model_ema is not None:
        #     print("[Model Info.] Loading pre-trained model for Teacher-Model (EMA)!")
        #     model_ema = load_weights(model_ema, args)
        if args.modelEMA is not None:
            # model = load_weights(model, args)
            # model_ema = load_weights(model_ema, args)
            # print("[Model Info.] Loading pre-trained model for Teacher-Model (EMA):", args.resume)
            # model_ema = copy.deepcopy(model)

            for param_q, param_k in zip(model.parameters(), model_ema.parameters()):
                        param_k.data.copy_(param_q.detach().data)
            print("[Model Info.] Using Epoch-wise EMA Model for Teacher.")
            for p in model_ema.parameters():
                p.requires_grad = False

        model_path_SEGMENTATION = args.output_dir
        if not os.path.exists(model_path_SEGMENTATION):
            os.makedirs(model_path_SEGMENTATION)
        logs_path = os.path.join(model_path_SEGMENTATION, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if os.path.exists(os.path.join(logs_path, "log.txt")):
            log_writter_SEGMENTATION = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            log_writter_SEGMENTATION = open(os.path.join(logs_path, "log.txt"), 'w')

        export_csvFile = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Model', 'Task-Test', 'mAP', 'DICE'])
        export_csvFile.to_csv(args.output_dir+'/export_csvFile.csv', index=False)

        # fields=[epoch, 'Vindr-Organ_Heart', 'Localization', 'Student', 'Train', str(0.00), str(0.00)] # AUC_SliceLevel_Res
        # with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(fields)


        args.total_epochs = args.total_epochs
        img_size = args.imgsize # 448 worked
        batch_size = args.batch_size
        num_workers = args.num_workers
        best_val_loss_SEGMENTATION = 100000
        patience_SEGMENTATION = 50
        patience_counter_SEGMENTATION = 0

        print()
        print("-------------")
        print("[Information]  TASK:", args.taskcomponent)
        print("[Information]  EMA:", args.modelEMA)
        print("[Information]  Backbone:", args.backbonemodel) 
        print("[Information]  Backbone_INIT:", args.init)
        print("[Information]  Backbone Weights:", args.backbone_dir)
        print("[Information]  Dataset:", args.segmentation_dataset)
        print("[Information]  Total Epoch:", args.total_epochs)
        print("[Information]  Image Size:", img_size)
        print("[Information]  Batch Size:", batch_size)
        print("[Information]  Learning Rate:", args.lr)
        print("[Information]  Learning Rate Backbone:", args.lr_backbone)
        print("[Information]  Num Workers:", num_workers)
        print("[Information]  Optimizer:", args.opt)
        # print("[Information]  Patience:", patience_SEGMENTATION)
        print("[Information]  Output Dir:", args.output_dir)
        # print("[Exported Files]", args.out_dir+'/export_csvFile.csv')
        print("-------------")
        print() # log_writter_SEGMENTATION

        print("-------------", file=log_writter_SEGMENTATION)
        print("[Information]  TASK:", args.taskcomponent, file=log_writter_SEGMENTATION)
        print("[Information]  EMA:", args.modelEMA, file=log_writter_SEGMENTATION)
        print("[Information]  Backbone:", args.backbonemodel, file=log_writter_SEGMENTATION)
        print("[Information]  Backbone_INIT:", args.init, file=log_writter_SEGMENTATION)
        print("[Information]  Backbone Weights:", args.backbone_dir, file=log_writter_SEGMENTATION)
        print("[Information]  Dataset:", args.segmentation_dataset, file=log_writter_SEGMENTATION)
        print("[Information]  Total Epoch:", args.total_epochs, file=log_writter_SEGMENTATION)
        print("[Information]  Image Size:", img_size, file=log_writter_SEGMENTATION)
        print("[Information]  Batch Size:", batch_size, file=log_writter_SEGMENTATION)
        print("[Information]  Learning Rate:", args.lr, file=log_writter_SEGMENTATION)
        print("[Information]  Learning Rate Backbone:", args.lr_backbone, file=log_writter_SEGMENTATION)
        print("[Information]  Num Workers:", num_workers, file=log_writter_SEGMENTATION)
        print("[Information]  Optimizer:", args.opt, file=log_writter_SEGMENTATION)
        # print("[Information]  Patience:", patience_SEGMENTATION, file=log_writter_SEGMENTATION)
        print("[Information]  Output Dir:", args.output_dir, file=log_writter_SEGMENTATION)
        # print("[Exported Files]", args.out_dir+'/export_csvFile.csv', file=log_writter_SEGMENTATION)
        print("-------------", file=log_writter_SEGMENTATION)

        loss_scaler = NativeScaler()

        train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLeftLung, val_loader_vindrcxrtLeftLung, train_loader_vindrcxrRightLung, val_loader_vindrcxrtRightLung, \
            data_loader_trainHeart, sampler_trainHeart, data_loader_valHeart, dataset_valHeart, \
            data_loader_trainLeftLung, sampler_trainLeftLung, data_loader_valLeftLung, dataset_valLeftLung, \
            data_loader_trainRightLung, sampler_trainRightLung, data_loader_valRightLung, dataset_valRightLung = dataloader_return(args)

        base_ds_Heart = get_coco_api_from_dataset(dataset_valHeart)
        base_ds_LeftLung = get_coco_api_from_dataset(dataset_valLeftLung)
        base_ds_RightLung = get_coco_api_from_dataset(dataset_valRightLung)









    ## T R A I N I N G   P H A S E ##
    print("Start training")
    start_time_ALL = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    args.start_epoch = 1

    # for param in model.parameters():
    #     param.requires_grad = True

    # for name, param in model.named_parameters():
    #     if 'backbone' in name and 'segmentation_' not in name:
    #         param.requires_grad = False
    # old_value_layernorm = sum(model.module.backbone[0].layers[3].blocks[1].mlp.fc2.weight)

    if args.modelEMA == "True_Epoch":
        Num_EPOCH_Iterative_Steps_MomentumSchduler = 10
        momentum_schedule = cosine_scheduler(0.9, 1, Num_EPOCH_Iterative_Steps_MomentumSchduler, 1) # Epoch-wise
        # momentum_schedule = cosine_scheduler(0.9, 1, args.total_epochs, 1)
        print("[Model Info] EMA Epoch-wise Update.")
    else:
        momentum_schedule = None
    
    print("[Model Info.] Optimizer:", optimizer)
    print("[Training Info.] Start_Epoch & Total_Epoch", args.start_epoch, args.total_epochs)
    print()

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model._set_static_graph() # added by Nahid because of adding Classification & Segmentation component -- Forward/Backward pass issue
    for epoch in range(args.start_epoch, args.total_epochs):
        epoch_start_time = time.time()

        ## D E T E C T I O N  T A S K ##
        if args.taskcomponent == 'detection':
            if args.distributed:   # Active for Detection
                sampler_train.set_epoch(epoch)

            lr_ = step_decay(epoch, args.lr, args.total_epochs, step_inc=8) # Localizer
            lrBackbone_ = step_decay(epoch, args.lr_backbone, args.total_epochs, step_inc=20) # Backbone
            if len(optimizer.param_groups) == 2:
                optimizer.param_groups[0]['lr'] = lr_
                optimizer.param_groups[1]['lr'] = lrBackbone_
                print('Epoch{} - learning_rate [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[0]['lr']), file=log_writter_DETECTION)
                print('Epoch{} - learning_rateBackbone [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[1]['lr']), file=log_writter_DETECTION)
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                print('Epoch{} - learning_rate [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[0]['lr']), file=log_writter_DETECTION)

                if len(optimizer.param_groups) == 2:
                    print('Epoch{} - learning_rate: {:.20f}'.format(epoch, optimizer.param_groups[0]['lr']), file=log_writter_DETECTION)
                    print('Epoch{} - learning_rateBackbone: {:.20f}'.format(epoch, optimizer.param_groups[1]['lr']), file=log_writter_DETECTION)
                else:
                    print('Epoch{} - learning_rate: {:.20f}'.format(epoch, optimizer.param_groups[0]['lr']), file=log_writter_DETECTION)
            log_writter_DETECTION.flush()


            # print('Epoch{} - learning_rate: {:.20f}'.format(epoch, optimizer.param_groups[0]['lr']), file=log_writter_DETECTION)
            # print('Epoch{} - learning_rateBackbone: {:.20f}'.format(epoch, optimizer.param_groups[1]['lr']), file=log_writter_DETECTION)


            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']

            # if not args.onecyclelr:
            #     lr_scheduler.step()

            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.save_checkpoint_interval == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    weights = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                    if args.use_ema:
                        weights.update({
                            'ema_model': ema_m.module.state_dict(),
                        })
                    utils.save_on_master(weights, checkpoint_path)
                    
            # eval
            test_stats, coco_evaluator, _ = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            map_regular = test_stats['coco_eval_bbox'][0]
            _isbest = best_map_holder.update(map_regular, epoch, is_ema=False)
            if _isbest:
                checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
            }
            result_output_dir = output_dir / 'results.txt'
            log_writer_detection = open(result_output_dir, 'a')
            formatted_stats_train = {f'train_{k}': v for k, v in train_stats.items()}
            formatted_stats_test = {f'test_{k}': v for k, v in test_stats.items()}
            log_writer_detection.write('Epoch: ' + str(epoch) + '\n')
            log_writer_detection.write('-- Training --' + '\n')
            for key, value in formatted_stats_train.items():
                log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('-- Testing --' + '\n')
            for key, value in formatted_stats_test.items():
                log_writer_detection.write(f'{key}: {value}\n')
                if key == "test_coco_eval_bbox":
                    print(f'{epoch} - {key}: {value}\n', file=log_writter_DETECTION)
            log_writer_detection.write('\n')
            log_writer_detection.write('\n')
            log_writer_detection.close()
            log_writter_DETECTION.flush()

            # eval ema
            if args.use_ema:
                ema_test_stats, ema_coco_evaluator = evaluate(
                    ema_m.module, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                    wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
                )
                log_stats.update({f'ema_test_{k}': v for k,v in ema_test_stats.items()})
                map_ema = ema_test_stats['coco_eval_bbox'][0]
                _isbest = best_map_holder.update(map_ema, epoch, is_ema=True)
                if _isbest:
                    checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                    utils.save_on_master({
                        'model': ema_m.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
            log_stats.update(best_map_holder.summary())

            ep_paras = {
                    'epoch': epoch,
                    'n_parameters': n_parameters
                }
            log_stats.update(ep_paras)
            try:
                log_stats.update({'now_time': str(datetime.datetime.now())})
            except:
                pass
            
            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                       output_dir / "eval" / name)



        if args.taskcomponent in ['detect_vindrcxr_heart_segTest', 'detect_vindrcxr_heart', 'detect_vindrcxr_leftlung', 'detect_vindrcxr_rightlung']: ## seg_train_loader, seg_val_loader
            lrBackbone_ = step_decay(epoch, args.lr_backbone, args.total_epochs, step_inc=20)
            lrSegmentor_ = step_decay(epoch, args.lr_segmentor, args.total_epochs, step_inc=20)
            lrLocalizer_ = step_decay(epoch, args.lr, args.total_epochs, step_inc=20)
            if len(optimizer.param_groups) == 2 or len(optimizer.param_groups) == 3:
                optimizer.param_groups[0]['lr'] = lrBackbone_
                optimizer.param_groups[1]['lr'] = lrSegmentor_
                optimizer.param_groups[2]['lr'] = lrLocalizer_                    
                print('Epoch{} - learning_rateBackbone [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[0]['lr']), file=log_writter_DETECTION)
                print('Epoch{} - learning_rateSegmentor [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[1]['lr']), file=log_writter_DETECTION)
                print('Epoch{} - learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[2]['lr']), file=log_writter_DETECTION)
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                print('Epoch{} - learning_rate [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[0]['lr']), file=log_writter_DETECTION)

            start_time = time.time()

            if args.distributed:   # Active for Detection # data_loader_train, data_loader_val, sampler_train, dataset_val
                sampler_train.set_epoch(epoch)

            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, wo_class_error=wo_class_error, DetHead=0, lr_scheduler=None, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
            log_stats = { **{f'train_{k}': v for k, v in train_stats.items()} }

            # train_lossAVG = train_one_epoch_SEGMENTATION(model, seg_train_loader, optimizer, loss_scaler, epoch, head_number=None, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
            # print( "Epoch {:04d}: Train Loss {:.5f} ".format(epoch, train_lossAVG) )
            # print( "Epoch {:04d}: Train Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_SEGMENTATION )

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print(' -- Train time for Epoch {}: {}'.format(epoch, total_time_str), file=log_writter_DETECTION)

            # save_file = os.path.join(args.output_dir, 'loc_ckpt_E'+str(epoch) + '.pth')
            save_file = os.path.join(args.output_dir, 'seg_ckpt_E'+str(epoch) + '.pth')
            save_model(model, optimizer, log_writter_SEGMENTATION, epoch, save_file)
            print('\n', file=log_writter_SEGMENTATION)
            ### Testing Phase ###

            start_time = time.time()

            # Test Segmentation on the Organ
            # test_y, test_p, _ = test_SEGMENTATION(model, seg_val_loader, head_number=None, log_writter_SEGMENTATION=log_writter_SEGMENTATION) # log_writter_SEGMENTATION
            # print("[INFO] Test Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
            # log_writter_SEGMENTATION.flush()

            ## Test/Eval for Localization - All [Heart, Left Lung, Right Lung]
            model.task_DetHead = 0 ## Localization Heart
            test_stats, coco_evaluator, features_detectionList = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                DetHead=0, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats = { **{f'test_{k}': v for k, v in test_stats.items()} }

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print(' -- Test time for Epoch {}: {}'.format(epoch, total_time_str), file=log_writter_DETECTION)
            log_writter_DETECTION.flush()

            ### Storing Detection Results ###
            result_output_dir = args.output_dir + '/results.txt'
            log_writer_detection = open(result_output_dir, 'a')

            formatted_stats_train = {f'train_{k}': v for k, v in train_stats.items()}
            formatted_stats_test = {f'test_{k}': v for k, v in test_stats.items()}
            log_writer_detection.write('Epoch: ' + str(epoch) + '\n')

            log_writer_detection.write('-- Training --' + '\n')
            for key, value in formatted_stats_train.items():
                log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('-- Testing --' + '\n')
            for key, value in formatted_stats_test.items():
                log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('\n')
            log_writer_detection.close()





        ## S E G M E N T A T I O N   T A S K  ##
        if args.taskcomponent == 'segmentation':
            start_time = time.time()
            # lr_ = step_decay(epoch, args.lr, args.total_epochs)
            # for param_group in optimizer.param_groups:
            #     param_group['lr'] = lr_
            # print('learning_rate: {}, @Epoch {}'.format(optimizer.param_groups[0]['lr'], epoch), file=log_writter_SEGMENTATION)

            lrBackbone_ = step_decay(epoch, args.lr_backbone, args.total_epochs, step_inc=20)
            lrSegmentor_ = step_decay(epoch, args.lr_segmentor, args.total_epochs, step_inc=20)
            lrLocalizer_ = step_decay(epoch, args.lr, args.total_epochs, step_inc=20)
            if len(optimizer.param_groups) == 2 or len(optimizer.param_groups) == 3:
                optimizer.param_groups[0]['lr'] = lrBackbone_
                optimizer.param_groups[1]['lr'] = lrSegmentor_
                optimizer.param_groups[2]['lr'] = lrLocalizer_                    
                print('Epoch{} - learning_rateBackbone [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
                print('Epoch{} - learning_rateSegmentor [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[1]['lr']), file=log_writter_SEGMENTATION)
                print('Epoch{} - learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[2]['lr']), file=log_writter_SEGMENTATION)
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                print('Epoch{} - learning_rate [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)


            train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader, optimizer, loss_scaler, epoch, head_number=None, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
            # print( 'Training loss: {}@Epoch: {}'.format(train_lossAVG, epoch), file=log_writter_SEGMENTATION )

            # print( "Epoch {:04d}: Train Loss {:.5f} ".format(epoch, train_lossAVG) )
            print( "Epoch {:04d}: Train Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_SEGMENTATION )
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            # print('Training time {}'.format(total_time_str))
            print('Training time {}\n'.format(total_time_str), file=log_writter_SEGMENTATION)

            start_time = time.time()
            # val_avg_SEGMENTATION = evaluation_SEGMENTATION(model, val_loader, epoch, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
            test_y, test_p, _ = test_SEGMENTATION(model, val_loader, head_number=None, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
            print("[INFO] Test Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            # print('Validation time {}'.format(total_time_str))
            print('Validation/Test time {}'.format(total_time_str), file=log_writter_SEGMENTATION)

            save_file = os.path.join(model_path_SEGMENTATION, 'ckpt_ep'+str(epoch)+'.pth')
            save_model(model, optimizer, log_writter_SEGMENTATION, epoch, save_file)
            
            # if val_avg_SEGMENTATION < best_val_loss_SEGMENTATION:
            #     save_file = os.path.join(model_path_SEGMENTATION, 'ckpt.pth')
            #     save_model(model, optimizer, log_writter_SEGMENTATION, epoch+1, save_file)

            #     print( "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}\n".format(epoch, best_val_loss_SEGMENTATION, val_avg_SEGMENTATION, save_file), file=log_writter_SEGMENTATION )
            #     best_val_loss_SEGMENTATION = val_avg_SEGMENTATION
            #     patience_counter_SEGMENTATION = 0
            # else:
            #     print( "Epoch {:04d}: val_loss did not improve from {:.5f} \n".format(epoch, best_val_loss_SEGMENTATION), file=log_writter_SEGMENTATION )
            #     patience_counter_SEGMENTATION += 1

            # if patience_counter_SEGMENTATION > patience_SEGMENTATION:
            #     print( "Early Stopping", file=log_writter_SEGMENTATION )
            #     break

        if args.taskcomponent == 'segmentation_cyclic':
            ### train_loader_jsrtLung  |  val_loader_jsrtLung  |  test_loader_jsrtLung
            ### train_loader_jsrtHeart  |  val_loader_jsrtHeart  |  test_loader_jsrtHeart
            ### train_loader_jsrtClavicle  |  val_loader_jsrtClavicle  |  test_loader_jsrtClavicle
            
            if args.segmentation_dataset == "jsrt_lung_heart_clavicle":
                head_number = epoch % 3
                start_time = time.time()

                lr_ = step_decay(epoch, args.lr, args.total_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                print('learning_rate: {}, @Epoch {}'.format(optimizer.param_groups[0]['lr'], epoch), file=log_writter_SEGMENTATION)

                print('-- Epoch Head {} --'.format(head_number), file=log_writter_SEGMENTATION)
                if head_number == 0:
                    train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader_jsrtLung, optimizer, loss_scaler, epoch, head_number=head_number, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                if head_number == 1:
                    train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader_jsrtHeart, optimizer, loss_scaler, epoch, head_number=head_number, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                if head_number == 2:
                    train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader_jsrtClavicle, optimizer, loss_scaler, epoch, head_number=head_number, log_writter_SEGMENTATION=log_writter_SEGMENTATION)

                print( "Epoch {:04d}: Train Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_SEGMENTATION )
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                print('Training time {}'.format(total_time_str), file=log_writter_SEGMENTATION)

                start_time = time.time()
                if head_number == 0:
                    val_avg_SEGMENTATION = evaluation_SEGMENTATION(model, val_loader_jsrtLung, epoch, head_number=head_number, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                if head_number == 1:
                    val_avg_SEGMENTATION = evaluation_SEGMENTATION(model, val_loader_jsrtHeart, epoch, head_number=head_number, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                if head_number == 2:
                    val_avg_SEGMENTATION = evaluation_SEGMENTATION(model, val_loader_jsrtClavicle, epoch, head_number=head_number, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print( "Epoch {:04d}: Val Loss {:.5f} ".format(epoch, val_avg_SEGMENTATION), file=log_writter_SEGMENTATION )
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                print('Validation time {}\n'.format(total_time_str), file=log_writter_SEGMENTATION)


                test_y, test_p,_ = test_SEGMENTATION(model, test_loader_jsrtLung, head_number=0, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print("[INFO] JSRT Lung Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
                print("JSRT Lung Mean Dice = {:.4f}\n".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter_SEGMENTATION)
                # print("[INFO] JSRT Lung Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)))
                # print("JSRT Lung Mean Dice = {:.4f}\n".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)))

                test_y, test_p,_ = test_SEGMENTATION(model, test_loader_jsrtHeart, head_number=1, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print("[INFO] JSRT Heart Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
                print("JSRT Heart Mean Dice = {:.4f}\n".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter_SEGMENTATION)
                # print("[INFO] JSRT Heart Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)))
                # print("JSRT Heart Mean Dice = {:.4f}\n".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)))

                test_y, test_p,_ = test_SEGMENTATION(model, test_loader_jsrtClavicle, head_number=2, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print("[INFO] JSRT Clavicle Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
                print("JSRT Clavicle Mean Dice = {:.4f}\n".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter_SEGMENTATION)
                # print("[INFO] JSRT Clavicle Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)))
                # print("JSRT Clavicle Mean Dice = {:.4f}\n".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)))


                save_file = os.path.join(model_path_SEGMENTATION, 'ckpt_E'+str(epoch)+'_H'+str(head_number)+'.pth')
                save_model(model, optimizer, log_writter_SEGMENTATION, epoch+1, save_file)


            elif args.segmentation_dataset == "vindrcxr_lung_heart":
                # train_loader_vindrcxrHeart  # val_loader_vindrcxrtHeart  # train_loader_vindrcxrLung  # val_loader_vindrcxrtLung
                head_number = epoch % 2
                start_time = time.time()

                lr_ = step_decay(epoch, args.lr, args.total_epochs)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
                print('learning_rate: {}, @Epoch {}'.format(optimizer.param_groups[0]['lr'], epoch), file=log_writter_SEGMENTATION)

                print('-- Epoch Head {} --'.format(head_number), file=log_writter_SEGMENTATION)
                if head_number == 0:
                    train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader_vindrcxrHeart, optimizer, loss_scaler, epoch, head_number=head_number, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                if head_number == 1:
                    train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader_vindrcxrLung, optimizer, loss_scaler, epoch, head_number=head_number, log_writter_SEGMENTATION=log_writter_SEGMENTATION)

                print( "Epoch {:04d}: Train Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_SEGMENTATION )
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                print('Training time {}'.format(total_time_str), file=log_writter_SEGMENTATION)

                start_time = time.time()
                test_y, test_p,_ = test_SEGMENTATION(model, val_loader_vindrcxrtHeart, head_number=0, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print("[INFO] Vindr-CXR Organ Lung Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
                print("Vindr-CXR Organ Lung Mean Dice = {:.4f}\n".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter_SEGMENTATION)

                test_y, test_p,_ = test_SEGMENTATION(model, val_loader_vindrcxrtLung, head_number=1, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print("[INFO] Vindr-CXR Organ Heart Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
                print("Vindr-CXR Organ Heart Mean Dice = {:.4f}\n".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter_SEGMENTATION)

                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                print('Test time {}\n'.format(total_time_str), file=log_writter_SEGMENTATION)

                save_file = os.path.join(model_path_SEGMENTATION, 'ckpt_E'+str(epoch)+'_H'+str(head_number)+'.pth')
                save_model(model, optimizer, log_writter_SEGMENTATION, epoch+1, save_file)



        ## C L A S S I F I C A T I O N   T A S K ##
        if args.taskcomponent == 'classification':
            start_time = time.time()
            print("- Training Epoch -")
            if args.distributed and args.classification_dataset == "imagenet":   # Active for Detection
                train_sampler.set_epoch(epoch)
                lr_ = adjust_learning_rate_CLASSIFICATION(optimizer, epoch, args.lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            # else:
            #     lr_ = step_decay_cosine(epoch, args.lr, args.total_epochs + 1,  warmup_epochs=10)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_
            lr_scheduler.step(epoch, 0)
            print()
            print( "- Epoch " + str(epoch) + " -- Learning Rate {:.6f}".format(optimizer.state_dict()['param_groups'][0]['lr']) )
            print( "- Epoch " + str(epoch) + " -- Learning Rate {:.6f}".format(optimizer.state_dict()['param_groups'][0]['lr']), file=log_writter_CLASSIFICATION )


            # train for one epoch
            train_lossAVG = train_CLASSIFICATION(train_loader, model, criterion, optimizer, epoch, args, log_writter_CLASSIFICATION)
            print( "Epoch {:04d}: Train Loss {:.5f} ".format(epoch, train_lossAVG) )
            print( "Epoch {:04d}: Train Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_CLASSIFICATION )
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            print('Training time {}'.format(total_time_str), file=log_writter_CLASSIFICATION)

            log_writter_CLASSIFICATION.flush()
            print()

            # evaluate on validation set
            print("- Validation Epoch -")
            if args.classification_dataset == "imagenet":
                acc1 = evaluate_CLASSIFICATION(val_loader, model, criterion, args, log_writter_CLASSIFICATION)

                # remember best acc@1 and save checkpoint
                is_best = acc1 > best_acc1_CLASSIFICATION
                best_acc1_CLASSIFICATION = max(acc1, best_acc1_CLASSIFICATION)

                save_file = os.path.join(model_path_CLASSIFICATION, 'ckpt_'+str(epoch+1)+'.pth')
                save_model(model, optimizer, log_writter_CLASSIFICATION, epoch+1, save_file)
            else:
                start_time = time.time()
                val_loss = evaluate_CLASSIFICATION(val_loader, model, criterion, args, log_writter_CLASSIFICATION)
                print( "Epoch {:04d}: Val Loss {:.5f} ".format(epoch, val_loss) )
                print( "Epoch {:04d}: Val Loss {:.5f} ".format(epoch, val_loss), file=log_writter_CLASSIFICATION )
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                print('Validation time {}'.format(total_time_str))
                print('Validation time {}'.format(total_time_str), file=log_writter_CLASSIFICATION)
                log_writter_CLASSIFICATION.flush()

                # lr_scheduler.step(epoch+1, 0)

                is_best = val_loss < best_val_CLASSIFICATION
                best_val_CLASSIFICATION = min(val_loss, best_val_CLASSIFICATION)

                if is_best:
                    patience_counter_CLASSIFICATION = 0
                    print('Saving Model - Best at Epoch ' + str(epoch+1))
                    print('Saving Model - Best at Epoch ' + str(epoch+1), file=log_writter_CLASSIFICATION)
                    save_file = os.path.join(model_path_CLASSIFICATION, 'ckpt_BEST.pth')
                    save_model(model, optimizer, log_writter_CLASSIFICATION, epoch+1, save_file)
                else:
                    patience_counter_CLASSIFICATION = patience_counter_CLASSIFICATION + 1
                    print('[Val Info.] Val_loss did not improve. Patience:' + str(patience_counter_CLASSIFICATION))
                    print('[Val Info.] Val_loss did not improve. Patience:' + str(patience_counter_CLASSIFICATION), file=log_writter_CLASSIFICATION)

                save_file = os.path.join(model_path_CLASSIFICATION, 'ckpt_'+str(epoch+1)+'.pth')
                save_model(model, optimizer, log_writter_CLASSIFICATION, epoch+1, save_file)

                # patience_counter_CLASSIFICATION patience_CLASSIFICATION
                if patience_counter_CLASSIFICATION > patience_CLASSIFICATION:
                    print( "Early Stopping" )
                    print( "Early Stopping", file=log_writter_CLASSIFICATION )
                    log_writter_CLASSIFICATION.flush()
                    break

        if args.taskcomponent == "detect_segmentation_cyclic":
            head_number = (epoch - 1) % 3

            if head_number == 0:  ### LR update --->  Backbone -- Segmentor -- Localizer
                lrBackbone_ = step_decay(epoch, args.lr_backbone, args.total_epochs, step_inc=20)
                lrSegmentor_ = step_decay(epoch, args.lr_segmentor, args.total_epochs, step_inc=20)
                lrLocalizer_ = step_decay(epoch, args.lr, args.total_epochs, step_inc=20)
                if len(optimizer.param_groups) == 2 or len(optimizer.param_groups) == 3:
                    optimizer.param_groups[0]['lr'] = lrBackbone_
                    optimizer.param_groups[1]['lr'] = lrSegmentor_
                    optimizer.param_groups[2]['lr'] = lrLocalizer_                    
                    print('Epoch{} - Head{} - learning_rateBackbone [Updated]: {:.20f}'.format(epoch, head_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
                    print('Epoch{} - Head{} - learning_rateSegmentor [Updated]: {:.20f}'.format(epoch, head_number, optimizer.param_groups[1]['lr']), file=log_writter_SEGMENTATION)
                    print('Epoch{} - Head{} - learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, head_number, optimizer.param_groups[2]['lr']), file=log_writter_SEGMENTATION)
                else:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
                    print('Epoch{} - Head{} - learning_rate [Updated]: {:.20f}'.format(epoch, head_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
            else:
                if len(optimizer.param_groups) == 2 or len(optimizer.param_groups) == 3:
                    print('Epoch{} - Head{} - learning_rateBackbone: {:.20f}'.format(epoch, head_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
                    print('Epoch{} - Head{} - learning_rateSegmentor: {:.20f}'.format(epoch, head_number, optimizer.param_groups[1]['lr']), file=log_writter_SEGMENTATION)
                    print('Epoch{} - Head{} - learning_rateLocalizer: {:.20f}'.format(epoch, head_number, optimizer.param_groups[2]['lr']), file=log_writter_SEGMENTATION)
                else:
                    print('Epoch{} - Head{} - learning_rate: {:.20f}'.format(epoch, head_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
            # print('Epoch{} - Head{} - learning_rate: {:.20f}'.format(epoch, head_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
            # print('Epoch{} - Head{} - learning_rateBackbone: {:.20f}'.format(epoch, head_number, optimizer.param_groups[1]['lr']), file=log_writter_SEGMENTATION)


            # for param in model.parameters():
            #     param.requires_grad = True
            # freeze_check = 0
            # if head_number == 2: # Training Localization
            #     for name, param in model.named_parameters():
            #         if 'segmentation_' in name:
            #             param.requires_grad = False
            #             freeze_check = 1
            #     print("[Check Freeze] Segmentation Component Frozen:", freeze_check, file=log_writter_SEGMENTATION)
            # elif head_number == 0 or head_number == 1: # Training Segmentation
            #     for name, param in model.named_parameters():
            #         if 'transformer' in name: # Dino Decoder
            #             param.requires_grad = False
            #             freeze_check = 1
            #     print("[Check Freeze] Localization Component Frozen:", freeze_check, file=log_writter_SEGMENTATION)
            # else:
            #     for param in model.parameters():
            #         param.requires_grad = True

            after_value_layernorm = sum(model.module.backbone[0].layers[3].blocks[1].mlp.fc2.weight)
            if sum(old_value_layernorm).item() == sum(after_value_layernorm).item():
                print("[Check Freeze] Backbone Component Frozen.", file=log_writter_SEGMENTATION)


            start_time = time.time()
            print('-- Epoch Head {} --'.format(head_number), file=log_writter_SEGMENTATION)


            if head_number == 0:
                train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader_vindrcxrHeart, optimizer, loss_scaler, epoch, head_number=head_number, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
            if head_number == 1:
                train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader_vindrcxrLung, optimizer, loss_scaler, epoch, head_number=head_number, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
            print( "Epoch {:04d}: Train Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_SEGMENTATION )
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Train Segmentation time for {}: {}\n'.format(head_number, total_time_str), file=log_writter_SEGMENTATION)


            start_time = time.time()
            test_y, test_p, features_segmentationList = test_SEGMENTATION(model, val_loader_vindrcxrtHeart, head_number=0, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
            print("[INFO] Vindr-CXR Organ Lung Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
            print("Vindr-CXR Organ Lung Mean Dice = {:.4f}\n".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter_SEGMENTATION)
            if head_number == 0 and args.tsneOut:
                directory_tsne = args.output_dir+'/tsneFeatures/'
                if not os.path.exists(directory_tsne):
                    os.makedirs(directory_tsne)
                np.save(directory_tsne+'segmentation_E'+str(epoch)+'_H'+str(head_number)+'.npy', np.array(features_segmentationList))


            test_y, test_p, features_segmentationList = test_SEGMENTATION(model, val_loader_vindrcxrtLung, head_number=1, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
            print("[INFO] Vindr-CXR Organ Heart Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
            print("Vindr-CXR Organ Heart Mean Dice = {:.4f}\n".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter_SEGMENTATION)
            if head_number == 1 and args.tsneOut:
                directory_tsne = args.output_dir+'/tsneFeatures/'
                if not os.path.exists(directory_tsne):
                    os.makedirs(directory_tsne)
                np.save(directory_tsne+'segmentation_E'+str(epoch)+'_H'+str(head_number)+'.npy', np.array(features_segmentationList))


            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Test time {}\n'.format(total_time_str), file=log_writter_SEGMENTATION)

            save_file = os.path.join(model_path_SEGMENTATION, 'ckpt_E'+str(epoch)+'_H'+str(head_number)+'.pth')
            save_model(model, optimizer, log_writter_SEGMENTATION, epoch, save_file)


            if args.distributed:   # Active for Detection
                sampler_train.set_epoch(epoch)
            # data_loader_train # data_loader_val # data_loader_train_Heart # data_loader_train_Lung # data_loader_val_Heart # data_loader_val_Lung
            start_time = time.time()
            if head_number == 2:
                train_stats = train_one_epoch(
                    model, criterion, data_loader_train, optimizer, device, epoch,
                    args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=None, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
                log_stats = { **{f'train_{k}': v for k, v in train_stats.items()} }
            # if head_number == 3:
            #     train_stats = train_one_epoch(
            #         model, criterion, data_loader_train_Lung, optimizer, device, epoch,
            #         args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=None, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
            #     log_stats = { **{f'train_{k}': v for k, v in train_stats.items()} }
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Train Detection time for Head {}: {}\n'.format(head_number, total_time_str), file=log_writter_SEGMENTATION)


            ### Eval -- Heart
            test_stats, coco_evaluator, features_detectionList = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats = { **{f'test_{k}': v for k, v in test_stats.items()} }

            if args.tsneOut:
                # if head_number == 2:
                # print("CHECK features_detectionList", len(features_detectionList))
                directory_tsne = args.output_dir+'/tsneFeatures/'
                if not os.path.exists(directory_tsne):
                    os.makedirs(directory_tsne)
                np.save(directory_tsne+'localization_E'+str(epoch)+'_H'+str(head_number)+'.npy', np.array(features_detectionList))

            result_output_dir = args.output_dir + '/results.txt'
            log_writer_detection = open(result_output_dir, 'a')
            if head_number == 2 or head_number == 3:
                formatted_stats_train = {f'train_{k}': v for k, v in train_stats.items()}
            formatted_stats_test = {f'test_{k}': v for k, v in test_stats.items()}
            log_writer_detection.write('Epoch: ' + str(epoch) + "| Head_number: " + str(head_number) + " Heart&Lung " + '\n')
            if head_number == 2 or head_number == 3:
                log_writer_detection.write('-- Training --' + '\n')
                for key, value in formatted_stats_train.items():
                    log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('-- Testing --' + '\n')
            for key, value in formatted_stats_test.items():
                log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('\n')
            log_writer_detection.close()


            # ### Eval -- Lung
            # test_stats, coco_evaluator = evaluate(
            #     model, criterion, postprocessors, data_loader_val_Lung, base_ds, device, args.output_dir,
            #     wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            # )
            # log_stats = { **{f'test_{k}': v for k, v in test_stats.items()} }

            # result_output_dir = args.output_dir + '/results.txt'
            # log_writer_detection = open(result_output_dir, 'a')
            # if head_number == 2 or head_number == 3:
            #     formatted_stats_train = {f'train_{k}': v for k, v in train_stats.items()}
            # formatted_stats_test = {f'test_{k}': v for k, v in test_stats.items()}
            # log_writer_detection.write('Epoch: ' + str(epoch) + "| Head_number: " + str(head_number) + " LUNG " + '\n')
            # if head_number == 2 or head_number == 3:
            #     log_writer_detection.write('-- Training --' + '\n')
            #     for key, value in formatted_stats_train.items():
            #         log_writer_detection.write(f'{key}: {value}\n')
            # log_writer_detection.write('\n')
            # log_writer_detection.write('-- Testing --' + '\n')
            # for key, value in formatted_stats_test.items():
            #     log_writer_detection.write(f'{key}: {value}\n')
            # log_writer_detection.write('\n')
            # log_writer_detection.write('\n')
            # log_writer_detection.close()

            # lr_scheduler.step()



        if args.taskcomponent == "detect_segmentation_cyclic_v2": # # if args.taskcomponent == "detect_segmentation_cyclic_v3": # (1)Loc Left Lung, (2)Seg Left Lung, (3)Loc Right Lung, (4)Seg Right Lung, (5)Loc Heart, (6)Seg Heart
            # train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLeftLung, val_loader_vindrcxrtLeftLung, train_loader_vindrcxrRightLung, val_loader_vindrcxrtRightLung, data_loader_train, data_loader_val,
            TaskHead_number = (epoch - 1) % 6

            if TaskHead_number == 0:  ### LR update --->  Backbone -- Segmentor -- Localizer
                lrBackbone_ = step_decay(epoch, args.lr_backbone, args.total_epochs, step_inc=20)
                lrSegmentor_ = step_decay(epoch, args.lr_segmentor, args.total_epochs, step_inc=20)
                lrLocalizer_ = step_decay(epoch, args.lr, args.total_epochs, step_inc=20)
                if len(optimizer.param_groups) == 2 or len(optimizer.param_groups) == 3:
                    optimizer.param_groups[0]['lr'] = lrBackbone_
                    optimizer.param_groups[1]['lr'] = lrSegmentor_
                    optimizer.param_groups[2]['lr'] = lrLocalizer_                    
                    print('Epoch{} - TaskHead{} - learning_rateBackbone [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
                    print('Epoch{} - TaskHead{} - learning_rateSegmentor [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[1]['lr']), file=log_writter_SEGMENTATION)
                    print('Epoch{} - TaskHead{} - learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[2]['lr']), file=log_writter_SEGMENTATION)
                else:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
                    print('Epoch{} - TaskHead{} - learning_rate [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
            else:
                if len(optimizer.param_groups) == 2 or len(optimizer.param_groups) == 3:
                    print('Epoch{} - TaskHead{} - learning_rateBackbone: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
                    print('Epoch{} - TaskHead{} - learning_rateSegmentor: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[1]['lr']), file=log_writter_SEGMENTATION)
                    print('Epoch{} - TaskHead{} - learning_rateLocalizer: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[2]['lr']), file=log_writter_SEGMENTATION)
                else:
                    print('Epoch{} - TaskHead{} - learning_rate: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
            # print('Epoch{} - Head{} - learning_rate: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
            # print('Epoch{} - Head{} - learning_rateBackbone: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[1]['lr']), file=log_writter_SEGMENTATION)


            for param in model.parameters():
                param.requires_grad = True
            # freeze_check = 0
            # if TaskHead_number == 2: # Training Localization
            #     for name, param in model.named_parameters():
            #         if 'segmentation_' in name:
            #             param.requires_grad = False
            #             freeze_check = 1
            #     print("[Check Freeze] Segmentation Component Frozen:", freeze_check, file=log_writter_SEGMENTATION)
            # elif TaskHead_number == 0 or TaskHead_number == 1: # Training Segmentation
            #     for name, param in model.named_parameters():
            #         if 'transformer' in name: # Dino Decoder
            #             param.requires_grad = False
            #             freeze_check = 1
            #     print("[Check Freeze] Localization Component Frozen:", freeze_check, file=log_writter_SEGMENTATION)
            # else:
            #     for param in model.parameters():
            #         param.requires_grad = True

            after_value_layernorm = sum(model.module.backbone[0].layers[3].blocks[1].mlp.fc2.weight)
            if sum(old_value_layernorm).item() == sum(after_value_layernorm).item():
                print("[Check Freeze] Backbone Component Frozen.", file=log_writter_SEGMENTATION)

            if args.distributed:   # Active for Detection
                sampler_train.set_epoch(epoch)


            start_time = time.time()
            print('-- Epoch {} TaskHead {} --'.format(epoch, TaskHead_number), file=log_writter_SEGMENTATION)

            ### Training Phase ###
            ### Localization Vindr-CXR Organ ==> (1) Heart, (2) Left Lung, (3) Right Lung
            ## Segmentation ==> train_loader_vindrcxrHeart, train_loader_vindrcxrLeftLung, train_loader_vindrcxrRightLung
            if TaskHead_number == 0: # Fine-tune for Localization - Heart
                model.task_DetHead = 0
                train_stats = train_one_epoch(
                    model, criterion, data_loader_train, optimizer, device, epoch,
                    args.clip_max_norm, wo_class_error=wo_class_error, DetHead=0, lr_scheduler=None, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
                log_stats = { **{f'train_Heart_{k}': v for k, v in train_stats.items()} }

            if TaskHead_number == 1: # Fine-tune for Segmentation - Heart || head_number = segmentation head number
                train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader_vindrcxrHeart, optimizer, loss_scaler, epoch, head_number=0, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print( "Epoch {:04d}: Train Heart Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_SEGMENTATION )

            if TaskHead_number == 2: # Fine-tune for Localization - Left Lung
                model.task_DetHead = 1
                train_stats = train_one_epoch(
                    model, criterion, data_loader_train, optimizer, device, epoch,
                    args.clip_max_norm, wo_class_error=wo_class_error, DetHead=1, lr_scheduler=None, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
                log_stats = { **{f'train_LeftLung_{k}': v for k, v in train_stats.items()} }

            if TaskHead_number == 3: # Fine-tune for Segmentation - Left Lung || head_number = segmentation head number
                train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader_vindrcxrLeftLung, optimizer, loss_scaler, epoch, head_number=1, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print( "Epoch {:04d}: Train Left Lung Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_SEGMENTATION )
            
            if TaskHead_number == 4: # Fine-tune for Localization - Right Lung
                model.task_DetHead = 2
                train_stats = train_one_epoch(
                    model, criterion, data_loader_train, optimizer, device, epoch,
                    args.clip_max_norm, wo_class_error=wo_class_error, DetHead=2, lr_scheduler=None, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
                log_stats = { **{f'train_RightLung{k}': v for k, v in train_stats.items()} }

            if TaskHead_number == 5: # Fine-tune for Segmentation - Right Lung || head_number = segmentation head number
                train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader_vindrcxrRightLung, optimizer, loss_scaler, epoch, head_number=2, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print( "Epoch {:04d}: Train Right Lung Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_SEGMENTATION )

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print(' -- Train time for Epoch {} - TaskHead {}: {}\n'.format(epoch, TaskHead_number, total_time_str), file=log_writter_SEGMENTATION)



            ### Testing Phase ###
            start_time = time.time()

            ## Test/Eval for Localization - All [Heart, Left Lung, Right Lung]
            model.task_DetHead = 0 ## Localization Heart
            test_stats, coco_evaluator, features_detectionList = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                DetHead=0, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats = { **{f'test_Heart_{k}': v for k, v in test_stats.items()} }
            ### Storing Detection Results ###
            result_output_dir = args.output_dir + '/results.txt'
            log_writer_detection = open(result_output_dir, 'a')
            if TaskHead_number == 0 or TaskHead_number == 2 or TaskHead_number == 4: ## 0 2 4
                formatted_stats_train = {f'train_Heart_{k}': v for k, v in train_stats.items()}
            formatted_stats_test = {f'test_Heart_{k}': v for k, v in test_stats.items()}
            log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Heart " + '\n')
            if TaskHead_number == 0 or TaskHead_number == 2 or TaskHead_number == 4:
                log_writer_detection.write('-- Training --' + '\n')
                for key, value in formatted_stats_train.items():
                    log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('-- Testing --' + '\n')
            for key, value in formatted_stats_test.items():
                log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('\n')
            log_writer_detection.close()

            model.task_DetHead = 1 ## Localization Left Lung
            test_stats, coco_evaluator, features_detectionList = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                DetHead=1, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats = { **{f'test_LeftLung_{k}': v for k, v in test_stats.items()} }
            ### Storing Detection Results ###
            result_output_dir = args.output_dir + '/results.txt'
            log_writer_detection = open(result_output_dir, 'a')
            if TaskHead_number == 0 or TaskHead_number == 2 or TaskHead_number == 4: ## 0 2 4
                formatted_stats_train = {f'train_LeftLung_{k}': v for k, v in train_stats.items()}
            formatted_stats_test = {f'test_LeftLung_{k}': v for k, v in test_stats.items()}
            log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Left Lung " + '\n')
            if TaskHead_number == 0 or TaskHead_number == 2 or TaskHead_number == 4:
                log_writer_detection.write('-- Training --' + '\n')
                for key, value in formatted_stats_train.items():
                    log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('-- Testing --' + '\n')
            for key, value in formatted_stats_test.items():
                log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('\n')
            log_writer_detection.close()

            model.task_DetHead = 2 ## Localization Right Lung
            test_stats, coco_evaluator, features_detectionList = evaluate(
                model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir,
                DetHead=2, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats = { **{f'test_RightLung_{k}': v for k, v in test_stats.items()} }
            ### Storing Detection Results ###
            result_output_dir = args.output_dir + '/results.txt'
            log_writer_detection = open(result_output_dir, 'a')
            if TaskHead_number == 0 or TaskHead_number == 2 or TaskHead_number == 4: ## 0 2 4
                formatted_stats_train = {f'train_RightLung_{k}': v for k, v in train_stats.items()}
            formatted_stats_test = {f'test_RightLung_{k}': v for k, v in test_stats.items()}
            log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Right Lung " + '\n')
            if TaskHead_number == 0 or TaskHead_number == 2 or TaskHead_number == 4:
                log_writer_detection.write('-- Training --' + '\n')
                for key, value in formatted_stats_train.items():
                    log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('-- Testing --' + '\n')
            for key, value in formatted_stats_test.items():
                log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('\n')
            log_writer_detection.close()


            ### val_loader_vindrcxrtHeart val_loader_vindrcxrtLeftLung val_loader_vindrcxrtRightLung
            ## Test/Eval for Segmentation - Heart
            test_y, test_p, features_segmentationList = test_SEGMENTATION(model, val_loader_vindrcxrtHeart, head_number=0, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
            print("[INFO] Vindr-CXR Organ Heart Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
            print("Vindr-CXR Organ Heart Mean Dice = {:.4f}\n".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter_SEGMENTATION)

            ## Test/Eval for Segmentation - Left Lung
            test_y, test_p, features_segmentationList = test_SEGMENTATION(model, val_loader_vindrcxrtLeftLung, head_number=1, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
            print("[INFO] Vindr-CXR Organ Left Lung Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
            print("Vindr-CXR Organ Left Lung Mean Dice = {:.4f}\n".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter_SEGMENTATION)

            ## Test/Eval for Segmentation - Right Lung
            test_y, test_p, features_segmentationList = test_SEGMENTATION(model, val_loader_vindrcxrtRightLung, head_number=2, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
            print("[INFO] Vindr-CXR Organ Right Lung Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
            print("Vindr-CXR Organ Right Lung Mean Dice = {:.4f}\n".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter_SEGMENTATION)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Test time Epoch {} TaskHead {} -- {}'.format(epoch, TaskHead_number, total_time_str), file=log_writter_SEGMENTATION)


            # lr_scheduler.step()

            save_file = os.path.join(model_path_SEGMENTATION, 'ckpt_E'+str(epoch)+'_TH'+str(TaskHead_number)+'.pth')
            save_model(model, optimizer, log_writter_SEGMENTATION, epoch, save_file)
            print('\n', file=log_writter_SEGMENTATION)



        if args.taskcomponent == "detect_segmentation_cyclic_v3": # # # (1)Loc Heart, (2)Seg Heart, (3)Loc Left Lung, (4)Seg Left Lung, (5)Loc Right Lung, (6)Seg Right Lung
            # train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLeftLung, val_loader_vindrcxrtLeftLung, train_loader_vindrcxrRightLung, val_loader_vindrcxrtRightLung, data_loader_train, data_loader_val,
            TaskHead_number = (epoch - 1) % 6 # model_ema

            if TaskHead_number == 0:  ### LR update --->  Backbone -- Segmentor -- Localizer
                lrBackbone_ = step_decay(epoch, args.lr_backbone, args.total_epochs, step_inc=20)
                lrSegmentor_ = step_decay(epoch, args.lr_segmentor, args.total_epochs, step_inc=20)
                lrLocalizer_ = step_decay(epoch, args.lr, args.total_epochs, step_inc=20)
                if len(optimizer.param_groups) == 2 or len(optimizer.param_groups) == 3:
                    optimizer.param_groups[0]['lr'] = lrBackbone_
                    optimizer.param_groups[1]['lr'] = lrSegmentor_
                    optimizer.param_groups[2]['lr'] = lrLocalizer_                    
                    print('Epoch{} - TaskHead{} - learning_rateBackbone [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
                    print('Epoch{} - TaskHead{} - learning_rateSegmentor [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[1]['lr']), file=log_writter_SEGMENTATION)
                    print('Epoch{} - TaskHead{} - learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[2]['lr']), file=log_writter_SEGMENTATION)
                else:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
                    print('Epoch{} - TaskHead{} - learning_rate [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
            else:
                if len(optimizer.param_groups) == 2 or len(optimizer.param_groups) == 3:
                    print('Epoch{} - TaskHead{} - learning_rateBackbone: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
                    print('Epoch{} - TaskHead{} - learning_rateSegmentor: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[1]['lr']), file=log_writter_SEGMENTATION)
                    print('Epoch{} - TaskHead{} - learning_rateLocalizer: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[2]['lr']), file=log_writter_SEGMENTATION)
                else:
                    print('Epoch{} - TaskHead{} - learning_rate: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
            # print('Epoch{} - Head{} - learning_rate: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
            # print('Epoch{} - Head{} - learning_rateBackbone: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[1]['lr']), file=log_writter_SEGMENTATION)


            for param in model.parameters():
                param.requires_grad = True
            # freeze_check = 0
            # if TaskHead_number == 2: # Training Localization
            #     for name, param in model.named_parameters():
            #         if 'segmentation_' in name:
            #             param.requires_grad = False
            #             freeze_check = 1
            #     print("[Check Freeze] Segmentation Component Frozen:", freeze_check, file=log_writter_SEGMENTATION)
            # elif TaskHead_number == 0 or TaskHead_number == 1: # Training Segmentation
            #     for name, param in model.named_parameters():
            #         if 'transformer' in name: # Dino Decoder
            #             param.requires_grad = False
            #             freeze_check = 1
            #     print("[Check Freeze] Localization Component Frozen:", freeze_check, file=log_writter_SEGMENTATION)
            # else:
            #     for param in model.parameters():
            #         param.requires_grad = True

            after_value_layernorm = sum(model.module.backbone[0].layers[3].blocks[1].mlp.fc2.weight)
            if sum(old_value_layernorm).item() == sum(after_value_layernorm).item():
                print("[Check Freeze] Backbone Component Frozen.", file=log_writter_SEGMENTATION)

            if args.distributed:   # Active for Detection
                sampler_train.set_epoch(epoch)


            start_time = time.time()
            print('-- Epoch {} TaskHead {} --'.format(epoch, TaskHead_number), file=log_writter_SEGMENTATION)

            ### Training Phase ###
            ### Localization Vindr-CXR Organ ==> (1) Heart, (2) Left Lung, (3) Right Lung
            ## Segmentation ==> train_loader_vindrcxrHeart, train_loader_vindrcxrLeftLung, train_loader_vindrcxrRightLung
            if TaskHead_number == 0: # Fine-tune for Localization - Heart
                model.task_DetHead = 0
                train_stats = train_one_epoch(
                    model, criterion, data_loader_train, optimizer, device, epoch,
                    args.clip_max_norm, wo_class_error=wo_class_error, DetHead=0, lr_scheduler=None, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
                log_stats = { **{f'train_Heart_{k}': v for k, v in train_stats.items()} }

            if TaskHead_number == 1: # Fine-tune for Segmentation - Heart || head_number = segmentation head number
                train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader_vindrcxrHeart, optimizer, loss_scaler, epoch, head_number=0, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print( "Epoch {:04d}: Train Heart Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_SEGMENTATION )

            if TaskHead_number == 2: # Fine-tune for Localization - Left Lung
                model.task_DetHead = 1
                train_stats = train_one_epoch(
                    model, criterion, data_loader_train, optimizer, device, epoch,
                    args.clip_max_norm, wo_class_error=wo_class_error, DetHead=1, lr_scheduler=None, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
                log_stats = { **{f'train_LeftLung_{k}': v for k, v in train_stats.items()} }

            if TaskHead_number == 3: # Fine-tune for Segmentation - Left Lung || head_number = segmentation head number
                train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader_vindrcxrLeftLung, optimizer, loss_scaler, epoch, head_number=1, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print( "Epoch {:04d}: Train Left Lung Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_SEGMENTATION )
            
            if TaskHead_number == 4: # Fine-tune for Localization - Right Lung
                model.task_DetHead = 2
                train_stats = train_one_epoch(
                    model, criterion, data_loader_train, optimizer, device, epoch,
                    args.clip_max_norm, wo_class_error=wo_class_error, DetHead=2, lr_scheduler=None, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
                log_stats = { **{f'train_RightLung{k}': v for k, v in train_stats.items()} }

            if TaskHead_number == 5: # Fine-tune for Segmentation - Right Lung || head_number = segmentation head number
                train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader_vindrcxrRightLung, optimizer, loss_scaler, epoch, head_number=2, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print( "Epoch {:04d}: Train Right Lung Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_SEGMENTATION )

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print(' -- Train time for Epoch {} - TaskHead {}: {}'.format(epoch, TaskHead_number, total_time_str), file=log_writter_SEGMENTATION)

            save_file = os.path.join(model_path_SEGMENTATION, 'ckpt_E'+str(epoch)+'_TH'+str(TaskHead_number)+'.pth')
            save_model(model, optimizer, log_writter_SEGMENTATION, epoch, save_file)
            print('\n', file=log_writter_SEGMENTATION)

            ### Testing Phase ###
            start_time = time.time()

            ## Test/Eval for Localization - All [Heart, Left Lung, Right Lung]
            model.task_DetHead = 0 ## Localization Heart
            test_stats, coco_evaluator, features_detectionList = evaluate(
                model, criterion, postprocessors, data_loader_valHeart, base_ds_Heart, device, args.output_dir,
                DetHead=0, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats = { **{f'test_Heart_{k}': v for k, v in test_stats.items()} }
            ### Storing Detection Results ###
            result_output_dir = args.output_dir + '/results.txt'
            log_writer_detection = open(result_output_dir, 'a')
            if TaskHead_number == 0 or TaskHead_number == 2 or TaskHead_number == 4: ## 0 2 4
                formatted_stats_train = {f'train_Heart_{k}': v for k, v in train_stats.items()}
            formatted_stats_test = {f'test_Heart_{k}': v for k, v in test_stats.items()}
            log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Heart " + '\n')
            if TaskHead_number == 0 or TaskHead_number == 2 or TaskHead_number == 4:
                log_writer_detection.write('-- Training --' + '\n')
                for key, value in formatted_stats_train.items():
                    log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('-- Testing --' + '\n')
            for key, value in formatted_stats_test.items():
                log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('\n')
            log_writer_detection.close()

            model.task_DetHead = 1 ## Localization Left Lung
            test_stats, coco_evaluator, features_detectionList = evaluate(
                model, criterion, postprocessors, data_loader_valLeftLung, base_ds_LeftLung, device, args.output_dir,
                DetHead=1, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats = { **{f'test_LeftLung_{k}': v for k, v in test_stats.items()} }
            ### Storing Detection Results ###
            result_output_dir = args.output_dir + '/results.txt'
            log_writer_detection = open(result_output_dir, 'a')
            if TaskHead_number == 0 or TaskHead_number == 2 or TaskHead_number == 4: ## 0 2 4
                formatted_stats_train = {f'train_LeftLung_{k}': v for k, v in train_stats.items()}
            formatted_stats_test = {f'test_LeftLung_{k}': v for k, v in test_stats.items()}
            log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Left Lung " + '\n')
            if TaskHead_number == 0 or TaskHead_number == 2 or TaskHead_number == 4:
                log_writer_detection.write('-- Training --' + '\n')
                for key, value in formatted_stats_train.items():
                    log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('-- Testing --' + '\n')
            for key, value in formatted_stats_test.items():
                log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('\n')
            log_writer_detection.close()

            model.task_DetHead = 2 ## Localization Right Lung
            test_stats, coco_evaluator, features_detectionList = evaluate(
                model, criterion, postprocessors, data_loader_valRightLung, base_ds_RightLung, device, args.output_dir,
                DetHead=2, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None)
            )
            log_stats = { **{f'test_RightLung_{k}': v for k, v in test_stats.items()} }
            ### Storing Detection Results ###
            result_output_dir = args.output_dir + '/results.txt'
            log_writer_detection = open(result_output_dir, 'a')
            if TaskHead_number == 0 or TaskHead_number == 2 or TaskHead_number == 4: ## 0 2 4
                formatted_stats_train = {f'train_RightLung_{k}': v for k, v in train_stats.items()}
            formatted_stats_test = {f'test_RightLung_{k}': v for k, v in test_stats.items()}
            log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Right Lung " + '\n')
            if TaskHead_number == 0 or TaskHead_number == 2 or TaskHead_number == 4:
                log_writer_detection.write('-- Training --' + '\n')
                for key, value in formatted_stats_train.items():
                    log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('-- Testing --' + '\n')
            for key, value in formatted_stats_test.items():
                log_writer_detection.write(f'{key}: {value}\n')
            log_writer_detection.write('\n')
            log_writer_detection.write('\n')
            log_writer_detection.close()


            ### val_loader_vindrcxrtHeart val_loader_vindrcxrtLeftLung val_loader_vindrcxrtRightLung
            ## Test/Eval for Segmentation - Heart
            test_y, test_p, features_segmentationList = test_SEGMENTATION(model, val_loader_vindrcxrtHeart, head_number=0, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
            print("[INFO] Vindr-CXR Organ Heart Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
            print("Vindr-CXR Organ Heart Mean Dice = {:.4f}\n".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter_SEGMENTATION)

            ## Test/Eval for Segmentation - Left Lung
            test_y, test_p, features_segmentationList = test_SEGMENTATION(model, val_loader_vindrcxrtLeftLung, head_number=1, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
            print("[INFO] Vindr-CXR Organ Left Lung Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
            print("Vindr-CXR Organ Left Lung Mean Dice = {:.4f}\n".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter_SEGMENTATION)

            ## Test/Eval for Segmentation - Right Lung
            test_y, test_p, features_segmentationList = test_SEGMENTATION(model, val_loader_vindrcxrtRightLung, head_number=2, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
            print("[INFO] Vindr-CXR Organ Right Lung Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
            print("Vindr-CXR Organ Right Lung Mean Dice = {:.4f}\n".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter_SEGMENTATION)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Test time Epoch {} TaskHead {} -- {}'.format(epoch, TaskHead_number, total_time_str), file=log_writter_SEGMENTATION)


            # lr_scheduler.step()


        if args.taskcomponent == "detect_segmentation_cyclic_v4": # # # (1)Loc Heart, (2)Seg Heart, (3)Loc Left Lung, (4)Seg Left Lung, (5)Loc Right Lung, (6)Seg Right Lung
            # train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLeftLung, val_loader_vindrcxrtLeftLung, train_loader_vindrcxrRightLung, val_loader_vindrcxrtRightLung, data_loader_train, data_loader_val,
            TaskHead_number = (epoch - 1) % 6 # was 2 or 6
            # EMA Update Epoch-wise
            if args.modelEMA == "True_Epoch":
                model_ema.eval()
                if epoch >= Num_EPOCH_Iterative_Steps_MomentumSchduler:
                    coff = 0.5
                else:
                    coff = (momentum_schedule[epoch] - 0.9) * 5 # Epoch-wise
                # coff = (momentum_schedule[epoch] - 0.9) * 5

            if TaskHead_number == 0:  ### LR update --->  Backbone -- Segmentor -- Localizer
                lrBackbone_ = step_decay(epoch, args.lr_backbone, args.total_epochs, step_inc=20)
                lrSegmentor_ = step_decay(epoch, args.lr_segmentor, args.total_epochs, step_inc=20)
                lrLocalizer_ = step_decay(epoch, args.lr, args.total_epochs, step_inc=20)
                if len(optimizer.param_groups) == 2 or len(optimizer.param_groups) == 3:
                    optimizer.param_groups[0]['lr'] = lrBackbone_
                    optimizer.param_groups[1]['lr'] = lrSegmentor_
                    optimizer.param_groups[2]['lr'] = lrLocalizer_                    
                    print('Epoch{} - TaskHead{} - learning_rateBackbone [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
                    print('Epoch{} - TaskHead{} - learning_rateSegmentor [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[1]['lr']), file=log_writter_SEGMENTATION)
                    print('Epoch{} - TaskHead{} - learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[2]['lr']), file=log_writter_SEGMENTATION)
                else:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
                    print('Epoch{} - TaskHead{} - learning_rate [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
            else:
                if len(optimizer.param_groups) == 2 or len(optimizer.param_groups) == 3:
                    print('Epoch{} - TaskHead{} - learning_rateBackbone: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
                    print('Epoch{} - TaskHead{} - learning_rateSegmentor: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[1]['lr']), file=log_writter_SEGMENTATION)
                    print('Epoch{} - TaskHead{} - learning_rateLocalizer: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[2]['lr']), file=log_writter_SEGMENTATION)
                else:
                    print('Epoch{} - TaskHead{} - learning_rate: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_SEGMENTATION)
            if args.modelEMA == "True_Epoch" or args.modelEMA == "True_Iteration":    
                print('Epoch{} - TaskHead{} - coff: {:.10} | {:.10}'.format(epoch, TaskHead_number, (1-coff), coff), file=log_writter_SEGMENTATION)


            for param in model.parameters():
                param.requires_grad = True
            # for param in model_ema.parameters():
            #     param.requires_grad = False

            # after_value_layernorm = sum(model.module.backbone[0].layers[3].blocks[1].mlp.fc2.weight)
            # if sum(old_value_layernorm).item() == sum(after_value_layernorm).item():
            #     print("[Check Freeze] Backbone Component Frozen.", file=log_writter_SEGMENTATION)

            # if args.distributed:   # Active for Detection
            #     sampler_train.set_epoch(epoch)

            # for name, param in model.named_parameters():
            #     if 'segmentation_' in name:
            #         param.requires_grad = False
            #         freeze_check = 1
            # print("[Check Freeze] Segmentation Component Frozen:", freeze_check, file=log_writter_SEGMENTATION)
            # for name, param in model.named_parameters():
            #     if 'transformer' in name: # Dino Decoder
            #         param.requires_grad = False
            #         freeze_check = 1
            # print("[Check Freeze] Localization Component Frozen:", freeze_check, file=log_writter_SEGMENTATION)
            for name, param in model.named_parameters():
                if 'backbone' in name and 'segmentation_' not in name:
                    param.requires_grad = False
            print("[Check Freeze] Backbone Component Frozen.", file=log_writter_SEGMENTATION)


            start_time = time.time()
            print('-- Epoch {} TaskHead {} --'.format(epoch, TaskHead_number), file=log_writter_SEGMENTATION)
            task_todo = "None"

            ### Training Phase ###
            ## | Ignore 2,3,4,5 = LocSeg Heart | Ignore 0,1,4,5 = LocSeg LeftLung | Ignore 0,1,2,3 = LocSeg RightLung | 
            if args.cyclictask == 'heart':
                if TaskHead_number in [2,3,4,5]: ## Ignore
                    continue
            elif args.cyclictask == 'leftlung':
                if TaskHead_number in [0,1,4,5]: ## Ignore
                    continue
            elif args.cyclictask == 'rightlung':
                if TaskHead_number in [0,1,2,3]: ## Ignore
                    continue
            elif args.cyclictask == 'heart_leftlung':
                if TaskHead_number in [4,5]: ## Ignore
                    continue
            elif args.cyclictask == 'leftlung_rightlung':
                if TaskHead_number in [0,1]: ## Ignore
                    continue
            elif args.cyclictask == 'heart_leftlung_rightlung':
                check = 0 ## Ignore nothing
            else:
                print("[W A R N N I N G] Cyclic Task Error! ", args.cyclictask)
                exit(0)


            ### Localization Vindr-CXR Organ ==> (1) Heart, (2) Left Lung, (3) Right Lung
            ## Segmentation ==> train_loader_vindrcxrHeart, train_loader_vindrcxrLeftLung, train_loader_vindrcxrRightLung
            if TaskHead_number == 0: # Fine-tune for Localization - Heart
                task_todo = "Localization_Heart_Train"
                model.task_DetHead = 0
                model_ema.task_DetHead = 0
                train_stats = train_one_epoch(
                    model, criterion, data_loader_trainHeart, optimizer, device, epoch,
                    args.clip_max_norm, wo_class_error=wo_class_error, DetHead=0, lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                log_stats = { **{f'train_Heart_{k}': v for k, v in train_stats.items()} }
                result_output_dir = args.output_dir + '/results.txt'
                log_writer_detection = open(result_output_dir, 'a')
                log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: Heart " + '\n')
                log_writer_detection.write('-- Training --' + '\n')
                formatted_stats_train = {f'train_Heart_{k}': v for k, v in train_stats.items()}
                for key, value in formatted_stats_train.items():
                    log_writer_detection.write(f'{key}: {value}\n')
                log_writer_detection.write('\n')
                log_writer_detection.close()

            if TaskHead_number == 1: # Fine-tune for Segmentation - Heart || head_number = segmentation head number
                task_todo = "Segmentation_Heart_Train"
                train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader_vindrcxrHeart, optimizer, loss_scaler, epoch, head_number=0, log_writter_SEGMENTATION=log_writter_SEGMENTATION, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                print( "Epoch {:04d}: Train Heart Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_SEGMENTATION )

            if TaskHead_number == 2: # Fine-tune for Localization - Left Lung
                task_todo = "Localization_LeftLung_Train"
                model.task_DetHead = 1
                model_ema.task_DetHead = 1
                train_stats = train_one_epoch(
                    model, criterion, data_loader_trainLeftLung, optimizer, device, epoch,
                    args.clip_max_norm, wo_class_error=wo_class_error, DetHead=1, lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                log_stats = { **{f'train_LeftLung_{k}': v for k, v in train_stats.items()} }
                result_output_dir = args.output_dir + '/results.txt'
                log_writer_detection = open(result_output_dir, 'a')
                log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: Left Lung " + '\n')
                log_writer_detection.write('-- Training --' + '\n')
                formatted_stats_train = {f'train_LeftLung_{k}': v for k, v in train_stats.items()}
                for key, value in formatted_stats_train.items():
                    log_writer_detection.write(f'{key}: {value}\n')
                log_writer_detection.write('\n')
                log_writer_detection.close()

            if TaskHead_number == 3: # Fine-tune for Segmentation - Left Lung || head_number = segmentation head number
                task_todo = "Segmentation_LeftLung_Train"
                train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader_vindrcxrLeftLung, optimizer, loss_scaler, epoch, head_number=1, log_writter_SEGMENTATION=log_writter_SEGMENTATION, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                print( "Epoch {:04d}: Train Left Lung Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_SEGMENTATION )
            
            if TaskHead_number == 4: # Fine-tune for Localization - Right Lung
                task_todo = "Localization_RightLung_Train"
                model.task_DetHead = 2
                model_ema.task_DetHead = 2
                train_stats = train_one_epoch(
                    model, criterion, data_loader_trainRightLung, optimizer, device, epoch,
                    args.clip_max_norm, wo_class_error=wo_class_error, DetHead=2, lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                log_stats = { **{f'train_RightLung{k}': v for k, v in train_stats.items()} }
                result_output_dir = args.output_dir + '/results.txt'
                log_writer_detection = open(result_output_dir, 'a')
                log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: Right Lung " + '\n')
                log_writer_detection.write('-- Training --' + '\n')
                formatted_stats_train = {f'train_RightLung_{k}': v for k, v in train_stats.items()}
                for key, value in formatted_stats_train.items():
                    log_writer_detection.write(f'{key}: {value}\n')
                log_writer_detection.write('\n')
                log_writer_detection.close()

            if TaskHead_number == 5: # Fine-tune for Segmentation - Right Lung || head_number = segmentation head number
                task_todo = "Segmentation_RightLung_Train"
                train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader_vindrcxrRightLung, optimizer, loss_scaler, epoch, head_number=2, log_writter_SEGMENTATION=log_writter_SEGMENTATION, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                print( "Epoch {:04d}: Train Right Lung Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_SEGMENTATION )

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print(' -- Train time for Epoch {} - TaskHead {}: {}'.format(epoch, TaskHead_number, total_time_str), file=log_writter_SEGMENTATION)

            ## Model EMA/Teacher Update -- Fix: updating EMA_Model within the trainer function
            # if args.modelEMA == "True_Epoch":
            #     ema_update_teacher(model, model_ema, momentum_schedule, epoch)
            #     ### model_ema = model # should ignore later # DEBUG purpose

            save_file = os.path.join(model_path_SEGMENTATION, 'ckpt_E'+str(epoch)+'_TH'+str(TaskHead_number)+'.pth')
            save_model(model, optimizer, log_writter_SEGMENTATION, epoch, save_file, model_ema=model_ema)
            # print('\n', file=log_writter_SEGMENTATION)


            ### Testing Phase ###
            start_time = time.time()

            ## Test/Eval for Localization - All [Heart, Left Lung, Right Lung]
            if 'heart' in args.cyclictask:
                model.task_DetHead = 0 ## Localization Heart
                test_stats, coco_evaluator, features_detectionList = evaluate(
                    model, criterion, postprocessors, data_loader_valHeart, base_ds_Heart, device, args.output_dir,
                    DetHead=0, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None))
                log_stats = { **{f'test_Heart_{k}': v for k, v in test_stats.items()} }
                result_output_dir = args.output_dir + '/results.txt'
                log_writer_detection = open(result_output_dir, 'a')
                log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Testing: Heart " + '\n')
                log_writer_detection.write('-- Testing --' + '\n')
                formatted_stats_test = {f'test_Heart_{k}': v for k, v in test_stats.items()}
                for key, value in formatted_stats_test.items():
                    log_writer_detection.write(f'{key}: {value}\n')
                log_writer_detection.write('\n')
                log_writer_detection.write('\n')
                log_writer_detection.close()
                fields=[epoch, 'Vindr-Organ', task_todo, 'Student', 'Localization_Heart_Test', str(100*value[0]), '-'] # AUC_SliceLevel_Res
                with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
                model_ema.task_DetHead = 0
                test_stats, coco_evaluator, features_detectionList = evaluate(
                    model_ema, criterion_ema, postprocessors_ema, data_loader_valHeart, base_ds_Heart, device, args.output_dir,
                    DetHead=0, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None))
                log_stats = { **{f'test_Heart_{k}': v for k, v in test_stats.items()} }
                result_output_dir = args.output_dir + '/results.txt'
                log_writer_detection = open(result_output_dir, 'a')
                log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " TestingEMA: Heart " + '\n')
                log_writer_detection.write('-- Testing Teacher --' + '\n')
                formatted_stats_test = {f'testEMA_Heart_{k}': v for k, v in test_stats.items()}
                for key, value in formatted_stats_test.items():
                    log_writer_detection.write(f'{key}: {value}\n')
                log_writer_detection.write('\n')
                log_writer_detection.write('\n')
                log_writer_detection.close()
                fields=[epoch, 'Vindr-Organ', task_todo, 'Teacher', 'Localization_Heart_Test', str(100*value[0]), '-'] # AUC_SliceLevel_Res
                with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)

            if 'leftlung' in args.cyclictask:
                model.task_DetHead = 1 ## Localization Left Lung
                test_stats, coco_evaluator, features_detectionList = evaluate(
                    model, criterion, postprocessors, data_loader_valLeftLung, base_ds_LeftLung, device, args.output_dir,
                    DetHead=1, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None))
                log_stats = { **{f'test_LeftLung_{k}': v for k, v in test_stats.items()} }
                result_output_dir = args.output_dir + '/results.txt'
                log_writer_detection = open(result_output_dir, 'a')
                log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Testing: Left Lung " + '\n')
                log_writer_detection.write('-- Testing --' + '\n')
                formatted_stats_test = {f'test_LeftLung_{k}': v for k, v in test_stats.items()}
                for key, value in formatted_stats_test.items():
                    log_writer_detection.write(f'{key}: {value}\n')
                log_writer_detection.write('\n')
                log_writer_detection.write('\n')
                log_writer_detection.close()
                fields=[epoch, 'Vindr-Organ', task_todo, 'Student', 'Localization_LeftLung_Test', str(100*value[0]), '-'] # AUC_SliceLevel_Res
                with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
                model_ema.task_DetHead = 1
                test_stats, coco_evaluator, features_detectionList = evaluate(
                    model_ema, criterion_ema, postprocessors_ema, data_loader_valLeftLung, base_ds_LeftLung, device, args.output_dir,
                    DetHead=1, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None))
                log_stats = { **{f'test_Heart_{k}': v for k, v in test_stats.items()} }
                result_output_dir = args.output_dir + '/results.txt'
                log_writer_detection = open(result_output_dir, 'a')
                log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " TestingEMA: Left Lung " + '\n')
                log_writer_detection.write('-- Testing Teacher --' + '\n')
                formatted_stats_test = {f'testEMA_LeftLung_{k}': v for k, v in test_stats.items()}
                for key, value in formatted_stats_test.items():
                    log_writer_detection.write(f'{key}: {value}\n')
                log_writer_detection.write('\n')
                log_writer_detection.write('\n')
                log_writer_detection.close()
                fields=[epoch, 'Vindr-Organ', task_todo, 'Teacher', 'Localization_LeftLung_Test', str(100*value[0]), '-'] # AUC_SliceLevel_Res
                with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)

            if 'rightlung' in args.cyclictask:
                model.task_DetHead = 2 ## Localization Right Lung
                test_stats, coco_evaluator, features_detectionList = evaluate(
                    model, criterion, postprocessors, data_loader_valRightLung, base_ds_RightLung, device, args.output_dir,
                    DetHead=2, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None))
                log_stats = { **{f'test_RightLung_{k}': v for k, v in test_stats.items()} }
                result_output_dir = args.output_dir + '/results.txt'
                log_writer_detection = open(result_output_dir, 'a')
                log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Testing: Right Lung " + '\n')
                log_writer_detection.write('-- Testing --' + '\n')
                formatted_stats_test = {f'test_RightLung_{k}': v for k, v in test_stats.items()}
                for key, value in formatted_stats_test.items():
                    log_writer_detection.write(f'{key}: {value}\n')
                log_writer_detection.write('\n')
                log_writer_detection.write('\n')
                log_writer_detection.close()
                fields=[epoch, 'Vindr-Organ', task_todo, 'Student', 'Localization_RightLung_Test', str(100*value[0]), '-'] # AUC_SliceLevel_Res
                with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
                model_ema.task_DetHead = 2
                test_stats, coco_evaluator, features_detectionList = evaluate(
                    model_ema, criterion_ema, postprocessors_ema, data_loader_valRightLung, base_ds_RightLung, device, args.output_dir,
                    DetHead=2, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None))
                log_stats = { **{f'test_RightLung_{k}': v for k, v in test_stats.items()} }
                result_output_dir = args.output_dir + '/results.txt'
                log_writer_detection = open(result_output_dir, 'a')
                log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " TestingEMA: Right Lung " + '\n')
                log_writer_detection.write('-- Testing Teacher --' + '\n')
                formatted_stats_test = {f'testEMA_RightLung_{k}': v for k, v in test_stats.items()}
                for key, value in formatted_stats_test.items():
                    log_writer_detection.write(f'{key}: {value}\n')
                log_writer_detection.write('\n')
                log_writer_detection.write('\n')
                log_writer_detection.close()
                fields=[epoch, 'Vindr-Organ', task_todo, 'Teacher', 'Localization_RightLung_Test', str(100*value[0]), '-'] # AUC_SliceLevel_Res
                with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)


            # Test/Eval for Segmentation - Heart
            if 'leftlung' in args.cyclictask:
                test_y, test_p, features_segmentationList = test_SEGMENTATION(model, val_loader_vindrcxrtHeart, head_number=0, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print("[INFO] Student Vindr-CXR Organ Heart Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
                print("[INFO] Student Vindr-CXR Organ Heart Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)))
                fields=[epoch, 'Vindr-Organ', task_todo, 'Student', 'Segmentation_Heart_Test', '-', str(100.0 * dice_score(test_p, test_y))] # AUC_SliceLevel_Res
                with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
                test_y, test_p, features_segmentationList = test_SEGMENTATION(model_ema, val_loader_vindrcxrtHeart, head_number=0, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print("[INFO] Teacher Vindr-CXR Organ Heart Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
                print("[INFO] Teacher Vindr-CXR Organ Heart Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)))
                fields=[epoch, 'Vindr-Organ', task_todo, 'Teacher', 'Segmentation_Heart_Test', '-', str(100.0 * dice_score(test_p, test_y))] # AUC_SliceLevel_Res
                with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)

            ## Test/Eval for Segmentation - Left Lung
            if 'leftlung' in args.cyclictask:
                test_y, test_p, features_segmentationList = test_SEGMENTATION(model, val_loader_vindrcxrtLeftLung, head_number=1, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print("[INFO] Student Vindr-CXR Organ Left Lung Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
                print("[INFO] Student Vindr-CXR Organ Left Lung Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)))
                fields=[epoch, 'Vindr-Organ', task_todo, 'Student', 'Segmentation_LeftLung_Test', '-', str(100.0 * dice_score(test_p, test_y))] # AUC_SliceLevel_Res
                with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
                test_y, test_p, features_segmentationList = test_SEGMENTATION(model_ema, val_loader_vindrcxrtLeftLung, head_number=1, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print("[INFO] Teacher Vindr-CXR Organ Left Lung Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
                print("[INFO] Teacher Vindr-CXR Organ Left Lung Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)))
                fields=[epoch, 'Vindr-Organ', task_todo, 'Teacher', 'Segmentation_LeftLung_Test', '-', str(100.0 * dice_score(test_p, test_y))] # AUC_SliceLevel_Res
                with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)

            # # Test/Eval for Segmentation - Right Lung
            if 'rightlung' in args.cyclictask:
                test_y, test_p, features_segmentationList = test_SEGMENTATION(model, val_loader_vindrcxrtRightLung, head_number=2, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print("[INFO] Student Vindr-CXR Organ Right Lung Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
                print("[INFO] Student Vindr-CXR Organ Right Lung Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)))
                fields=[epoch, 'Vindr-Organ', task_todo, 'Student', 'Segmentation_RightLung_Test', '-', str(100.0 * dice_score(test_p, test_y))] # AUC_SliceLevel_Res
                with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)
                test_y, test_p, features_segmentationList = test_SEGMENTATION(model_ema, val_loader_vindrcxrtRightLung, head_number=2, log_writter_SEGMENTATION=log_writter_SEGMENTATION)
                print("[INFO] Teacher Vindr-CXR Organ Right Lung Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
                print("[INFO] Teacher Vindr-CXR Organ Right Lung Dice = {:.8f}%".format(100.0 * dice_score(test_p, test_y)))
                fields=[epoch, 'Vindr-Organ', task_todo, 'Teacher', 'Segmentation_RightLung_Test', '-', str(100.0 * dice_score(test_p, test_y))] # AUC_SliceLevel_Res
                with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Test time Epoch {} TaskHead {} -- {}\n\n'.format(epoch, TaskHead_number, total_time_str), file=log_writter_SEGMENTATION)
            log_writter_SEGMENTATION.flush()






    total_time = time.time() - start_time_ALL
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total time {}'.format(total_time_str))
    if args.taskcomponent == 'classification':
        print('Total time {}'.format(total_time_str), file=log_writter_CLASSIFICATION)
    if args.taskcomponent == 'segmentation':
        print('Total time {}'.format(total_time_str), file=log_writter_SEGMENTATION)

    # remove the copied files.
    copyfilelist = vars(args).get('copyfilelist')
    if copyfilelist and args.local_rank == 0:
        from datasets.data_util import remove
        for filename in copyfilelist:
            print("Removing: {}".format(filename))
            remove(filename)





if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
