# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------

import os 

# os.environ['MASTER_ADDR'] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '29505'

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

import datasets
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch, test, test_NAD
from sklearn.metrics import roc_auc_score, accuracy_score

# Segmentation from Jiaxuan
from utils_segmentation import load_popar_weight, AverageMeter, save_model, dice_score, mean_dice_coef, torch_dice_coef_loss, exp_lr_scheduler_with_warmup, step_decay, load_swin_pretrained
from datasets_medical import build_transform_segmentation, PXSDataset, MontgomeryDataset, JSRTClavicleDataset, JSRTHeartDataset,JSRTLungDataset, VinDrRibCXRDataset
from timm.utils import NativeScaler, ModelEma
import math

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer, create_optimizer_v2, optimizer_kwargs


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
    parser.add_argument('--numClasses', type=int, default=1000) 
    parser.add_argument('--backbonemodel', default='Swin-L', help='Swin-T, Swin-B, Swin-L') 

    # training parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--note', default='',
                        help='add some notes to the experiment')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint') 
    parser.add_argument('--backbone_dir', default=None, type=str, help='loading backbone weights')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--save_results', action='store_true')
    parser.add_argument('--save_log', action='store_true')

    parser.add_argument('--taskcomponent', default='detection', help='classification | segmentation | detection')



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
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 5e-4)')
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


def train_one_epoch_SEGMENTATION(model, train_loader, optimizer, loss_scaler, epoch, log_writter_SEGMENTATION):
    model.train(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    criterion = torch_dice_coef_loss
    end = time.time()

    for idx, (img,mask) in enumerate(train_loader):
        data_time.update(time.time() - end)
        bsz = img.shape[0]


        # img = img.double().cuda(non_blocking=True) # was active
        # mask = mask.double().cuda(non_blocking=True) # was active
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
            out_SegmentationHead = model.module.backbone[0].extra_features_seg(img)
        else:
            # out_features, out_classifierHead, out_SegmentationHead = model.backbone(img)
            out_SegmentationHead = model.backbone[0].extra_features_seg(img)
        outputs = torch.sigmoid( out_SegmentationHead )

        # del out_features, out_classifierHead, out_SegmentationHead
        del out_SegmentationHead


        loss = criterion(mask, outputs)
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


        if (idx + 1) % 10 == 0:
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
    return losses.avg

def evaluation_SEGMENTATION(model, val_loader, epoch, log_writter_SEGMENTATION):
    model.eval()
    losses = AverageMeter()
    criterion = torch_dice_coef_loss

    with torch.no_grad():
        for idx, (img, mask) in enumerate(val_loader):
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
                out_SegmentationHead = model.module.backbone[0].extra_features_seg(img)
            else:
                # out_features, out_classifierHead, out_SegmentationHead = model.backbone(img)
                out_SegmentationHead = model.backbone[0].extra_features_seg(img)
            outputs = torch.sigmoid( out_SegmentationHead )
            # del out_features, out_classifierHead, out_SegmentationHead

            loss = criterion(mask, outputs)

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), file=log_writter_SEGMENTATION)
                sys.exit(1)
                # update metric
            losses.update(loss.item(), bsz)


            torch.cuda.synchronize()


            if (idx + 1) % 10 == 0:
                print('Evaluation: [{0}][{1}/{2}]\t'
                      'Total loss {ttloss.val:.5f} ({ttloss.avg:.5f})'.format(
                    epoch, idx + 1, len(val_loader), ttloss=losses), file=log_writter_SEGMENTATION)
                log_writter_SEGMENTATION.flush()
                # if conf.debug_mode:
                #     break
    return losses.avg

def test_SEGMENTATION(model, test_loader, log_writter_SEGMENTATION):

    # checkpoint = torch.load(checkpoint_path, map_location='cpu')
    # checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
    # model.load_state_dict(checkpoint_model)

    # model = model.cuda()
    # if torch.cuda.is_available():
    #     model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    #     model = model.cuda()
    #     cudnn.benchmark = True

    model.eval()
    with torch.no_grad():
        test_p = None
        test_y = None
        for idx, (img, mask) in enumerate(test_loader):
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
                    out_SegmentationHead = model.module.backbone[0].extra_features_seg(img)
                else:
                    # out_features, out_classifierHead, out_SegmentationHead = model.backbone(img)
                    out_SegmentationHead = model.backbone[0].extra_features_seg(img)
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
                if (idx + 1) % 20 == 0:
                    print("Testing Step[{}/{}] ".format(idx + 1, len(test_loader)), file=log_writter_SEGMENTATION)
                    log_writter_SEGMENTATION.flush()
                    # if conf.debug_mode:
                    #     break



        print("Done testing iteration!", file=log_writter_SEGMENTATION)
        log_writter_SEGMENTATION.flush()

    test_p = test_p.numpy()
    test_y = test_y.numpy()
    test_y = test_y.reshape(test_p.shape)
    return test_y, test_p



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
        out_classifierHead = model.module.backbone[0].extra_features(images)
      else:
        # _, out_classifierHead, _ = model.backbone(images)
        out_classifierHead = model.backbone[0].extra_features(images)
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

    # build model
    model, criterion, postprocessors = build_model_main(args)
    wo_class_error = False
    model.to(device)

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
    logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))

    param_dicts = get_param_dict(args, model_without_ddp)

    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    if args.taskcomponent == 'detection':
        dataset_train = build_dataset(image_set='train', args=args)
        dataset_val = build_dataset(image_set='val', args=args)
        dataset_test = build_dataset(image_set='test', args=args) # added by Nahid

        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_test = DataLoader(dataset_test, 1, sampler=sampler_test,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


        if args.onecyclelr:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(data_loader_train), epochs=args.epochs, pct_start=0.2)
        elif args.multi_step_lr:
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drop_list)
        else:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    if args.taskcomponent == 'detection':
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
    if args.resume:

        # df = open("model_InternImageT_DINO_w.txt",'w')
        # checkpoint_w = model_without_ddp.state_dict()
        # for name, _ in checkpoint_w.items():
        #     df.write(name + "\n")
        # df.close()

        # df = open("checkpoint_InternImageT_DINO_w.txt",'w')
        # checkpoint_w = torch.load(args.resume, map_location='cpu')['state_dict'] # state_dict for InternImageT_with_Dino checkpoint
        # for name, _ in checkpoint_w.items():
        #     df.write(name + "\n")
        # df.close()


        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        try:
            model_without_ddp.load_state_dict(checkpoint['model'], strict=False) # strict=False added for integrated model
            # model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
            model_summary = str(model_without_ddp)
            print("[Model Info.] SwinL+Dino pretrained model loaded.")
            with open("model_summary_SwinL_DINO.txt", "w") as f:
                f.write(model_summary)
        
        except:
            # model_without_ddp.load_state_dict(checkpoint['state_dict'])

            checkpoint = torch.load(args.resume, map_location='cpu')['state_dict'] # state_dict for InternImageT_with_Dino checkpoint

            model_dict = model_without_ddp.state_dict()
            new_dict = {}
            for k, v in checkpoint.items(): # was state_dict for RSNASlice
                if k.startswith("backbone."):
                    temp_k = k.replace("backbone.", "backbone.0.")
                    new_dict[temp_k] = v
                    continue

                if "bbox_head." in k:
                    temp_k = k.replace("bbox_head.", "")
                if "level_embeds" in temp_k:
                    temp_k = temp_k.replace("level_embeds", "level_embed")
                if "attentions.0" in temp_k:
                    temp_k = temp_k.replace("attentions.0", "self_attn")
                if "attentions.1" in temp_k:
                    temp_k = temp_k.replace("attentions.1", "cross_attn")
                if "self_attn.attn" in temp_k:
                    temp_k = temp_k.replace("self_attn.attn", "self_attn")
                if "norms.0" in temp_k:
                    temp_k = temp_k.replace("norms.0", "norm1")
                if "norms.1" in temp_k:
                    temp_k = temp_k.replace("norms.1", "norm2")
                if "norms.2" in temp_k:
                    temp_k = temp_k.replace("norms.2", "norm3")
                if "ffns.0.layers.0.0" in temp_k:
                    temp_k = temp_k.replace("ffns.0.layers.0.0", "linear1")
                    new_dict[temp_k] = v
                if "ffns.0.layers.1" in temp_k:
                    temp_k = temp_k.replace("ffns.0.layers.1", "linear2")
                if "ref_point_head.0" in temp_k:
                    temp_k = temp_k.replace("ref_point_head.0", "ref_point_head.layers.0")
                if "ref_point_head.2" in temp_k:
                    temp_k = temp_k.replace("ref_point_head.2", "ref_point_head.layers.1")
                if "query_embed" in temp_k:
                    temp_k = temp_k.replace("query_embed", "tgt_embed")
                if "label_embedding" in temp_k:
                    temp_k = temp_k.replace("label_embedding", "label_enc")
                if "reg_branches.0.0.weight" in temp_k: # not sure
                    temp_k = temp_k.replace("reg_branches.0.0.weight", "transformer.decoder.bbox_embed.0.0.weight")
                new_dict[temp_k] = v


            # ignore_keys = ['conv_head.0.weight', 'conv_head.1.0.weight', 'conv_head.1.0.bias', 'conv_head.1.0.running_mean', 'conv_head.1.0.running_var', 'conv_head.1.0.num_batches_tracked', 'head.weight', 'head.bias']
            # new_dict = {k: v for k, v in new_dict.items() if k in model_dict and k not in ignore_keys}

            df = open("checkpoint_MODIFIED_InternImageT_DINO_w.txt",'w')
            for name, _ in new_dict.items():
                df.write(name + "\n")
            df.close()

            model_summary = str(model_without_ddp)
            with open("model_summary_InternImageT_DINO.txt", "w") as f:
                f.write(model_summary)

            _tmp_st_output = model_without_ddp.load_state_dict(new_dict, strict=False)
            print("------")
            print("[Model Info]", str(_tmp_st_output))



        if args.use_ema:
            if 'ema_model' in checkpoint:
                ema_m.module.load_state_dict(utils.clean_state_dict(checkpoint['ema_model']))
            else:
                del ema_m
                ema_m = ModelEma(model, args.ema_decay)                

        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

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


    if args.taskcomponent == 'detection' and args.eval:
        os.environ['EVAL_FLAG'] = 'TRUE'
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
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




    if args.taskcomponent == 'segmentation':
        
        if args.backbone_dir is not None:
            renaming_dict = {
                'norm.weight': 'module.backbone.classification_norm.weight',
                'norm.bias': 'module.backbone.classification_norm.bias',
                # 'head.weight': 'backbone.classification_head.weight',
                # 'head.bias': 'backbone.classification_head.bias'
            }
            old_value_normW = sum(model.module.backbone[0].norm.weight)
            old_value_layernorm = sum(model.module.backbone[0].layers[2].blocks[11].norm1.weight)

            new_state_dict = {}
            prefix = "module.backbone.0."
            checkpoint = torch.load(args.backbone_dir, map_location='cpu')
            state_dict = checkpoint['model']
            for key, value in state_dict.items():
                if "head.weight" in key or "head.bias" in key:
                    continue
                new_key = prefix + key
                new_state_dict[new_key] = value
            status_w = model.load_state_dict(new_state_dict, strict=False)

            new_state_dict = {}
            for old_key, new_key in renaming_dict.items():
                new_state_dict[new_key] = state_dict[old_key]

            status_w = model.load_state_dict(new_state_dict, strict=False)

            # torch.nn.init.constant_(model.module.backbone[0].head.bias, 0.)
            # torch.nn.init.constant_(model.module.backbone[0].head.weight, 0.)
            # print(f"Error in loading classifier head, re-init classifier head to 0")

            # print(status_w)
            new_value_normW = sum(model.module.backbone[0].norm.weight)
            new_value_layernorm = sum(model.module.backbone[0].layers[2].blocks[11].norm1.weight)

            print("[Model Info.] Pretrained weights loaded for backbone:", args.backbone_dir)
            print("[Model CHECK] Loaded backbone's norm.weight and norm.bias.", old_value_layernorm, new_value_layernorm)
            print("[Model CHECK] Loaded backbone's norm.weight and norm.bias.", old_value_normW, new_value_normW)
            del checkpoint, state_dict, new_state_dict


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

        args.epochs = 500
        img_size = args.imgsize # 448 worked
        batch_size = args.batch_size
        num_workers = 4
        best_val_loss_SEGMENTATION = 100000
        patience_SEGMENTATION = 50
        patience_counter_SEGMENTATION = 0

        print()
        print("-------------")
        print("[Information]  TASK:", "Segmentation")
        print("[Information]  Backbone:", args.backbonemodel)
        # print("[Information]  Backbone Weights:", args.backbone_dir)
        print("[Information]  Dataset:", args.segmentation_dataset)
        print("[Information]  Total Epoch:", args.epochs)
        print("[Information]  Image Size:", img_size)
        print("[Information]  Batch Size:", batch_size)
        print("[Information]  Num Workers:", num_workers)
        print("[Information]  Optimizer:", "SGD")
        print("[Information]  Patience:", patience_SEGMENTATION)
        print("[Information]  Output Dir:", args.output_dir)
        print("-------------")
        print() # log_writter_SEGMENTATION

        print("-------------", file=log_writter_SEGMENTATION)
        print("[Information]  TASK:", "Segmentation", file=log_writter_SEGMENTATION)
        print("[Information]  Backbone:", args.backbonemodel, file=log_writter_SEGMENTATION)
        # print("[Information]  Backbone Weights:", args.backbone_dir, file=log_writter_SEGMENTATION)
        print("[Information]  Dataset:", args.segmentation_dataset, file=log_writter_SEGMENTATION)
        print("[Information]  Total Epoch:", args.epochs, file=log_writter_SEGMENTATION)
        print("[Information]  Image Size:", img_size, file=log_writter_SEGMENTATION)
        print("[Information]  Batch Size:", batch_size, file=log_writter_SEGMENTATION)
        print("[Information]  Num Workers:", num_workers, file=log_writter_SEGMENTATION)
        print("[Information]  Optimizer:", "SGD", file=log_writter_SEGMENTATION)
        print("[Information]  Patience:", patience_SEGMENTATION, file=log_writter_SEGMENTATION)
        print("[Information]  Output Dir:", args.output_dir, file=log_writter_SEGMENTATION)
        print("-------------", file=log_writter_SEGMENTATION)


        loss_scaler = NativeScaler()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0, momentum=0.9, nesterov=False)

        if args.segmentation_dataset == 'jsrt_lung':
            train_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/train.txt")]
            val_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/val.txt")]
            test_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/test.txt")]

            train_dataset = JSRTLungDataset(train_image_path_file, image_size=(img_size,img_size))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       num_workers=num_workers, pin_memory=True, shuffle=True,drop_last=True )

            val_dataset = JSRTLungDataset(val_image_path_file,image_size=(img_size,img_size), mode="val")
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                                                     pin_memory=True, shuffle=True,drop_last=False )

            test_dataset = JSRTLungDataset(test_image_path_file,image_size=(img_size,img_size), mode="val")
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                      num_workers=num_workers, pin_memory=True,drop_last=False )
        elif args.segmentation_dataset == 'jsrt_clavicle':
            train_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/train.txt")]
            val_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/val.txt")]
            test_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/test.txt")]

            train_dataset = JSRTClavicleDataset(train_image_path_file, image_size=(img_size,img_size))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       num_workers=num_workers, pin_memory=True, shuffle=True,drop_last=True )

            val_dataset = JSRTClavicleDataset(val_image_path_file,image_size=(img_size,img_size), mode="val")
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers,
                                                     pin_memory=True, shuffle=True,drop_last=False )

            test_dataset = JSRTClavicleDataset(test_image_path_file,image_size=(img_size,img_size), mode="val")

            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                      num_workers=num_workers, pin_memory=True,drop_last=False )

        if args.test:
            print()
            print("[CHECK-Testing] Segmentation Model.")
            checkpoint = torch.load(args.pretrain_model_path, map_location='cpu')
            checkpoint_model = checkpoint['model']
            # checkpoint_model = {k.replace("module.", ""): v for k, v in checkpoint['model'].items()}
            model.load_state_dict(checkpoint_model)
            # print()
            print("[MODEL INFO.] Pretrained model loaded for Segmentation...")

            test_y, test_p = test_SEGMENTATION( model, test_loader, log_writter_SEGMENTATION )
            print("[INFO] Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)), file=log_writter_SEGMENTATION)
            print("Mean Dice = {:.4f}".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)), file=log_writter_SEGMENTATION)

            print("[INFO] Dice = {:.2f}%".format(100.0 * dice_score(test_p, test_y)))
            print("Mean Dice = {:.4f}".format(mean_dice_coef(test_y > 0.5, test_p > 0.5)))

            log_writter_SEGMENTATION.flush()

            exit(0)

    if args.taskcomponent == 'classification':
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if args.backbone_dir is not None:
            renaming_dict = {
                'norm.weight': 'module.backbone.classification_norm.weight',
                'norm.bias': 'module.backbone.classification_norm.bias',
                # 'head.weight': 'backbone.classification_head.weight',
                # 'head.bias': 'backbone.classification_head.bias'
            }
            old_value_normW = sum(model.module.backbone[0].norm.weight)
            old_value_layernorm = sum(model.module.backbone[0].layers[2].blocks[11].norm1.weight)

            new_state_dict = {}
            prefix = "module.backbone.0."
            checkpoint = torch.load(args.backbone_dir, map_location='cpu')
            state_dict = checkpoint['model']
            for key, value in state_dict.items():
                if "head.weight" in key or "head.bias" in key:
                    continue
                new_key = prefix + key
                new_state_dict[new_key] = value
            status_w = model.load_state_dict(new_state_dict, strict=False)

            new_state_dict = {}
            for old_key, new_key in renaming_dict.items():
                new_state_dict[new_key] = state_dict[old_key]

            status_w = model.load_state_dict(new_state_dict, strict=False)

            # torch.nn.init.constant_(model.module.backbone[0].head.bias, 0.)
            # torch.nn.init.constant_(model.module.backbone[0].head.weight, 0.)
            # print(f"Error in loading classifier head, re-init classifier head to 0")

            # print(status_w)
            new_value_normW = sum(model.module.backbone[0].norm.weight)
            new_value_layernorm = sum(model.module.backbone[0].layers[2].blocks[11].norm1.weight)

            print("[Model Info.] Pretrained weights loaded for backbone:", args.backbone_dir)
            print("[Model CHECK] Loaded backbone's norm.weight and norm.bias.", old_value_layernorm, new_value_layernorm)
            print("[Model CHECK] Loaded backbone's norm.weight and norm.bias.", old_value_normW, new_value_normW)
            del checkpoint, state_dict, new_state_dict
            # exit(0)

        model_path_CLASSIFICATION = args.output_dir

        logs_path = os.path.join(model_path_CLASSIFICATION, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if os.path.exists(os.path.join(logs_path, "log.txt")):
            log_writter_CLASSIFICATION = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            log_writter_CLASSIFICATION = open(os.path.join(logs_path, "log.txt"), 'w')


        if args.classification_dataset == "imagenet":
            args.epochs = 300
            img_size = args.imgsize # 448 worked
            args.lr = 0.1
            batch_size = 128 # 128
            num_workers = args.num_workers # 16 | 32
            best_acc1_CLASSIFICATION = 100000

            print()
            print("-------------")
            print("[Information]  TASK:", "Classification")
            print("[Information]  Backbone:", args.backbonemodel)
            # print("[Information]  Backbone Weights:", args.backbone_dir)
            print("[Information]  Dataset:", 'ImageNet1k')
            print("[Information]  Total Epoch:", args.epochs)
            print("[Information]  Image Size:", img_size)
            print("[Information]  Batch Size:", batch_size)
            print("[Information]  Num Workers:", num_workers)
            print("[Information]  Optimizer:", "SGD")
            print("[Information]  Output Dir:", args.output_dir)
            print("-------------")
            print() # file=log_writter_SEGMENTATION

            print("-------------", file=log_writter_CLASSIFICATION)
            print("[Information]  TASK:", "Classification", file=log_writter_CLASSIFICATION)
            print("[Information]  Backbone:", args.backbonemodel, file=log_writter_CLASSIFICATION)
            # print("[Information]  Backbone Weights:", args.backbone_dir, file=log_writter_CLASSIFICATION)
            print("[Information]  Dataset:", 'ImageNet1k', file=log_writter_CLASSIFICATION)
            print("[Information]  Total Epoch:", args.epochs, file=log_writter_CLASSIFICATION)
            print("[Information]  Image Size:", img_size, file=log_writter_CLASSIFICATION)
            print("[Information]  Batch Size:", batch_size, file=log_writter_CLASSIFICATION)
            print("[Information]  Num Workers:", num_workers, file=log_writter_CLASSIFICATION)
            print("[Information]  Optimizer:", "SGD", file=log_writter_CLASSIFICATION)
            print("[Information]  Output Dir:", args.output_dir, file=log_writter_CLASSIFICATION)
            print("-------------", file=log_writter_CLASSIFICATION)

            criterion = nn.CrossEntropyLoss().cuda(args.gpu)
            optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                        momentum=0.9,
                                        weight_decay=1e-4)
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            start_time_dataload = time.time()
            if args.train:
                # print("[Classification] Working on Training dataset...")
                traindir = os.path.join("/scratch/jliang12/data/ImageNet", 'train') # /data/jliang12/rfeng12/Data/ImageNet  /scratch/jliang12/data/ImageNet
                train_dataset = datasets.ImageFolder(
                    traindir,
                    transforms.Compose([
                        transforms.RandomResizedCrop(img_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]))
                if args.distributed:
                    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
                else:
                    train_sampler = None
                print("[INFO.] Distributed:", args.distributed)
                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                    num_workers=num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
            # print("[Classification] Working on Validation dataset...")
            valdir = os.path.join("/scratch/jliang12/data/ImageNet", 'val') # /data/jliang12/rfeng12/Data/ImageNet  /scratch/jliang12/data/ImageNet
            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(img_size),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=True)
            print("[INFO.] Classification Data loaded...")
            end_time_dataload = time.time()
            hours, rem = divmod(end_time_dataload - start_time_dataload, 3600)
            minutes, seconds = divmod(rem, 60)
            print("[Info.] Dataloading time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
            if args.test:
                print()
                print("Classification Validation Started...")
                acc1 = evaluate_CLASSIFICATION(val_loader, model, criterion, args, log_writter_CLASSIFICATION)
                log_writter_CLASSIFICATION.flush()
                exit(0)

        elif args.classification_dataset == "ChestXray14":
            from datasets_medical import build_transform_classification, ChestXray14Dataset

            args.epochs = 200
            img_size = args.imgsize # 448 worked
            # args.lr = 0.1
            batch_size = args.batch_size # 128
            num_workers = 12 # 16 | 32
            # best_acc1_CLASSIFICATION = 100000
            best_val_CLASSIFICATION = 10000

            patience_counter_CLASSIFICATION = 0
            patience_CLASSIFICATION = 35

            print()
            print("-------------")
            print("[Information]  TASK:", "Classification")
            print("[Information]  Backbone:", args.backbonemodel)
            # print("[Information]  Backbone Weights:", args.backbone_dir)
            print("[Information]  Dataset:", args.classification_dataset)
            print("[Information]  Total Epoch:", args.epochs)
            print("[Information]  Learning Rate:", args.lr)
            print("[Information]  Image Size:", img_size)
            print("[Information]  Batch Size:", batch_size)
            print("[Information]  Num Workers:", num_workers)
            print("[Information]  Optimizer:", "SGD")
            print("[Information]  Output Dir:", args.output_dir)
            print("-------------")
            print() # file=log_writter_SEGMENTATION

            print("-------------", file=log_writter_CLASSIFICATION)
            print("[Information]  TASK:", "Classification", file=log_writter_CLASSIFICATION)
            print("[Information]  Backbone:", args.backbonemodel, file=log_writter_CLASSIFICATION)
            # print("[Information]  Backbone Weights:", args.backbone_dir, file=log_writter_CLASSIFICATION)
            print("[Information]  Dataset:", args.classification_dataset, file=log_writter_CLASSIFICATION)
            print("[Information]  Total Epoch:", args.epochs, file=log_writter_CLASSIFICATION)
            print("[Information]  Learning Rate:", args.lr, file=log_writter_CLASSIFICATION)
            print("[Information]  Image Size:", img_size, file=log_writter_CLASSIFICATION)
            print("[Information]  Batch Size:", batch_size, file=log_writter_CLASSIFICATION)
            print("[Information]  Num Workers:", num_workers, file=log_writter_CLASSIFICATION)
            print("[Information]  Optimizer:", "SGD", file=log_writter_CLASSIFICATION)
            print("[Information]  Output Dir:", args.output_dir, file=log_writter_CLASSIFICATION)
            print("-------------", file=log_writter_CLASSIFICATION)

            diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
            train_list = 'data/xray14/official/train_official.txt'
            val_list = 'data/xray14/official/val_official.txt'
            test_list = 'data/xray14/official/test_official.txt'
            data_dir = "/data/jliang12/jpang12/dataset/nih_xray14/images/images/"
            dataset_train = ChestXray14Dataset(images_path=data_dir, file_path=train_list,
                                               augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
            dataset_val = ChestXray14Dataset(images_path=data_dir, file_path=val_list,
                                             augment=build_transform_classification(normalize="chestx-ray", mode="valid"))
            dataset_test = ChestXray14Dataset(images_path=data_dir, file_path=test_list,
                                              augment=build_transform_classification(normalize="chestx-ray", mode="test"))

            train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers, pin_memory=True, drop_last=True)
            val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False,
                                         num_workers=num_workers, pin_memory=True)
            test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size//2, shuffle=False,
                                        num_workers=num_workers, pin_memory=True)

            print("[Information]  TrainLoader Length:", len(train_loader))
            print("[Information]  ValLoader Length:", len(val_loader))
            print("[Information]  TestLoader Length:", len(test_loader))

            criterion = torch.nn.BCEWithLogitsLoss()
            # optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=0)
            loss_scaler = NativeScaler()

            optimizer = create_optimizer(args, model)
            # optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args)) ## create_optimizer(args, model)
            lr_scheduler, _ = create_scheduler(args, optimizer)

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






    ## T R A I N I N G   P H A S E ##
    print("Start training")
    start_time = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    print("[Training Info.] Start_Epoch & Total_Epoch", args.start_epoch, args.epochs)

    model._set_static_graph() # added by Nahid because of adding Classification & Segmentation component -- Forward/Backward pass issue
    for epoch in range(args.start_epoch, args.epochs):
        epoch_start_time = time.time()

        ## D E T E C T I O N  T A S K ##
        if args.taskcomponent == 'detection':
            if args.distributed:   # Active for Detection
                sampler_train.set_epoch(epoch)
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer, device, epoch,
                args.clip_max_norm, wo_class_error=wo_class_error, lr_scheduler=lr_scheduler, args=args, logger=(logger if args.save_log else None), ema_m=ema_m)
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']

            if not args.onecyclelr:
                lr_scheduler.step()
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
            test_stats, coco_evaluator = evaluate(
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


        ## S E G M E N T A T I O N   T A S K  ##
        if args.taskcomponent == 'segmentation':
            start_time = time.time()

            lr_ = step_decay(epoch, args.lr, args.epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            print('learning_rate: {}, @Epoch {}'.format(optimizer.param_groups[0]['lr'], epoch), file=log_writter_SEGMENTATION)

            train_lossAVG = train_one_epoch_SEGMENTATION(model, train_loader, optimizer, loss_scaler, epoch, log_writter_SEGMENTATION)
            # print( 'Training loss: {}@Epoch: {}'.format(train_lossAVG, epoch), file=log_writter_SEGMENTATION )

            # print( "Epoch {:04d}: Train Loss {:.5f} ".format(epoch, train_lossAVG) )
            print( "Epoch {:04d}: Train Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_SEGMENTATION )
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            # print('Training time {}'.format(total_time_str))
            print('Training time {}\n'.format(total_time_str), file=log_writter_SEGMENTATION)

            start_time = time.time()
            val_avg_SEGMENTATION = evaluation_SEGMENTATION(model, val_loader, epoch, log_writter_SEGMENTATION)
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            # print('Validation time {}'.format(total_time_str))
            print('Validation time {}'.format(total_time_str), file=log_writter_SEGMENTATION)

            if val_avg_SEGMENTATION < best_val_loss_SEGMENTATION:
                save_file = os.path.join(model_path_SEGMENTATION, 'ckpt.pth')
                save_model(model, optimizer, log_writter_SEGMENTATION, epoch+1, save_file)

                print( "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}\n".format(epoch, best_val_loss_SEGMENTATION, val_avg_SEGMENTATION, save_file), file=log_writter_SEGMENTATION )
                best_val_loss_SEGMENTATION = val_avg_SEGMENTATION
                patience_counter_SEGMENTATION = 0
            else:
                print( "Epoch {:04d}: val_loss did not improve from {:.5f} \n".format(epoch, best_val_loss_SEGMENTATION), file=log_writter_SEGMENTATION )
                patience_counter_SEGMENTATION += 1

            if patience_counter_SEGMENTATION > patience_SEGMENTATION:
                print( "Early Stopping", file=log_writter_SEGMENTATION )
                break




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
            #     lr_ = step_decay_cosine(epoch, args.lr, args.epochs + 1,  warmup_epochs=10)
            #     for param_group in optimizer.param_groups:
            #         param_group['lr'] = lr_
            lr_scheduler.step(epoch+1, 0)
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


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

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
