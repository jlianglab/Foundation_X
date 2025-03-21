import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device

import torch
import copy
import time
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from torch import nn
from sklearn.manifold import TSNE
from collections import OrderedDict
import matplotlib.pyplot as plt
from utils_segmentation import load_popar_weight, AverageMeter, save_model, dice_score, mean_dice_coef, torch_dice_coef_loss, exp_lr_scheduler_with_warmup, step_decay, load_swin_pretrained

def ema_update_teacher(model, teacher, momentum_schedule, it):
    with torch.no_grad():
        m = 0.80
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
    return teacher

def ema_update_teacher_Seg(model, teacher, momentum_schedule, it):
    with torch.no_grad():
        m = 0.50
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
    return teacher



def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, DetHead=None, wo_class_error=False, args=None, logger=None):
    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    features_detectionList = []
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    useCats = True
    try:
        useCats = args.useCats
    except:
        useCats = True
    if not useCats:
        print("useCats: {} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!".format(useCats))
    coco_evaluator = CocoEvaluator(base_ds, iou_types, useCats=useCats, args=args)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    _cnt = 0
    output_state_dict = {} # for debug only
    for samples, targets in metric_logger.log_every(data_loader, 1500, header, logger=logger):
        # if _cnt == 10:
        #     break

        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs, _, _ = model(samples, targets)
            else:
                outputs, _, _ = model(samples)
            # outputs = model(samples)

            if args.tsneOut:
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    _, features, _ = model.module.backbone[0].forward_raw(samples)
                else:
                    _, features, _ = model.backbone[0].forward_raw(samples)

                outs_detached = features[-1].cpu().detach().numpy()
                outs_detached = outs_detached.reshape(outs_detached.shape[1], outs_detached.shape[2]*outs_detached.shape[3])
                # embeddings_2d_detection = tsne.fit_transform(outs_detached)
                features_detectionList.append(outs_detached)

            loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        try:
            results = postprocessors['bbox'](outputs, orig_target_sizes) ## Regular without DetHead
        except:
            results = postprocessors['bbox'](outputs, orig_target_sizes[0]) ## with DetHead
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # print("[CHECK] Eval RES", len(targets), len(outputs['pred_boxes']) )

        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)
        
        if args.save_results:
            for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['pred_boxes'])):
                """
                pred vars:
                    K: number of bbox pred
                    score: Tensor(K),
                    label: list(len: K),
                    bbox: Tensor(K, 4)
                    idx: list(len: K)
                tgt: dict.

                """
                # compare gt and res (after postprocess)
                gt_bbox = tgt['boxes']
                gt_label = tgt['labels']
                gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
                
                _res_bbox = outbbox
                _res_prob = res['scores']
                _res_label = res['labels']
                res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
                # import ipdb;ipdb.set_trace()

                if 'gt_info' not in output_state_dict:
                    output_state_dict['gt_info'] = []
                output_state_dict['gt_info'].append(gt_info.cpu())

                if 'res_info' not in output_state_dict:
                    output_state_dict['res_info'] = []
                output_state_dict['res_info'].append(res_info.cpu())

            # # for debug only
            # import random
            # if random.random() > 0.7:
            #     print("Now let's break")
            #     break

        _cnt += 1
        if args.debug:
            if _cnt % 10 == 0:
                print("BREAK!"*5)
                break

    if args.save_results:
        import os.path as osp
        savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
        print("Saving res to {}".format(savepath))
        torch.save(output_state_dict, savepath)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]

    return stats, coco_evaluator, features_detectionList



def train_one_epoch_SEGMENTATION(args, model, train_loader, optimizer, loss_scaler, epoch, head_number=None, log_writter_SEGMENTATION=None, model_ema=None, momen=None, coff=None, criterionMSE=None):
    model.train(True)
    # model_ema.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesDICE = AverageMeter()
    lossesCONS = AverageMeter()
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

        img = img.float()
        mask = mask.float()

        # with torch.cuda.amp.autocast():
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            out_SegmentationHead, features, features_cons = model.module.backbone[0].extra_features_seg(img, head_n=head_number)
        else:
            out_SegmentationHead, features, features_cons = model.backbone[0].extra_features_seg(img, head_n=head_number)

        outputs = torch.sigmoid( out_SegmentationHead )

        del out_SegmentationHead, features

        loss = criterion(mask, outputs)
        loss_trainDice_temp = loss

        if model_ema is not None:
            # with torch.cuda.amp.autocast():
            # with torch.no_grad():
            if isinstance(model_ema, torch.nn.parallel.DistributedDataParallel):
                _, _, features_cons_ema = model_ema.module.backbone[0].extra_features_seg(img, head_n=head_number)
            else:
                _, _, features_cons_ema = model_ema.backbone[0].extra_features_seg(img, head_n=head_number)
            loss_cons = criterionMSE(features_cons, features_cons_ema)
            loss = (1-coff)*loss + coff*loss_cons
        else:
            loss_cons = 0


        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), file=log_writter_SEGMENTATION)
            sys.exit(1)
            # update metric
        losses.update(loss.item(), bsz)
        lossesDICE.update(loss_trainDice_temp.item(), bsz)
        if loss_cons != 0:
            lossesCONS.update(loss_cons.item(), bsz)

        optimizer.zero_grad(set_to_none=True)
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
        model_ema = ema_update_teacher_Seg(model, model_ema, momen, epoch)

    return losses.avg, lossesDICE.avg, lossesCONS.avg, model_ema


def evaluation_SEGMENTATION(args, model, val_loader, epoch, head_number=None, log_writter_SEGMENTATION=None):
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
            img = img.cuda(non_blocking=True) 
            mask = mask.cuda(non_blocking=True) 

            img = img.float()
            mask = mask.float()

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

def test_SEGMENTATION(args, model, test_loader, head_number=None, log_writter_SEGMENTATION=None):
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
                img = img.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

                img = img.float()
                mask = mask.float()

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
        
def MultiLabel_accuracy(preds, target, num_labels=3, multidim_average='samplewise'):
    """ Compute Accuracy for multilabel tasks by average per image"""
    acc = multilabel_accuracy(preds, target, num_labels=num_labels)
    return acc
    
def accuracy_CLASSIFICATION(output, target,topk=(1,)): # used for ImageNet
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


def train_CLASSIFICATION(args, train_loader, model, criterion, optimizer, epoch, log_writter_CLASSIFICATION, head_number=None, model_ema=None, momen=None, coff=None, criterionMSE=None, task_cls_type=None):
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
    device = torch.device(args.device)

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        if args.debug and i == 100: # Debug
            break

        # measure data loading time
        data_time.update(time.time() - end)

        images, target = images.float().to(device), target.float().to(device) # NIH14 # int for CheXpert

        target = target.unsqueeze(1) ## For Binary Classification
        if args.taskcomponent == 'detection_vindrcxr_disease' or task_cls_type == 'nonBinary': ## For Multi-label Classification
          target = target.squeeze(1)


        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            out_classifierHead, features_cons = model.module.backbone[0].extra_features(images, head_number)
        else:
            out_classifierHead, features_cons = model.backbone[0].extra_features(images, head_number)
        output = out_classifierHead
        loss = criterion(output, target)


        if model_ema is not None:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                _, features_cons_ema = model_ema.module.backbone[0].extra_features(images, head_number)
            else:
                _, features_cons_ema = model_ema.backbone[0].extra_features(images, head_number)
            loss_cons = criterionMSE(features_cons, features_cons_ema)
            loss = (1-coff)*loss + coff*loss_cons


        # compute gradient and do SGD step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if args.classification_dataset == 'imagenet':
            # measure accuracy and record loss
            if args.taskcomponent == 'detection_vindrcxr_disease':
              acc1 = MultiLabel_accuracy(output, target, num_labels=args.numClasses,multidim_average=None)
            else:
              acc1, acc5 = accuracy_CLASSIFICATION(output, target, topk=(1, 5))
              top1.update(acc1[0], images.size(0))
              top5.update(acc5[0], images.size(0))
              
            losses.update(loss.item(), images.size(0))
            
        else:
            losses.update(loss.item(), images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.classification_dataset == 'imagenet':
            if args.taskcomponent == 'detection_vindrcxr_disease':
              if i % 100 == 0:
                  progress.display(i)
                  print( "Epoch:", epoch, "|", "["+str(i)+"/"+str(len(train_loader))+"]", "Loss {:.4e}".format(loss.item()), "Acc {:.4f}".format(acc1), file=log_writter_CLASSIFICATION)
                  log_writter_CLASSIFICATION.flush()
            else:
              if i % 100 == 0:
                  progress.display(i)
                  print( "Epoch:", epoch, "|", "["+str(i)+"/"+str(len(train_loader))+"]", "Loss {:.4e}".format(loss.item()), "Acc@1 {:.4f}".format(acc1[0]), " Acc@5 {:.4f}".format(acc5[0]), file=log_writter_CLASSIFICATION)
                  log_writter_CLASSIFICATION.flush()
        else:
            if i % 500 == 0:
                print( "Epoch:", epoch, "|", "["+str(i)+"/"+str(len(train_loader))+"]", "Loss {:.5f} ({:.5f}) LR {:.6f}".format(loss.item(),losses.avg,optimizer.state_dict()['param_groups'][0]['lr']))
                print( "Epoch:", epoch, "|", "["+str(i)+"/"+str(len(train_loader))+"]", "Loss {:.5f} ({:.5f})".format(loss.item(),losses.avg), file=log_writter_CLASSIFICATION)
                log_writter_CLASSIFICATION.flush()
    
    if args.modelEMA == "True_Epoch": # Epoch based EMA update # ACTIVE
        model_ema = ema_update_teacher(model, model_ema, momen, epoch)

    return losses.avg, model_ema


def evaluate_CLASSIFICATION(args, val_loader, model, criterion, log_writter_CLASSIFICATION, head_number=None, task_cls_type=None):
    batch_time = AverageMeter_CLASSIFICATION('Time', ':6.3f')
    losses = AverageMeter_CLASSIFICATION('Loss', ':.4e')
    top1 = AverageMeter_CLASSIFICATION('Acc@1', ':6.2f')
    top5 = AverageMeter_CLASSIFICATION('Acc@5', ':6.2f')
    progress = ProgressMeter_CLASSIFICATION(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    #print("im inside the eval loop")
    y_test = torch.FloatTensor().cuda()
    p_test = torch.FloatTensor().cuda()
    # switch to evaluate mode
    model.eval()
    device = torch.device(args.device)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.debug and i == 500:
                break

            images, target = images.float().to(device), target.float().to(device) # NIH14 # int for CheXpert
            target = target.unsqueeze(1)
            if args.taskcomponent == 'detection_vindrcxr_disease' or task_cls_type == 'nonBinary':
              target = target.squeeze(1)
              
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                # _, out_classifierHead, _ = model.module.backbone(images)
                out_classifierHead, _ = model.module.backbone[0].extra_features(images, head_number)
            else:
                # _, out_classifierHead, _ = model.backbone(images)
                out_classifierHead, _ = model.backbone[0].extra_features(images, head_number)
            output = out_classifierHead

            # output = torch.sigmoid( output )

            loss = criterion(output, target)

            if args.classification_dataset == 'imagenet':
                # measure accuracy and record loss
                if args.taskcomponent == 'detection_vindrcxr_disease':
                  acc1 = MultiLabel_accuracy(output, target, num_labels=14,multidim_average=None)
                else:
                  acc1, acc5 = accuracy_CLASSIFICATION(output, target, topk=(1, 5))
                  top1.update(acc1[0], images.size(0))
                  top5.update(acc5[0], images.size(0))
                  
                losses.update(loss.item(), images.size(0))
                
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
                if i % 500 == 0:
                    #print("I'm in")
                    print( " - Test:", "["+str(i)+"/"+str(len(val_loader))+"]", "Loss {:.5f}".format(losses.avg))
                    print( " - Test:", "["+str(i)+"/"+str(len(val_loader))+"]", "Loss {:.5f}".format(losses.avg), file=log_writter_CLASSIFICATION)
                    log_writter_CLASSIFICATION.flush()

        if args.classification_dataset == 'imagenet':
            # TODO: this should also be done with the ProgressMeter
            if args.taskcomponent == 'detection_vindrcxr_disease':
              #print("I'm in4")
              individual_results = metric_AUROC(y_test, p_test, args.numClasses) # for NIH dataset
              individual_results = np.array(individual_results).mean() # for NIH dataset
              print("Validation/Test AUC =", individual_results)
              print("Validation/Test AUC =", individual_results, file=log_writter_CLASSIFICATION)
              return losses.avg, individual_results
            else:
              #print("I'm in3") 
              print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
              print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5), file=log_writter_CLASSIFICATION)
              log_writter_CLASSIFICATION.flush()
              return top1.avg
        else:
            #print("I'm in4")
            individual_results = metric_AUROC(y_test, p_test, args.numClasses) # for NIH dataset
            individual_results = np.array(individual_results).mean() # for NIH dataset
            print("Validation/Test AUC =", individual_results)
            print("Validation/Test AUC =", individual_results, file=log_writter_CLASSIFICATION)
            return losses.avg, individual_results

def test_CLASSIFICATION(args, data_loader_test, model):

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