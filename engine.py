# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""

import math
import os
import sys
from typing import Iterable

from util.utils import slprint, to_device

import torch
import copy
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator

import numpy as np
from torch import nn
from sklearn.manifold import TSNE
from collections import OrderedDict
import matplotlib.pyplot as plt
import time
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)


def l2_regularizer(weights):
  """
  Calculates the sum of squared L2 norms of all weights in the given list.

  Args:
    weights: A list of torch.Tensor representing the weights of the encoder.

  Returns:
    A torch.Tensor representing the sum of squared L2 norms of all weights.
  """
  regularization_term = 0
  # for w in weights:
  for n,w in weights.items():
    regularization_term += torch.norm(w).pow(2)
  return regularization_term


def ema_update_teacher(model, teacher, momentum_schedule, it, total_epochs_args):
    with torch.no_grad():
        # if it < 10:
        #     m = momentum_schedule[it]  # momentum parameter
        # else:
        #     m = momentum_schedule[9]
        m = 0.80
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

# def ema_update_teacher(model, teacher, momentum_schedule, it, total_epochs_args):
#     with torch.no_grad():
#         if it < total_epochs_args:
#             m = momentum_schedule[it]  # momentum parameter
#         else:
#             m = momentum_schedule[total_epochs_args-1]
            
#         for param_q, param_k in zip(model.parameters(), teacher.parameters()):
#             param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)



def train_ignore_labels_multiLabelTraining(outputs, label2ignore=None):
    if label2ignore == None:
        return outputs

    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    ### for JJJ in range(0, len(igmoreLabelList)):
    for III in range(0, len(out_logits)): ## For each samples in the batch
        ignore_index_label = label2ignore ## Ignore predictions for Heart
        probabilities = out_logits[III].sigmoid()
        
        ignore_index_list_labels = (torch.argmax(probabilities, dim=-1) == ignore_index_label).nonzero()
        raw_ignore_index_list_labels_values = ignore_index_list_labels[:,0]
        available_indices = [index for index in range(0, out_logits.shape[1]) if index not in raw_ignore_index_list_labels_values]

        if len(available_indices) == 0:
            continue
        
        filtered_out_logits = out_logits[III].clone()
        filtered_out_logits[raw_ignore_index_list_labels_values] = out_logits[III, available_indices[0]] ## Using the first available index value
        filtered_out_bbox = out_bbox[III].clone()
        filtered_out_bbox[raw_ignore_index_list_labels_values] = out_bbox[III, available_indices[0]]

        out_logits[III] = filtered_out_logits
        out_bbox[III] = filtered_out_bbox

    outputs['pred_logits']  = out_logits
    outputs['pred_boxes'] = out_bbox


    # for JJJ in range(0, len(outputs['aux_outputs'])):
    #     out_logits, out_bbox = outputs['aux_outputs'][JJJ]['pred_logits'], outputs['aux_outputs'][JJJ]['pred_boxes']
    #     print("AUX out_logits", JJJ, out_logits.shape)
    #     for III in range(0, len(out_logits)): ## For each samples in the batch
    #         ignore_index_label = label2ignore ## Ignore predictions for Heart
    #         print("AUX out_logits", JJJ, III, out_logits[III].shape)

    #         probabilities = out_logits[III].sigmoid()
            
    #         ignore_index_list_labels = (torch.argmax(probabilities, dim=-1) == ignore_index_label).nonzero()
    #         raw_ignore_index_list_labels_values = ignore_index_list_labels[:,0]
    #         available_indices = [index for index in range(0, out_logits.shape[1]) if index not in raw_ignore_index_list_labels_values]
            
    #         filtered_out_logits = out_logits[III].clone()
    #         filtered_out_logits[raw_ignore_index_list_labels_values] = out_logits[III, available_indices[0]] ## Using the first available index value

    #         filtered_out_bbox = out_bbox[III].clone()
    #         filtered_out_bbox[raw_ignore_index_list_labels_values] = out_bbox[III, available_indices[0]]

    #         out_logits[III] = filtered_out_logits
    #         out_bbox[III] = filtered_out_bbox

    #     outputs['aux_outputs'][JJJ]['pred_logits'], outputs['aux_outputs'][JJJ]['pred_boxes'] = out_logits, out_bbox

    return outputs

# def train_ignore_labels_multiLabelTraining(outputs, label2ignore=None):
#     if label2ignore is None:
#         return outputs

#     def process_outputs(out_logits, out_bbox):
#         for III in range(len(out_logits)):
#             ignore_index_label = label2ignore
#             probabilities = out_logits[III].sigmoid()

#             ignore_index_list_labels = (torch.argmax(probabilities, dim=-1) == ignore_index_label).nonzero()
#             raw_ignore_index_list_labels_values = ignore_index_list_labels[:, 0]
#             available_indices = [index for index in range(out_logits.shape[1]) if index not in raw_ignore_index_list_labels_values]

#             filtered_out_logits = out_logits[III].clone()
#             filtered_out_logits[raw_ignore_index_list_labels_values] = out_logits[III, available_indices[0]]

#             filtered_out_bbox = out_bbox[III].clone()
#             filtered_out_bbox[raw_ignore_index_list_labels_values] = out_bbox[III, available_indices[0]]

#             out_logits[III] = filtered_out_logits
#             out_bbox[III] = filtered_out_bbox

#         return out_logits, out_bbox

#     outputs['pred_logits'], outputs['pred_boxes'] = process_outputs(outputs['pred_logits'], outputs['pred_boxes'])

#     for JJJ in range(len(outputs['aux_outputs'])):
#         out_logits, out_bbox = outputs['aux_outputs'][JJJ]['pred_logits'], outputs['aux_outputs'][JJJ]['pred_boxes']
#         outputs['aux_outputs'][JJJ]['pred_logits'], outputs['aux_outputs'][JJJ]['pred_boxes'] = process_outputs(out_logits, out_bbox)

#     return outputs



def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    DetHead=None, train_type='uF', wo_class_error=False, lr_scheduler=None, args=None, logger=None, model_ema=None, momen=None, coff=None, criterionMSE=None):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    try:
        need_tgt_for_training = args.use_dn
    except:
        need_tgt_for_training = False

    model.train()
    # model_ema.eval()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if not wo_class_error:
        metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 500

    _cnt = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header, logger=logger):
        # if _cnt == 501: # for cyclic individual localization task (Left Lung, Right Lung, Heart)
        #     break

        # print()
        # print("[CHECK Freeze unFreeze] train_type", train_type, "length dataloader", len(data_loader), "half", len(data_loader)//2)
        # print()
        if train_type == 'F': # 'F' -- means the backbone and loc.encoder is frozen --> train randomly on half of the trainingSet
            if _cnt == len(data_loader)//2:
                break

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # print("CHECK ImageSize:", samples.shape)
        # exit()
        with torch.cuda.amp.autocast(enabled=args.amp):
            if need_tgt_for_training:
                outputs, features_cons, features_Encons = model(samples, targets)
                # print("[Check - Train1Epoch] outputs", outputs.shape)
            else:
                outputs, features_cons, features_Encons = model(samples)
            
            # if DetHead != None:
            #     target_gt_dictList = []
            #     for jjj in range(0, len(targets)):
            #         # print("[CHECK] size", targets[jjj]['size'])
            #         target_gt_dict = {
            #             'boxes': torch.tensor( [targets[jjj]['boxes'][DetHead].tolist()] ).cuda(),
            #             'labels': torch.tensor( [targets[jjj]['labels'][DetHead].tolist()] ).cuda(),
            #             'image_id': torch.tensor( [targets[jjj]['image_id'][0].item()] ).cuda(), # image_id always contain one element
            #             'area': torch.tensor( [targets[jjj]['area'][DetHead].item()] ).cuda(),
            #             'iscrowd': torch.tensor( [targets[jjj]['iscrowd'][DetHead].item()] ).cuda(),
            #             'orig_size': torch.tensor( [targets[jjj]['orig_size'].tolist()] ).cuda(), # Always 2 items - img's [H, W]
            #             'size': torch.tensor( [targets[jjj]['size'].tolist()] ).cuda(), # Always 2 items - img's [H, W]
            #         }
            #         target_gt_dictList.append(target_gt_dict)
            #     targets = target_gt_dictList
            #     del target_gt_dict, target_gt_dictList
            
            
            # igmoreLabelList = []
            # if DetHead == 1: # Keep H and Ignore LL RL
            #     igmoreLabelList = [2,3]
            #     # print(" --- CHECK --- Keep H and Ignore LL RL")
            # elif DetHead == 2: # Keep LL and Ignore H RL
            #     igmoreLabelList = [1,3]
            #     # print(" --- CHECK --- Keep LL and Ignore H RL")
            # elif DetHead == 3: # Keep RL and Ignore H LL
            #     igmoreLabelList = [1,2]
            #     # print(" --- CHECK --- Keep RL and Ignore H LL")
            # for JJJ in range(0, len(igmoreLabelList)):
            #     outputs = train_ignore_labels_multiLabelTraining(outputs, label2ignore=igmoreLabelList[JJJ])


            # print(outputs['pred_logits'].shape, outputs['pred_boxes'].shape)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict

            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        if model_ema is not None:
            model_org_temp = copy.deepcopy(model)
            model_org_temp.eval()
            # with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.amp):
                # print("[Check] Engine Train - Student TaskDetHead", model_org_temp.task_DetHead)
                # print("[Check] Engine Train - Student TaskDetHead", model_ema.task_DetHead)
                if need_tgt_for_training:
                    _, features_cons, features_Encons = model_org_temp(samples, targets)
                    _, features_cons_ema, features_Encons_ema = model_ema(samples, targets)
                else:
                    _, features_cons, features_Encons = model_org_temp(samples)
                    _, features_cons_ema, features_Encons_ema = model_ema(samples)
            del model_org_temp
            loss_cons = criterionMSE(features_cons, features_cons_ema)
            loss_cons_2 = criterionMSE(features_Encons, features_Encons_ema)
            losses = (1-coff)*losses + coff*( (loss_cons+loss_cons_2)/2 )
            

        ### Collecting Localization Encoder Weights for Regularizer
        localization_encoder_weights = OrderedDict()
        for name, param in model.named_parameters():
          if "transformer.encoder" in name:
            localization_encoder_weights[name] = param
        ### l2_regularizer with weight_decay implemented
        losses = losses + ( 0.0001 * l2_regularizer(localization_encoder_weights) )
        del localization_encoder_weights


        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)


        # amp backward function
        if args.amp:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(losses).backward()
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            # original backward function
            optimizer.zero_grad(set_to_none=True)
            losses.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        if args.onecyclelr:
            lr_scheduler.step()
        if args.use_ema:
            if epoch >= args.ema_epoch:
                model_ema.update(model)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        if 'class_error' in loss_dict_reduced:
            metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        _cnt += 1
        if args.debug:
            if _cnt % 10 == 0:
                print("Detection DEBUG BREAK!"*5)
                break

    if getattr(criterion, 'loss_weight_decay', False):
        criterion.loss_weight_decay(epoch=epoch)
    if getattr(criterion, 'tuning_matching', False):
        criterion.tuning_matching(epoch)

    if args.modelEMA == "True_Epoch":
        ema_update_teacher(model, model_ema, momen, epoch, args.total_epochs)
        ### model_ema = model # should ignore later


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
    if getattr(criterion, 'loss_weight_decay', False):
        resstat.update({f'weight_{k}': v for k,v in criterion.weight_dict.items()})
    return resstat, model_ema


@torch.no_grad()
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


        # if DetHead != None:
        #     target_gt_dictList = []
        #     for jjj in range(0, len(targets)):
        #         # print("[CHECK] size", targets[jjj]['size'])
        #         target_gt_dict = {
        #             'boxes': torch.tensor( [targets[jjj]['boxes'][DetHead].tolist()] ).cuda(),
        #             'labels': torch.tensor( [targets[jjj]['labels'][DetHead].tolist()] ).cuda(),
        #             'image_id': torch.tensor( [targets[jjj]['image_id'][0].item()] ).cuda(), # image_id always contain one element
        #             'area': torch.tensor( [targets[jjj]['area'][DetHead].item()] ).cuda(),
        #             'iscrowd': torch.tensor( [targets[jjj]['iscrowd'][DetHead].item()] ).cuda(),
        #             'orig_size': torch.tensor( [targets[jjj]['orig_size'].tolist()] ).cuda(), # Always 2 items - img's [H, W]
        #             'size': torch.tensor( [targets[jjj]['size'].tolist()] ).cuda(), # Always 2 items - img's [H, W]
        #         }
        #         target_gt_dictList.append(target_gt_dict)
        #     targets = target_gt_dictList
        #     del target_gt_dict, target_gt_dictList


        with torch.cuda.amp.autocast(enabled=args.amp):
            # print("[Check] Engine Eval - TaskDetHead", model.task_DetHead)
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
            # res_score = outputs['res_score']
            # res_label = outputs['res_label']
            # res_bbox = outputs['res_bbox']
            # res_idx = outputs['res_idx']


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
                
                # img_h, img_w = tgt['orig_size'].unbind()
                # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
                # _res_bbox = res['boxes'] / scale_fct
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
        
        # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
        # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
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


@torch.no_grad()
def test(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    final_res = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id), 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        

    return final_res






@torch.no_grad()
def test_NAD(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, wo_class_error=False, args=None, logger=None):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    # if not wo_class_error:
    #     metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    panoptic_evaluator = None
    if 'panoptic' in postprocessors.keys():
        panoptic_evaluator = PanopticEvaluator(
            data_loader.dataset.ann_file,
            data_loader.dataset.ann_folder,
            output_dir=os.path.join(output_dir, "panoptic_eval"),
        )

    output_state_dict = {}
    final_res = []
    all_gt = []
    for samples, targets in metric_logger.log_every(data_loader, 10, header, logger=logger):
        samples = samples.to(device)

        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets = [{k: to_device(v, device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        # loss_dict = criterion(outputs, targets)
        # weight_dict = criterion.weight_dict

        # # reduce losses over all GPUs for logging purposes
        # loss_dict_reduced = utils.reduce_dict(loss_dict)
        # loss_dict_reduced_scaled = {k: v * weight_dict[k]
        #                             for k, v in loss_dict_reduced.items() if k in weight_dict}
        # loss_dict_reduced_unscaled = {f'{k}_unscaled': v
        #                               for k, v in loss_dict_reduced.items()}
        # metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
        #                      **loss_dict_reduced_scaled,
        #                      **loss_dict_reduced_unscaled)
        # if 'class_error' in loss_dict_reduced:
        #     metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, not_to_xyxy=True)
        # [scores: [100], labels: [100], boxes: [100, 4]] x B
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}


        # import pickle
        # # Save dictionary to a pickle file
        # with open("data_"+target['image_id']+".pkl", "wb") as pickle_file:
        #     pickle.dump(res, pickle_file)
        # print( targets['boxes'] )
        # break
        
        for image_id, outputs in res.items():
            _scores = outputs['scores'].tolist()
            _labels = outputs['labels'].tolist()
            _boxes = outputs['boxes'].tolist()
            for s, l, b in zip(_scores, _labels, _boxes):
                assert isinstance(l, int)
                itemdict = {
                        "image_id": int(image_id), 
                        "category_id": l, 
                        "bbox": b, 
                        "score": s,
                        }
                final_res.append(itemdict)


        # for image_id, outputs in res.items():
        #     _labels = targets['labels'].tolist()
        #     _boxes = targets['boxes'].tolist()
        #     for l, b in zip(_labels, _boxes):
        #         assert isinstance(l, int)
        #         itemdict = {
        #                 "image_id": int(image_id), 
        #                 "category_id": l, 
        #                 "bbox": b, 
        #                 }
        #         all_gt.append(itemdict)



        # if args.save_results:
        #     # res_score = outputs['res_score']
        #     # res_label = outputs['res_label']
        #     # res_bbox = outputs['res_bbox']
        #     # res_idx = outputs['res_idx']
        #     for i, (tgt, res, outbbox) in enumerate(zip(targets, results, outputs['boxes'])):
        #         """
        #         pred vars:
        #             K: number of bbox pred
        #             score: Tensor(K),
        #             label: list(len: K),
        #             bbox: Tensor(K, 4)
        #             idx: list(len: K)
        #         tgt: dict.

        #         """
        #         # compare gt and res (after postprocess)
        #         gt_bbox = tgt['boxes']
        #         gt_label = tgt['labels']
        #         # print("[CHECK GT] gt_bbox gt_label", gt_bbox, gt_label)
        #         # gt_info = torch.cat((gt_bbox, gt_label.unsqueeze(-1)), 1)
        #         gt_info = gt_bbox
                
        #         # img_h, img_w = tgt['orig_size'].unbind()
        #         # scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=0)
        #         # _res_bbox = res['boxes'] / scale_fct
        #         _res_bbox = outbbox[0]
        #         _res_prob = res['scores'][0]
        #         _res_label = res['labels'][0]
        #         # res_info = torch.cat((_res_bbox, _res_prob.unsqueeze(-1), _res_label.unsqueeze(-1)), 1)
        #         # print("[CHECK Res] _res_bbox _res_label", _res_bbox, _res_label)
        #         res_info = _res_bbox

        #         # print( "[CHECK] GT PRE", gt_info.shape, res_info.shape)

        #         # import ipdb;ipdb.set_trace()

        #         if 'gt_info' not in output_state_dict:
        #             output_state_dict['gt_info'] = []
        #         output_state_dict['gt_info'].append(gt_info.cpu())

        #         if 'res_info' not in output_state_dict:
        #             output_state_dict['res_info'] = []
        #         output_state_dict['res_info'].append(res_info.cpu())


    # if args.save_results:
    #     import os.path as osp
        
    #     # output_state_dict['gt_info'] = torch.cat(output_state_dict['gt_info'])
    #     # output_state_dict['res_info'] = torch.cat(output_state_dict['res_info'])
    #     savepath = osp.join(args.output_dir, 'results-{}.pkl'.format(utils.get_rank()))
    #     print("Saving res to {}".format(savepath))
    #     torch.save(output_state_dict, savepath)


    # if args.output_dir:
    #     import json
    #     with open(args.output_dir + f'/resultsGT{args.rank}.json', 'w') as f:
    #         json.dump(all_gt, f)  

    if args.output_dir:
        import json
        with open(args.output_dir + f'/results{args.rank}.json', 'w') as f:
            json.dump(final_res, f)        

    return final_res
