import os 
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
# from engine import evaluate, train_one_epoch, test, test_NAD
from sklearn.metrics import roc_auc_score, accuracy_score
from torchmetrics.functional.classification import multilabel_accuracy
# Segmentation from Jiaxuan
from utils_segmentation import load_popar_weight, AverageMeter, save_model, dice_score, mean_dice_coef, torch_dice_coef_loss, exp_lr_scheduler_with_warmup, step_decay, load_swin_pretrained
from datasets_medical import build_transform_segmentation, dataloader_return, PXSDataset, MontgomeryDataset, JSRTClavicleDataset, JSRTHeartDataset,JSRTLungDataset, VinDrRibCXRDataset, ChestXDetDataset, JSRTLungDataset, VindrCXRHeartDataset
from timm.utils import NativeScaler, ModelEma
from models.load_weights_model import load_weights, load_weights_resume
import math

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer, create_optimizer_v2, optimizer_kwargs

import pandas as pd
import csv

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from engineClsSeg import evaluate, train_one_epoch_SEGMENTATION, evaluation_SEGMENTATION, test_SEGMENTATION, train_CLASSIFICATION, evaluate_CLASSIFICATION, test_CLASSIFICATION
# from engine import train_one_epoch, evaluate

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
    parser.add_argument('--resume', default=None, type=str, help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', type=str, help='load from other checkpoint') 
    parser.add_argument('--backbone_dir', default=None, type=str, help='loading backbone weights') #
    parser.add_argument('--init', default=None, type=str, help='imagenet22k | ark')
    parser.add_argument('--finetune_ignore', type=str, nargs='+')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--total_epochs', default=500, type=int)
    parser.add_argument('--cocoeval_pred_result', action='store_true') ## Shiv's
    parser.add_argument('--eval', action='store_true') 
    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true') 
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debugOnlyTest', action='store_true')
    parser.add_argument('--saveAllModel', action='store_true')
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
    parser.add_argument('--lr_locEnc', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_locDec', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, metavar='LR_Backbone',
                        help='learning rate (default: 1e-5)')
    parser.add_argument('--lr_segmentor', type=float, default=1e-3, metavar='LR_Segmentor',
                        help='learning rate (default: 1e-3)')

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
    
    parser.add_argument('--distributed', action='store_true', help='Enable distributed training') # 
    parser.add_argument('--eval_json_file_name', default=None, type=str, help='Eval Json File Name')

    parser.add_argument('--path_to_models', type=str, help='load from other checkpoint') 
    return parser



def save_image(input,idx):
    from PIL import Image

    def disparity_normalization(disp):  # disp is an array in uint8 data type
        _min = np.amin(disp)
        _max = np.amax(disp)
        disp_norm = (disp - _min) * 255.0 / (_max - _min)
        return np.uint8(disp_norm)

    im = disparity_normalization(input)
    im = Image.fromarray(im)
    im.save("{}.jpeg".format(idx))

def build_model_main(args):
    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS
    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model, criterion, postprocessors = build_func(args)
    return model, criterion, postprocessors



def evaluate_for_classification(checkpoint_path, model, model_ema, criterion_CLS, log_writter_DETECTION, datasetname__, head_number_temp, task_cls_type_temp, test_loader_cls_temp):
    logs_path = os.path.join(args.output_dir, "Logs")
    if os.path.exists(os.path.join(logs_path, "log_TestTime.txt")):
        log_writter_timer = open(os.path.join(logs_path, "log_TestTime.txt"), 'a')
    else:
        log_writter_timer = open(os.path.join(logs_path, "log_TestTime.txt"), 'w')
    
    start_time = time.time()

    epoch = args.start_epoch
    print("Testing Classification:", datasetname__)
    val_loss, auc_eval = evaluate_CLASSIFICATION(args, test_loader_cls_temp, model, criterion_CLS, log_writter_DETECTION, head_number=head_number_temp, task_cls_type=task_cls_type_temp)
    print( "Epoch {:04d}: {} Student Val Loss {:.5f} ".format(epoch, datasetname__, val_loss) )
    print( "Epoch {:04d}: {} Student Val Loss {:.5f} ".format(epoch, datasetname__, val_loss), file=log_writter_DETECTION )
    file1 = open(args.output_dir+'/export_csvFile_TEST.txt',"a")
    file1.write("Epoch {:04d}: {} Student Classification AUC {:.5f} \n".format(epoch, datasetname__, 100*auc_eval))
    file1.close()
    fields=[epoch, datasetname__, checkpoint_path, 'Student', 'Classification_'+datasetname__, str(100*auc_eval), '-', '-', '-', '-'] # AUC_SliceLevel_Res
    with open(args.output_dir+'/export_csvFile_TEST.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    if args.modelEMA is not None:
        val_loss, auc_eval = evaluate_CLASSIFICATION(args, test_loader_cls_temp, model_ema, criterion_CLS, log_writter_DETECTION, head_number=head_number_temp, task_cls_type=task_cls_type_temp)
        print( "Epoch {:04d}: {} Teacher Val Loss {:.5f} ".format(epoch, datasetname__, val_loss) )
        print( "Epoch {:04d}: {} Teacher Val Loss {:.5f} ".format(epoch, datasetname__, val_loss), file=log_writter_DETECTION )
        file1 = open(args.output_dir+'/export_csvFile_TEST.txt',"a")
        file1.write("Epoch {:04d}: {} Teacher Classification AUC {:.5f} \n\n".format(epoch, datasetname__, 100*auc_eval))
        file1.close()
        fields=[epoch, datasetname__, checkpoint_path, 'Teacher', 'Classification_'+datasetname__, str(100*auc_eval), '-', '-', '-', '-'] # AUC_SliceLevel_Res
        with open(args.output_dir+'/export_csvFile_TEST.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
    end_time = time.time()
    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Test Time {}: {:0>2}:{:0>2}:{:05.2f}".format(datasetname__, int(hours),int(minutes),seconds)) # log_writter
    print("{} - Classification Test Time {}: {:0>2}:{:0>2}:{:05.2f}".format(args.resume, datasetname__, int(hours),int(minutes),seconds), file=log_writter_timer)

def evaluate_for_localization(checkpoint_path, model, model_ema, criterion, criterion_ema, postprocessors, postprocessors_ema, wo_class_error, datasetname__, model_task_DetHead, model_ema_task_DetHead, DetHead_temp, base_ds_temp, test_loader_loc_temp, device, logger):
    logs_path = os.path.join(args.output_dir, "Logs")
    if os.path.exists(os.path.join(logs_path, "log_TestTime.txt")):
        log_writter_timer = open(os.path.join(logs_path, "log_TestTime.txt"), 'a')
    else:
        log_writter_timer = open(os.path.join(logs_path, "log_TestTime.txt"), 'w')
    
    start_time = time.time()

    epoch = args.start_epoch
    print("Localization Testing Student: "+datasetname__)
    args.eval_json_file_name = args.eval_json_file_name + "_Student"
    model.task_DetHead = model_task_DetHead ## Localization Heart
    test_stats, coco_evaluator, features_detectionList = evaluate(
        model, criterion, postprocessors, test_loader_loc_temp, base_ds_temp, device, args.output_dir,
        DetHead=DetHead_temp, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None))
    log_stats = { **{f'test_{k}': v for k, v in test_stats.items()} }
    print("Localization Testing Student: "+datasetname__)
    result_output_dir = args.output_dir + '/resultsTEST.txt'
    log_writer_detection = open(result_output_dir, 'a')
    log_writer_detection.write('Epoch: ' + str(epoch) + " Testing: "+datasetname__ + '\n')
    log_writer_detection.write('-- Testing --' + '\n')
    formatted_stats_test = {f'test_{k}': v for k, v in test_stats.items()}
    for key, value in formatted_stats_test.items():
        log_writer_detection.write(f'{key}: {value}\n')
    log_writer_detection.write('\n')
    log_writer_detection.write('\n')
    log_writer_detection.close()
    file1 = open(args.output_dir+'/export_csvFile_TEST.txt',"a")
    file1.write("Epoch {:04d}: {} Student Localization mAP40 {:.5f} \n".format(epoch, datasetname__, 100*value[1]))
    file1.write("Epoch {:04d}: {} Student Localization mAP50 {:.5f} \n".format(epoch, datasetname__, 100*value[2]))
    file1.write("Epoch {:04d}: {} Student Localization mAP50-95 {:.5f} \n".format(epoch, datasetname__, 100*value[0]))
    file1.close()
    fields=[epoch, datasetname__, checkpoint_path, 'Student', 'Localization_'+datasetname__, '-', 100*value[1], 100*value[2], 100*value[0], '-'] # AUC_SliceLevel_Res
    with open(args.output_dir+'/export_csvFile_TEST.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    if args.modelEMA:
        print("Localization Testing Teacher: "+datasetname__)
        args.eval_json_file_name = args.eval_json_file_name + "_Teacher"
        model_ema.task_DetHead = model_ema_task_DetHead
        test_stats, coco_evaluator, features_detectionList = evaluate(
            model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, device, args.output_dir,
            DetHead=DetHead_temp, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None))
        log_stats = { **{f'test_{k}': v for k, v in test_stats.items()} }
        print("Localization Testing Teacher: "+datasetname__)
        result_output_dir = args.output_dir + '/resultsTEST.txt'
        log_writer_detection = open(result_output_dir, 'a')
        log_writer_detection.write('Epoch: ' + str(epoch) + " TestingEMA: "+datasetname__ + '\n')
        log_writer_detection.write('-- Testing Teacher --' + '\n')
        formatted_stats_test = {f'testEMA_{k}': v for k, v in test_stats.items()}
        for key, value in formatted_stats_test.items():
            log_writer_detection.write(f'{key}: {value}\n')
        log_writer_detection.write('\n')
        log_writer_detection.write('\n')
        log_writer_detection.close()
        file1 = open(args.output_dir+'/export_csvFile_TEST.txt',"a")
        file1.write("Epoch {:04d}: {} Teacher Localization mAP40 {:.5f} \n".format(epoch, datasetname__, 100*value[1]))
        file1.write("Epoch {:04d}: {} Teacher Localization mAP50 {:.5f} \n".format(epoch, datasetname__, 100*value[2]))
        file1.write("Epoch {:04d}: {} Teacher Localization mAP50-95 {:.5f} \n\n".format(epoch, datasetname__, 100*value[0]))
        file1.close()
        fields=[epoch, datasetname__, checkpoint_path, 'Teacher', 'Localization_'+datasetname__, '-', 100*value[1], 100*value[2], 100*value[0], '-'] # AUC_SliceLevel_Res
        with open(args.output_dir+'/export_csvFile_TEST.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    end_time = time.time()
    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Test Time {}: {:0>2}:{:0>2}:{:05.2f}".format(datasetname__, int(hours),int(minutes),seconds)) # log_writter
    print("{} - Localization Test Time {}: {:0>2}:{:0>2}:{:05.2f}".format(args.resume, datasetname__, int(hours),int(minutes),seconds), file=log_writter_timer)

def evaluate_for_segmentation(checkpoint_path, model, model_ema, datasetname__, head_number_temp, test_loader_loc_temp, log_writter_DETECTION):
    epoch = args.start_epoch
    test_y, test_p, _ = test_SEGMENTATION(args, model, test_loader_loc_temp, head_number=head_number_temp, log_writter_SEGMENTATION=log_writter_DETECTION)
    print("Epoch {}: Student {} Dice = {:.8f}%".format(epoch, datasetname__, 100.0 * dice_score(test_p, test_y)), file=log_writter_DETECTION)
    print("Epoch {}: Student {} Dice = {:.8f}%".format(epoch, datasetname__, 100.0 * dice_score(test_p, test_y)))
    file1 = open(args.output_dir+'/export_csvFile_TEST.txt',"a")
    file1.write("Epoch {:04d}: {} Teacher Segmentation AUC {:.5f} \n\n".format(epoch, datasetname__, 100.0 * dice_score(test_p, test_y)))
    file1.close()
    fields=[epoch, datasetname__, checkpoint_path, 'Student', 'Localization_'+datasetname__, '-', '-', '-', '-', 100.0 * dice_score(test_p, test_y)] # AUC_SliceLevel_Res
    with open(args.output_dir+'/export_csvFile_TEST.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    if args.modelEMA:
        test_y, test_p, _ = test_SEGMENTATION(args, model_ema, test_loader_loc_temp, head_number=head_number_temp, log_writter_SEGMENTATION=log_writter_DETECTION)
        print("Epoch {}: Teacher {} Dice = {:.8f}%".format(epoch, datasetname__, 100.0 * dice_score(test_p, test_y)), file=log_writter_DETECTION)
        print("Epoch {}: Teacher {} Dice = {:.8f}%".format(epoch, datasetname__, 100.0 * dice_score(test_p, test_y)))
        file1 = open(args.output_dir+'/export_csvFile_TEST.txt',"a")
        file1.write("Epoch {:04d}: {} Teacher Segmentation AUC {:.5f} \n\n".format(epoch, datasetname__, 100.0 * dice_score(test_p, test_y)))
        file1.close()
        fields=[epoch, datasetname__, checkpoint_path, 'Teacher', 'Localization_'+datasetname__, '-', '-', '-', '-', 100.0 * dice_score(test_p, test_y)] # AUC_SliceLevel_Res
        with open(args.output_dir+'/export_csvFile_TEST.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)



def main(args):
    cfg = SLConfig.fromfile(args.config_file)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k,v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))
        
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(output=os.path.join(args.output_dir, 'info.txt'), distributed_rank=args.rank, color=False, name="detr")
    device = torch.device(args.device)

    ### ----------------- BUILD MODEL ----------------- ###
    if args.taskcomponent == "detect_segmentation_cyclic" or args.taskcomponent == "detect_segmentation_cyclic_v2" or args.taskcomponent == "segmentation_vindrcxr_organ_1h3ch":
        args.num_classes = 3+1
        args.dn_labelbook_size = 5
    if args.taskcomponent in ["detect_segmentation_cyclic_v3", "detect_segmentation_cyclic_v4", "detect_vindrcxr_heart_segTest", "detect_vindrcxr_heart", "detect_vindrcxr_leftlung", "detect_vindrcxr_rightlung"]: # 
        # args.num_classes = 1+1
        # args.dn_labelbook_size = 3
        args.num_classes = 3+1
        args.dn_labelbook_size = 5
    if args.taskcomponent in ["segmentation"]: # 
        args.num_classes = 3+1
        args.dn_labelbook_size = 5
    if args.taskcomponent in ["detect_node21_nodule", "detect_tbx11k_catagnostic", "ClsLoc_tbx11k_catagnostic"]: # Binary Class
        args.num_classes = 2
        args.dn_labelbook_size = 3
    if args.taskcomponent in ["detection_vindrcxr_disease"]: 
        args.num_classes = 14
        args.dn_labelbook_size = 15
    if args.taskcomponent in ["detection_vindrmammo_disease"]: # Binary Class
        args.num_classes = 10
        args.dn_labelbook_size = 12
    if args.taskcomponent in ["detect_chestxdet_dataset"]: 
        args.num_classes = 14
        args.dn_labelbook_size = 15
    if args.taskcomponent in ["detection"]: # Binary Class
        args.num_classes = 14
        args.dn_labelbook_size = 15
    if args.taskcomponent in ["foundation_x_pretraining"]: # Binary Class
        args.num_classes = 14
        args.dn_labelbook_size = 15

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

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)



     ### ----------------- Optimizer ----------------- ###
    if args.taskcomponent in ["foundation_x_pretraining"]:
        loss_scaler = NativeScaler()
        param_dicts = get_param_dict(args, model_without_ddp)
        if args.opt == "adamw":
            optimizer = torch.optim.AdamW([
                {'params': [param for name, param in model.named_parameters() if 'backbone' in name and 'segmentation_' not in name], 'lr': args.lr_backbone},##Backbone
                {'params': [param for name, param in model.named_parameters() if 'segmentation_' in name], 'lr': args.lr_segmentor},##Segmentor
                {'params': [param for name, param in model.named_parameters() if 'transformer.encoder' in name], 'lr': args.lr_locEnc},##Localizer Encoder
                {'params': [param for name, param in model.named_parameters() if 'transformer.decoder' in name], 'lr': args.lr_locDec},##Localizer Decoder
            ], lr=args.lr_locDec, weight_decay=args.weight_decay)

            args.lr_drop = 15
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
            print("[Info.] Lr Scheduler:", "StepLR", "Drop:", args.lr_drop)

    


     ### ----------------- Data Loading ----------------- ###
    if args.taskcomponent in ['foundation_x_pretraining']: # detect_chestxdet_dataset
        logs_path = os.path.join(args.output_dir, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if os.path.exists(os.path.join(logs_path, "log.txt")):
            log_writter_DETECTION = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            log_writter_DETECTION = open(os.path.join(logs_path, "log.txt"), 'w')

        if not os.path.exists(os.path.join(args.output_dir, "cocoeval_json")):
            os.makedirs( os.path.join(args.output_dir, "cocoeval_json") )

        print()
        print("-------------")
        print("[Information]  TASK:", args.taskcomponent)
        print("[Information]  Backbone:", args.backbonemodel) 
        print("[Information]  Backbone_INIT:", args.init)
        print("[Information]  Backbone Weights:", args.backbone_dir)
        print("[Information]  Dataset:", args.dataset_file)
        print("[Information]  Total Epoch:", args.total_epochs)
        print("[Information]  Batch Size:", args.batch_size)
        print("[Information]  Learning Rate Backbone:", args.lr_backbone)
        print("[Information]  Learning Rate LocEnc:", args.lr_locEnc)
        print("[Information]  Learning Rate LocDec:", args.lr_locDec)
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
        print("[Information]  Learning Rate Backbone:", args.lr_backbone, file=log_writter_DETECTION)
        print("[Information]  Learning Rate LocEnc:", args.lr_locEnc, file=log_writter_DETECTION)
        print("[Information]  Learning Rate Loc:", args.lr_locDec, file=log_writter_DETECTION)
        print("[Information]  Learning Rate Backbone:", args.lr_backbone, file=log_writter_DETECTION)
        print("[Information]  Num Workers:", args.num_workers, file=log_writter_DETECTION)
        print("[Information]  Optimizer:", args.opt, file=log_writter_DETECTION)
        print("[Information]  Output Dir:", args.output_dir, file=log_writter_DETECTION)
        print("-------------", file=log_writter_DETECTION)

        
        if args.backbone_dir is not None:
            model = load_weights(model, args)
        if args.modelEMA is not None:
            model_ema = copy.deepcopy(model)

        try:
            data_loader_train, data_loader_val, sampler_train, dataset_val = dataloader_return(args)
            base_ds = get_coco_api_from_dataset(dataset_val)
        except:
            do_nothing = 1

        criterion_CLS = torch.nn.BCEWithLogitsLoss()

        if args.dataset_file == "foundation6_datasets":
            train_loader_cls_TBX11k, test_loader_cls_TBX11k, train_loader_cls_NODE21, test_loader_cls_NODE21, train_loader_cls_ChestXDet, test_loader_cls_ChestXDet, \
            train_loader_cls_RSNApneumonia, test_loader_cls_RSNApneumonia, train_loader_cls_SIIMACRptx, test_loader_cls_SIIMACRptx, train_loader_cls_CANDIDptx, test_loader_cls_CANDIDptx, \
            train_loader_loc_TBX11k, test_loader_loc_TBX11k, dataset_val_loc_TBX11k, sampler_train_TBX11k, train_loader_loc_TBX11k_B, \
            train_loader_loc_Node21, test_loader_loc_Node21, dataset_val_loc_Node21, sampler_train_Node21, train_loader_loc_Node21_B, \
            train_loader_loc_CANDIDptx, test_loader_loc_CANDIDptx, dataset_val_loc_CANDIDptx, sampler_train_CANDIDptx, train_loader_loc_CANDIDptx_B, \
            train_loader_loc_ChestXDet, test_loader_loc_ChestXDet, dataset_val_loc_ChestXDet, sampler_train_ChestXDet, train_loader_loc_ChestXDet_B, \
            train_loader_loc_RSNApneumonia, test_loader_loc_RSNApneumonia, dataset_val_loc_RSNApneumonia, sampler_train_RSNApneumonia, train_loader_loc_RSNApneumonia_B, \
            train_loader_loc_SiimACR, test_loader_loc_SiimACR, dataset_val_loc_SiimACR, sampler_train_SiimACR, train_loader_loc_SiimACR_B, \
            train_loader_seg_ChestXDet, test_loader_seg_ChestXDet, train_loader_seg_SIIM, test_loader_seg_SIIM, train_loader_seg_CANDIDptx, test_loader_seg_CANDIDptx = dataloader_return(args)
            base_ds_TBX11k = get_coco_api_from_dataset(dataset_val_loc_TBX11k)
            base_ds_Node21 = get_coco_api_from_dataset(dataset_val_loc_Node21)
            base_ds_ChestXDet = get_coco_api_from_dataset(dataset_val_loc_ChestXDet)
            base_ds_CANDIDptx = get_coco_api_from_dataset(dataset_val_loc_CANDIDptx)
            base_ds_RSNApneumonia = get_coco_api_from_dataset(dataset_val_loc_RSNApneumonia)
            base_ds_SiimACR = get_coco_api_from_dataset(dataset_val_loc_SiimACR)
            del dataset_val_loc_TBX11k, dataset_val_loc_Node21, dataset_val_loc_ChestXDet
            for name, param in model.named_parameters():
                if ('decoder.1' in name) or ('decoder.2' in name) or ('segmentation' in name): ## Only Localization Encoder
                    param.requires_grad = False
            model_ema = copy.deepcopy(model)
            if args.eval == False:
                if args.resume == None:
                    if not os.path.exists(args.output_dir+'/export_csvFile_TRAIN.csv'):
                        export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_DiceLoss', 'Seg_ConsLoss'])
                        export_csvFile_train.to_csv(args.output_dir+'/export_csvFile_TRAIN.csv', index=False)

            if not os.path.exists(args.output_dir+'/export_csvFile_TEST.csv'):
                export_csvFile = pd.DataFrame(columns=['Epoch', 'Dataset', 'Checkpoint', 'Model', 'Task-Test', 'AUC', 'mAP40','mAP50','mAP50_95', 'DICE'])
                export_csvFile.to_csv(args.output_dir+'/export_csvFile_TEST.csv', index=False)





    ## ---------------- TESTING STARTS ---------------- ##
    print("Testing Started...")
    # PATH_TO_LOOK_AT = "/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX/run104_F5_Loc_1stRun/" ## TESTALL - part1
    # PATH_TO_LOOK_AT = "/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX/run104_F5_Loc_2ndRun/" ## TESTALL - part1
    # PATH_TO_LOOK_AT = "/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX/run104_F5_ClsLoc_1stRun/" ## TESTALL - part2
    PATH_TO_LOOK_AT = args.path_to_models

    # files = os.listdir(PATH_TO_LOOK_AT)
    # pth_files = [file for file in files if file.endswith('.pth')]
    # pth_files_sorted = sorted(pth_files)
    ### print(pth_files_sorted)

    # pth_files_sorted = []
    # ## TESTALL - part1
    # pth_files_sorted = [
    #     # 'ckpt_E254_TH1.pth', 'ckpt_E256_TH3.pth', 'ckpt_E258_TH5.pth', 'ckpt_E261_TH8.pth', 'ckpt_E263_TH10.pth'
    #     # 'ckpt_E506_TH1.pth', 'ckpt_E508_TH3.pth', 'ckpt_E510_TH5.pth', 'ckpt_E513_TH8.pth', 'ckpt_E515_TH10.pth'
    #     # 'ckpt_E752_TH1.pth', 'ckpt_E754_TH3.pth', 'ckpt_E756_TH5.pth', 'ckpt_E759_TH8.pth', 'ckpt_E761_TH10.pth'
    #     'ckpt_E1007_TH1.pth', 'ckpt_E1009_TH3.pth', 'ckpt_E1011_TH5.pth', 'ckpt_E1014_TH8.pth', 'ckpt_E1016_TH10.pth'

    # ]

    # with open(PATH_TO_LOOK_AT+'/sorted_filenames.txt', 'r') as f:
    #     pth_files_sorted = f.readlines()
    # pth_files_sorted = [path.strip() for path in pth_files_sorted]
    # pth_files_sorted = pth_files_sorted[148:] # 1st run
    # pth_files_sorted = pth_files_sorted[109:] # 2nd run
    # print(len(pth_files_sorted))
    
    # if 'TESTloc' in args.cyclictask and 'tbx11kLOC' in args.cyclictask:
    #     with open("/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX"+'/sorted_bestTrainLoss_F5_onlyLOC_TBX11k.txt', 'r') as f:
    #         pth_files_sorted = f.readlines()
    #     pth_files_sorted = [path.strip() for path in pth_files_sorted]
    #     pth_files_sorted = pth_files_sorted[43:]
    # if 'TESTloc' in args.cyclictask and 'node21LOC' in args.cyclictask:
    #     with open("/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX"+'/sorted_bestTrainLoss_F5_onlyLOC_NODE21.txt', 'r') as f:
    #         pth_files_sorted = f.readlines()
    #     pth_files_sorted = [path.strip() for path in pth_files_sorted]
    #     pth_files_sorted = pth_files_sorted[37:]
    # if 'TESTloc' in args.cyclictask and 'candidptxLOC' in args.cyclictask:
    #     with open("/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX"+'/sorted_bestTrainLoss_F5_onlyLOC_CANDIDptx.txt', 'r') as f:
    #         pth_files_sorted = f.readlines()
    #     pth_files_sorted = [path.strip() for path in pth_files_sorted]
    #     pth_files_sorted = pth_files_sorted[37:]
    # if 'TESTloc' in args.cyclictask and 'rsnapneumoniaLOC' in args.cyclictask:
    #     with open("/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX"+'/sorted_bestTrainLoss_F5_onlyLOC_RSNApneu.txt', 'r') as f:
    #         pth_files_sorted = f.readlines()
    #     pth_files_sorted = [path.strip() for path in pth_files_sorted]
    #     pth_files_sorted = pth_files_sorted[37:]
    # if 'TESTloc' in args.cyclictask and 'chestxdetLOC' in args.cyclictask:
    #     with open("/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX"+'/sorted_bestTrainLoss_F5_onlyLOC_ChestXDet.txt', 'r') as f:
    #         pth_files_sorted = f.readlines()
    #     pth_files_sorted = [path.strip() for path in pth_files_sorted]
    #     pth_files_sorted = pth_files_sorted[36:]


    # if os.path.exists(os.path.join(logs_path, "log_TestTime.txt")):
    #     log_writter = open(os.path.join(logs_path, "log_TestTime.txt"), 'a')
    # else:
    #     log_writter = open(os.path.join(logs_path, "log_TestTime.txt"), 'w')

    # for index_pth in range(0, len(pth_files_sorted)):
    #     if args.debug and index_pth > 0:
    #         break

    if args.debugOnlyTest: # args.resume is not None:
        # temp_extract_TH = pth_files_sorted[index_pth]
        # temp_extract_TH = temp_extract_TH.split(".")[0]
        # temp_extract_TH = temp_extract_TH.split("_")[-1]
        # if 'tbx11kLOConly' in args.cyclictask and temp_extract_TH != "TH1":
        #     continue
        # if 'node21LOConly' in args.cyclictask and temp_extract_TH != "TH3":
        #     continue
        print(" ---- Checkpoint Loading -----")
        # args.resume = PATH_TO_LOOK_AT + pth_files_sorted[index_pth]
        # args.resume = pth_files_sorted[index_pth]
        args.resume = PATH_TO_LOOK_AT
        print("[Model Checkpoint] Reading Model Checkpoint from:", args.resume)
        model, model_ema, optimizer, args.start_epoch = load_weights_resume(model, model_ema, optimizer, args)

    ### Test CLASSIFICATION ##
    if 'TESTcls' in args.cyclictask and 'tbx11kCLS' in args.cyclictask:
        datasetname__ = "TBX11k"
        head_number_temp = 0
        task_cls_type_temp = None
        test_loader_cls_temp = test_loader_cls_TBX11k
        args.numClasses = 1
        evaluate_for_classification(args.resume, model, model_ema, criterion_CLS, log_writter_DETECTION, datasetname__, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
    if 'TESTcls' in args.cyclictask and 'node21CLS' in args.cyclictask:
        datasetname__ = "Node21"
        head_number_temp = 1
        task_cls_type_temp = None
        test_loader_cls_temp = test_loader_cls_NODE21
        args.numClasses = 1
        evaluate_for_classification(args.resume, model, model_ema, criterion_CLS, log_writter_DETECTION, datasetname__, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
    if 'TESTcls' in args.cyclictask and 'candidptxCLS' in args.cyclictask:
        datasetname__ = "CANDID-PTX"
        head_number_temp = 2
        task_cls_type_temp = 'nonBinary'
        test_loader_cls_temp = test_loader_cls_CANDIDptx
        args.numClasses = 1
        evaluate_for_classification(args.resume, model, model_ema, criterion_CLS, log_writter_DETECTION, datasetname__, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
    if 'TESTcls' in args.cyclictask and 'rsnapneumoniaCLS' in args.cyclictask:
        datasetname__ = "RSNApneumonia"
        head_number_temp = 3
        task_cls_type_temp = 'nonBinary'
        test_loader_cls_temp = test_loader_cls_RSNApneumonia
        args.numClasses = 3
        evaluate_for_classification(args.resume, model, model_ema, criterion_CLS, log_writter_DETECTION, datasetname__, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
    if 'TESTcls' in args.cyclictask and 'chestxdetCLS' in args.cyclictask:
        datasetname__ = "ChestX-Det"
        head_number_temp = 4
        task_cls_type_temp = 'nonBinary'
        test_loader_cls_temp = test_loader_cls_ChestXDet
        args.numClasses = 13
        evaluate_for_classification(args.resume, model, model_ema, criterion_CLS, log_writter_DETECTION, datasetname__, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
    if 'TESTcls' in args.cyclictask and 'siimacrCLS' in args.cyclictask:
        datasetname__ = "SIIM-ACRpneumothorax"
        head_number_temp = 5
        task_cls_type_temp = 'nonBinary'
        test_loader_cls_temp = test_loader_cls_SIIMACRptx
        args.numClasses = 1
        evaluate_for_classification(args.resume, model, model_ema, criterion_CLS, log_writter_DETECTION, datasetname__, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
    # if 'TESTcls' in args.cyclictask:
    #     epoch = args.start_epoch
    #     val_loss, auc_eval = evaluate_CLASSIFICATION(args, test_loader_cls_temp, model, criterion_CLS, args, log_writter_DETECTION, head_number=head_number_temp, task_cls_type=task_cls_type_temp)
    #     print( "Epoch {:04d}: {} Student Val Loss {:.5f} ".format(epoch, datasetname__, val_loss) )
    #     print( "Epoch {:04d}: {} Student Val Loss {:.5f} ".format(epoch, datasetname__, val_loss), file=log_writter_DETECTION )
    #     file1 = open(args.output_dir+'/export_csvFile_TEST.txt',"a")
    #     file1.write("Epoch {:04d}: {} Student Classification AUC {:.5f} \n".format(epoch, datasetname__, 100*auc_eval))
    #     file1.close()
    #     fields=[epoch, datasetname__, pth_files_sorted[index_pth], 'Student', 'Classification_'+datasetname__, str(100.0 * dice_score(test_p, test_y)), '-', '-', '-', '-'] # AUC_SliceLevel_Res
    #     with open(args.output_dir+'/export_csvFile_TEST.csv', 'a') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(fields)
    #     if args.modelEMA is not None:
    #         val_loss, auc_eval = evaluate_CLASSIFICATION(args, test_loader_cls_temp, model_ema, criterion_CLS, args, log_writter_DETECTION, head_number=head_number_temp, task_cls_type=task_cls_type_temp)
    #         print( "Epoch {:04d}: {} Teacher Val Loss {:.5f} ".format(epoch, datasetname__, val_loss) )
    #         print( "Epoch {:04d}: {} Teacher Val Loss {:.5f} ".format(epoch, datasetname__, val_loss), file=log_writter_DETECTION )
    #         file1 = open(args.output_dir+'/export_csvFile_TEST.txt',"a")
    #         file1.write("Epoch {:04d}: {} Teacher Classification AUC {:.5f} \n\n".format(epoch, datasetname__, 100*auc_eval))
    #         file1.close()
    #         fields=[epoch, datasetname__, pth_files_sorted[index_pth], 'Teacher', 'Classification_'+datasetname__, str(100.0 * dice_score(test_p, test_y)), '-', '-', '-', '-'] # AUC_SliceLevel_Res
    #         with open(args.output_dir+'/export_csvFile_TEST.csv', 'a') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(fields)

    ### Test LOCALIZATION ##
    if 'TESTloc' in args.cyclictask and 'tbx11kLOC' in args.cyclictask:
        datasetname__ = "TBX11k"
        args.eval_json_file_name = PATH_TO_LOOK_AT.split("/")[-1] + "_" + datasetname__
        model.task_DetHead = 0
        model_ema.task_DetHead = 0
        DetHead_temp = 0
        base_ds_temp = base_ds_TBX11k
        test_loader_loc_temp = test_loader_loc_TBX11k
        evaluate_for_localization(args.resume, model, model_ema, criterion, criterion_ema, postprocessors, postprocessors_ema, wo_class_error, datasetname__, model.task_DetHead, model_ema.task_DetHead, DetHead_temp, base_ds_temp, test_loader_loc_temp, device, logger)
    if 'TESTloc' in args.cyclictask and 'node21LOC' in args.cyclictask:
        datasetname__ = "Node21"
        args.eval_json_file_name = PATH_TO_LOOK_AT.split("/")[-1] + "_" + datasetname__
        model.task_DetHead = 1
        model_ema.task_DetHead = 1
        DetHead_temp = 1
        base_ds_temp = base_ds_Node21
        test_loader_loc_temp = test_loader_loc_Node21
        evaluate_for_localization(args.resume, model, model_ema, criterion, criterion_ema, postprocessors, postprocessors_ema, wo_class_error, datasetname__, model.task_DetHead, model_ema.task_DetHead, DetHead_temp, base_ds_temp, test_loader_loc_temp, device, logger)
    if 'TESTloc' in args.cyclictask and 'candidptxLOC' in args.cyclictask:
        datasetname__ = "CANDID-PTX"
        args.eval_json_file_name = PATH_TO_LOOK_AT.split("/")[-1] + "_" + datasetname__
        model.task_DetHead = 2
        model_ema.task_DetHead = 2
        DetHead_temp = 2
        base_ds_temp = base_ds_CANDIDptx
        test_loader_loc_temp = test_loader_loc_CANDIDptx
        evaluate_for_localization(args.resume, model, model_ema, criterion, criterion_ema, postprocessors, postprocessors_ema, wo_class_error, datasetname__, model.task_DetHead, model_ema.task_DetHead, DetHead_temp, base_ds_temp, test_loader_loc_temp, device, logger)
    if 'TESTloc' in args.cyclictask and 'rsnapneumoniaLOC' in args.cyclictask:
        datasetname__ = "RSNApneumonia"
        args.eval_json_file_name = PATH_TO_LOOK_AT.split("/")[-1] + "_" + datasetname__
        model.task_DetHead = 3
        model_ema.task_DetHead = 3
        DetHead_temp = 3
        base_ds_temp = base_ds_RSNApneumonia
        test_loader_loc_temp = test_loader_loc_RSNApneumonia
        evaluate_for_localization(args.resume, model, model_ema, criterion, criterion_ema, postprocessors, postprocessors_ema, wo_class_error, datasetname__, model.task_DetHead, model_ema.task_DetHead, DetHead_temp, base_ds_temp, test_loader_loc_temp, device, logger)
    if 'TESTloc' in args.cyclictask and 'chestxdetLOC' in args.cyclictask:
        datasetname__ = "ChestX-Det"
        args.eval_json_file_name = PATH_TO_LOOK_AT.split("/")[-1] + "_" + datasetname__
        model.task_DetHead = 4
        model_ema.task_DetHead = 4
        DetHead_temp = 5
        base_ds_temp = base_ds_ChestXDet
        test_loader_loc_temp = test_loader_loc_ChestXDet
        evaluate_for_localization(args.resume, model, model_ema, criterion, criterion_ema, postprocessors, postprocessors_ema, wo_class_error, datasetname__, model.task_DetHead, model_ema.task_DetHead, DetHead_temp, base_ds_temp, test_loader_loc_temp, device, logger)
    if 'TESTloc' in args.cyclictask and 'siimacrLOC' in args.cyclictask:
        datasetname__ = "SIIM-ACRpneumothorax"
        args.eval_json_file_name = PATH_TO_LOOK_AT.split("/")[-1] + "_" + datasetname__
        model.task_DetHead = 5
        model_ema.task_DetHead = 5
        DetHead_temp = 5
        base_ds_temp = base_ds_SiimACR
        test_loader_loc_temp = test_loader_loc_SiimACR
        evaluate_for_localization(args.resume, model, model_ema, criterion, criterion_ema, postprocessors, postprocessors_ema, wo_class_error, datasetname__, model.task_DetHead, model_ema.task_DetHead, DetHead_temp, base_ds_temp, test_loader_loc_temp, device, logger)
    # if 'TESTloc' in args.cyclictask:
    #     epoch = args.start_epoch
    #     print("Localization Testing Student: "+datasetname__)
    #     # model.task_DetHead = 0 ## Localization Heart
    #     test_stats, coco_evaluator, features_detectionList = evaluate(
    #         model, criterion, postprocessors, test_loader_loc_temp, base_ds_temp, device, args.output_dir,
    #         DetHead=DetHead_temp, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None))
    #     log_stats = { **{f'test_{k}': v for k, v in test_stats.items()} }
    #     print("Localization Testing Student: "+datasetname__)
    #     result_output_dir = args.output_dir + '/resultsTEST.txt'
    #     log_writer_detection = open(result_output_dir, 'a')
    #     log_writer_detection.write('Epoch: ' + str(epoch) + " Testing: "+datasetname__ + '\n')
    #     log_writer_detection.write('-- Testing --' + '\n')
    #     formatted_stats_test = {f'test_{k}': v for k, v in test_stats.items()}
    #     for key, value in formatted_stats_test.items():
    #         log_writer_detection.write(f'{key}: {value}\n')
    #     log_writer_detection.write('\n')
    #     log_writer_detection.write('\n')
    #     log_writer_detection.close()
    #     file1 = open(args.output_dir+'/export_csvFile_TEST.txt',"a")
    #     file1.write("Epoch {:04d}: {} Student Localization mAP40 {:.5f} \n".format(epoch, datasetname__, 100*value[1]))
    #     file1.write("Epoch {:04d}: {} Student Localization mAP50 {:.5f} \n".format(epoch, datasetname__, 100*value[2]))
    #     file1.write("Epoch {:04d}: {} Student Localization mAP50-95 {:.5f} \n".format(epoch, datasetname__, 100*value[0]))
    #     file1.close()
    #     fields=[epoch, datasetname__, pth_files_sorted[index_pth], 'Student', 'Localization_'+datasetname__, '-', 100*value[1], 100*value[2], 100*value[0], '-'] # AUC_SliceLevel_Res
    #     with open(args.output_dir+'/export_csvFile_TEST.csv', 'a') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(fields)
    #     if args.modelEMA:
    #         print("Localization Testing Teacher: "+datasetname__)
    #         # model_ema.task_DetHead = 0
    #         test_stats, coco_evaluator, features_detectionList = evaluate(
    #             model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, device, args.output_dir,
    #             DetHead=DetHead_temp, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None))
    #         log_stats = { **{f'test_{k}': v for k, v in test_stats.items()} }
    #         print("Localization Testing Teacher: "+datasetname__)
    #         result_output_dir = args.output_dir + '/resultsTEST.txt'
    #         log_writer_detection = open(result_output_dir, 'a')
    #         log_writer_detection.write('Epoch: ' + str(epoch) + " TestingEMA: "+datasetname__ + '\n')
    #         log_writer_detection.write('-- Testing Teacher --' + '\n')
    #         formatted_stats_test = {f'testEMA_{k}': v for k, v in test_stats.items()}
    #         for key, value in formatted_stats_test.items():
    #             log_writer_detection.write(f'{key}: {value}\n')
    #         log_writer_detection.write('\n')
    #         log_writer_detection.write('\n')
    #         log_writer_detection.close()
    #         file1 = open(args.output_dir+'/export_csvFile_TEST.txt',"a")
    #         file1.write("Epoch {:04d}: {} Teacher Localization mAP40 {:.5f} \n".format(epoch, datasetname__, 100*value[1]))
    #         file1.write("Epoch {:04d}: {} Teacher Localization mAP50 {:.5f} \n".format(epoch, datasetname__, 100*value[2]))
    #         file1.write("Epoch {:04d}: {} Teacher Localization mAP50-95 {:.5f} \n\n".format(epoch, datasetname__, 100*value[0]))
    #         file1.close()
    #         fields=[epoch, datasetname__, pth_files_sorted[index_pth], 'Teacher', 'Localization_'+datasetname__, '-', 100*value[1], 100*value[2], 100*value[0], '-'] # AUC_SliceLevel_Res
    #         with open(args.output_dir+'/export_csvFile_TEST.csv', 'a') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(fields)

    ### Test SEGMENTATION ##
    if 'TESTseg' in args.cyclictask and 'candidptxSEG' in args.cyclictask:
        datasetname__ = "CANDID-PTX"
        head_number_temp = 2
        test_loader_loc_temp = test_loader_seg_CANDIDptx
        evaluate_for_segmentation(args.resume, model, model_ema, datasetname__, head_number_temp, test_loader_loc_temp, log_writter_DETECTION)
    if 'TESTseg' in args.cyclictask and 'chestxdetSEG' in args.cyclictask:
        datasetname__ = "ChestX-Det"
        head_number_temp = 4
        test_loader_loc_temp = test_loader_seg_ChestXDet
        evaluate_for_segmentation(args.resume, model, model_ema, datasetname__, head_number_temp, test_loader_loc_temp, log_writter_DETECTION)
    if 'TESTseg' in args.cyclictask and 'siimacrSEG' in args.cyclictask:
        datasetname__ = "SIIM-ACRpneumothorax"
        head_number_temp = 5
        test_loader_loc_temp = test_loader_seg_SIIM
        evaluate_for_segmentation(args.resume, model, model_ema, datasetname__, head_number_temp, test_loader_loc_temp, log_writter_DETECTION)
    # if 'TESTseg' in args.cyclictask:
    #     epoch = args.start_epoch
    #     test_y, test_p, _ = test_SEGMENTATION(args, model, test_loader_loc_temp, head_number=head_number_temp, log_writter_SEGMENTATION=log_writter_DETECTION)
    #     print("Epoch {}: Student {} Dice = {:.8f}%".format(epoch, datasetname__, 100.0 * dice_score(test_p, test_y)), file=log_writter_DETECTION)
    #     print("Epoch {}: Student {} Dice = {:.8f}%".format(epoch, datasetname__, 100.0 * dice_score(test_p, test_y)))
    #     file1 = open(args.output_dir+'/export_csvFile_TEST.txt',"a")
    #     file1.write("Epoch {:04d}: {} Teacher Segmentation AUC {:.5f} \n\n".format(epoch, datasetname__, 100.0 * dice_score(test_p, test_y)))
    #     file1.close()
    #     fields=[epoch, datasetname__, pth_files_sorted[index_pth], 'Student', 'Localization_'+datasetname__, '-', '-', '-', '-', 100.0 * dice_score(test_p, test_y)] # AUC_SliceLevel_Res
    #     with open(args.output_dir+'/export_csvFile_TEST.csv', 'a') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(fields)
    #     if args.modelEMA:
    #         test_y, test_p, _ = test_SEGMENTATION(args, model_ema, test_loader_loc_temp, head_number=head_number_temp, log_writter_SEGMENTATION=log_writter_DETECTION)
    #         print("Epoch {}: Teacher {} Dice = {:.8f}%".format(epoch, datasetname__, 100.0 * dice_score(test_p, test_y)), file=log_writter_DETECTION)
    #         print("Epoch {}: Teacher {} Dice = {:.8f}%".format(epoch, datasetname__, 100.0 * dice_score(test_p, test_y)))
    #         file1 = open(args.output_dir+'/export_csvFile_TEST.txt',"a")
    #         file1.write("Epoch {:04d}: {} Teacher Segmentation AUC {:.5f} \n\n".format(epoch, datasetname__, 100.0 * dice_score(test_p, test_y)))
    #         file1.close()
    #         fields=[epoch, datasetname__, pth_files_sorted[index_pth], 'Teacher', 'Localization_'+datasetname__, '-', '-', '-', '-', 100.0 * dice_score(test_p, test_y)] # AUC_SliceLevel_Res
    #         with open(args.output_dir+'/export_csvFile_TEST.csv', 'a') as f:
    #             writer = csv.writer(f)
    #             writer.writerow(fields)

    print("Done.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser('IntegratedModel Evaluation Script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)