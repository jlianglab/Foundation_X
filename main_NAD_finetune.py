# Copyright (c) 2022 IDEA. All Rights Reserved.
# ------------------------------------------------------------------------

import os 
### NAD-Implementation ##
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
from collections import OrderedDict
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
from torchmetrics.functional.classification import multilabel_accuracy
# Segmentation from Jiaxuan
from utils_segmentation import load_popar_weight, AverageMeter, save_model, save_model2, dice_score, mean_dice_coef, torch_dice_coef_loss, exp_lr_scheduler_with_warmup, step_decay, load_swin_pretrained
from datasets_medical import build_transform_segmentation, dataloader_return, PXSDataset, MontgomeryDataset, JSRTClavicleDataset, JSRTHeartDataset,JSRTLungDataset, VinDrRibCXRDataset, ChestXDetDataset, JSRTLungDataset, VindrCXRHeartDataset
from timm.utils import NativeScaler, ModelEma
from models.load_weights_model import load_weights, load_weights_resume, load_weights_resume2, load_weights_foundationX
import math

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer, create_optimizer_v2, optimizer_kwargs

from engineClsSeg import evaluate as evaluateLocSepFunc

import pandas as pd
import csv

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
tsne = TSNE(n_components=2, perplexity=30, n_iter=300)


# Print the current random seed for PyTorch
torch_seed = torch.initial_seed()
print(f"PyTorch random seed: {torch_seed}")

# Print the current random seed for NumPy
numpy_seed = np.random.seed()
print(f"NumPy random seed: {numpy_seed}")

device = torch.device("cuda")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    parser.add_argument('--segmentation_dataset_ann', type=int, default=None)
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
    parser.add_argument('--foundationX', default=None, type=str, help='foundationX from checkpoint')
    parser.add_argument('--foundationXMODEL', default=None, type=str, help='foundationX from checkpoint model or teacher_model')
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
    parser.add_argument('--IgnoreTest', action='store_true')
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
    parser.add_argument('--lr_backbone2', type=float, default=1e-5, metavar='LR_Backbone2',
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
    
    parser.add_argument('--eval_json_file_name', default=None, type=str, help='Eval Json File Name')

    parser.add_argument('--distributed', action='store_true')
    
    return parser


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

def Freeze_Backbone_and_Localization_Encoder(model):
    for name, param in model.named_parameters():
        if ('backbone' in name) and ('segmentation_' not in name): # BACKBONE only
            param.requires_grad = False
    for name, param in model.named_parameters():
        if ('transformer.encoder' in name): ## Only Localization Encoder
            param.requires_grad = False

    for name, param in model.named_parameters(): ## Newly added by NAD
        if ('transformer.enc_output' in name) or ('transformer.enc_output_norm' in name) or ('transformer.enc_outputMem_LinearProjection' in name): ## Localizer Encoder -> Query Selection Box Selection parts
            param.requires_grad = True
    return model

def unFreeze_Backbone_and_Localization_Encoder(model):
    for name, param in model.named_parameters():
        if ('backbone' in name) and ('segmentation_' not in name): # BACKBONE only
            param.requires_grad = True
    for name, param in model.named_parameters():
        if ('transformer.encoder' in name): ## Only Localization Encoder
            param.requires_grad = True
    return model


## Freeze_Backbone Freeze_Localization_Encoder  unFreeze_Backbone  unFreeze_Localization_Encoder
def Freeze_Backbone(model):
    for name, param in model.named_parameters():
        if ('backbone' in name) and ('segmentation_' not in name): # BACKBONE only
            param.requires_grad = False
    return model

def Freeze_Localization_Encoder(model):
    for name, param in model.named_parameters():
        if ('transformer.encoder' in name): ## Only Localization Encoder
            param.requires_grad = False

    for name, param in model.named_parameters(): ## Newly added by NAD
        if ('transformer.enc_output' in name) or ('transformer.enc_output_norm' in name) or ('transformer.enc_outputMem_LinearProjection' in name): ## Localizer Encoder -> Query Selection Box Selection parts
            param.requires_grad = True
    return model

def unFreeze_Backbone(model):
    for name, param in model.named_parameters():
        if ('backbone' in name) and ('segmentation_' not in name): # BACKBONE only
            param.requires_grad = True
    return model

def unFreeze_Localization_Encoder(model):
    for name, param in model.named_parameters():
        if ('transformer.encoder' in name): ## Only Localization Encoder
            param.requires_grad = True
    return model


## Freeze unFreeze for Segmentation Tasks -- Training Strategy 1
def Freeze_Backbone_SegmentationDecoder(model): # Segmentation Head should be always trainable (unFreeze)
    for name, param in model.named_parameters(): ## Freeze everything
            param.requires_grad = False
    for name, param in model.named_parameters(): ## only make the segmentation head trainable
        if 'segmentation_heads' in name:
            param.requires_grad = True
    return model
def unFreeze_Backbone_SegmentationDecoder(model): # Segmentation Head should be always trainable (unFreeze)
    for name, param in model.named_parameters(): ## unFreeze everything
            param.requires_grad = True
    return model


## Freeze unFreeze for Segmentation Tasks -- Training Strategy 2
def Freeze_SegBackbone_SegmentationDecoder(model): ## Freeze Backbone(SegEnc) and SegDecoder for Segmentation Training
    for name, param in model.named_parameters(): ## Freeze everything
            param.requires_grad = False
    for name, param in model.named_parameters(): ## only make the segmentation head trainable
        if 'segmentation_heads' in name:
            param.requires_grad = True
    return model
def Freeze_SegBackbone(model): ## Freeze only Backbone(SegEnc) for Segmentation Training
    for name, param in model.named_parameters(): 
            param.requires_grad = True
    for name, param in model.named_parameters():
        if 'backbone' in name: 
            param.requires_grad = False
        if 'segmentation_heads' in name or 'segmentation_PPN' in name or 'segmentation_FPN' in name: ## Make SegDec and Head trainable
            param.requires_grad = True
    return model



# def Freeze_Backbone_SegmentationDecoder(model): # Segmentation Head should be always trainable (unFreeze)
#     for name, param in model.named_parameters():
#         if ('backbone' in name) and ('segmentation_' not in name): # BACKBONE only
#             param.requires_grad = False
#         if (('backbone' in name) and ('segmentation_PPN' in name)) or (('backbone' in name) and ('segmentation_FPN' in name)): # Segmentaiton Decoder only
#             param.requires_grad = False
#         if 'segmentation_heads' in name:
#             param.requires_grad = True
#     return model
# def unFreeze_Backbone_SegmentationDecoder(model): # Segmentation Head should be always trainable (unFreeze)
#     for name, param in model.named_parameters():
#         if ('backbone' in name) and ('segmentation_' not in name): # BACKBONE only
#             param.requires_grad = True
#         if (('backbone' in name) and ('segmentation_PPN' in name)) or (('backbone' in name) and ('segmentation_FPN' in name)): # Segmentaiton Decoder only
#             param.requires_grad = True
#         if 'segmentation_heads' in name:
#             param.requires_grad = True
    # return model


def reinitialize_zero_weights(m):
    for name, param in m.named_parameters():
        if '.weight' in name and torch.sum(param.data) == 0:
            nn.init.xavier_uniform_(param.data)
    return m


def ema_update_teacher(model, teacher, momentum_schedule, it):
    with torch.no_grad():
        # if it < 10:
        #     m = momentum_schedule[it]  # momentum parameter
        # else:
        #     m = momentum_schedule[9]
        m = 0.80
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data) ## (1-m) taken from student model | m taken from teacher model
    return teacher

def ema_update_teacher_Seg(model, teacher, momentum_schedule, it, log_writter_SEGMENTATION=None):
    with torch.no_grad():
        # if it < args.total_epochs:
        #     m = momentum_schedule[it]  # momentum parameter
        # else:
        #     m = momentum_schedule[args.total_epochs-1]
        m = 0.80
        if args.opt == 'adamw_and_sgd' or args.opt == 'sgd':
            m = 0.90
        if log_writter_SEGMENTATION is not None:
            print("Epoch", it, " | Segmentation EMA Update Momentum m = ", m, file=log_writter_SEGMENTATION)
            log_writter_SEGMENTATION.flush()
        for param_q, param_k in zip(model.parameters(), teacher.parameters()):
        # for (name_q, param_q), (name_k, param_k) in zip(model.named_parameters(), teacher.named_parameters()):
            # with open("/home/nuislam/projects/DINO_Detection/IntegratedModel_GitHub_V/zDebugTestFindings/Monitor_EMAupdate.txt", "a") as f:
            #     f.write(f"Contribution from Student Model: {name_q} {param_q.detach().data.shape} {param_q.detach().data}\n")
            #     f.write(f"Contribution from Teacher Model: {name_k} {param_k.data.shape} {param_k.data}\n")
            
            # debug_temp_param_q = param_q.detach().data.sum().item()
            # debug_temp_param_k = param_k.data.sum().item()

            param_k.data.mul_(m).add_((1 - m) * param_q.detach().data) ## (1-m) taken from student model | m taken from teacher model

            # with open("/home/nuislam/projects/DINO_Detection/IntegratedModel_GitHub_V/zDebugTestFindings/Monitor_EMAupdate.txt", "a") as f:
            #     f.write(f"Contribution from Teacher (FINAL) Model: {name_k} {param_k.data.shape} {param_k.data}\n")
            #     f.write("-" * 30 + "\n")
            # fields=[it, name_k, param_k.data.shape, debug_temp_param_q, debug_temp_param_k, param_k.data.sum().item()] ## Train Loss
            # with open(args.output_dir+'/export_csvFile_DEBUG_ema.csv', 'a') as f:
            #     writer = csv.writer(f)
            #     writer.writerow(fields)
    return teacher


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


def evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp):  
    start_time = time.time()
    # epoch = args.start_epoch
    print("Testing Classification:", datasetname__)
    if datasetname__ in ['CheXpert', 'NIHChestXray14', 'VinDRCXR', 'NIHShenzhen', 'MIMICII', 'RSNApneumonia']:
        y_test, p_test = test_CLASSIFICATION(datasetname__, test_loader_cls_temp, model, head_number_temp, args)
        if args.modelEMA is not None:
            y_test_teacher, p_test_teacher = test_CLASSIFICATION(datasetname__, test_loader_cls_temp, model_ema, head_number_temp, args)

        if datasetname__ in ["CheXpert", 'MIMICII']:
            diseases = test_loader_cls_temp.dataset.diseases_LIST
            test_diseases_name = test_loader_cls_temp.dataset.diseases_LIST_test
            test_diseases = [diseases.index(c) for c in test_diseases_name]
            y_test = copy.deepcopy(y_test[:,test_diseases])
            p_test = copy.deepcopy(p_test[:, test_diseases])
            individual_results_student = metric_AUROC(y_test, p_test, len(test_diseases))
            auc_eval = np.array(individual_results_student).mean()
            if args.modelEMA is not None:
                y_test_teacher = copy.deepcopy(y_test_teacher[:,test_diseases])
                p_test_teacher = copy.deepcopy(p_test_teacher[:, test_diseases])
                individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(test_diseases)) 
                auc_eval_teacher = np.array(individual_results_teacher).mean()
        else:
            individual_results_student = metric_AUROC(y_test, p_test, len(test_loader_cls_temp.dataset.diseases_LIST))
            auc_eval = np.array(individual_results_student).mean()
            if args.modelEMA is not None:
                individual_results_teacher = metric_AUROC(y_test_teacher, p_test_teacher, len(test_loader_cls_temp.dataset.diseases_LIST)) 
                auc_eval_teacher = np.array(individual_results_teacher).mean()

        # print( "Epoch {:04d}: {} Student AUC {:.5f} ".format(epoch, datasetname__, individual_results_student), file=log_writter_DETECTION )
        # print( "Epoch {:04d}: {} Teacher AUC {:.5f} ".format(epoch, datasetname__, individual_results_teacher), file=log_writter_DETECTION )
        # file1 = open(args.output_dir+'/export_csvFile.txt',"a")
        # file1.write("Epoch {:04d}: {} Student Classification AUC {:.5f} \n".format(epoch, datasetname__, 100*auc_eval))
        # file1.write("Epoch {:04d}: {} Teacher Classification AUC {:.5f} \n".format(epoch, datasetname__, 100*auc_eval_teacher))
        # file1.close()
        fields=[epoch, task_todo, datasetname__, 'Student', 'Classification_'+datasetname__, str(100*auc_eval), '-', '-', '-', '-']
        with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        if args.modelEMA is not None:    
            fields=[epoch, task_todo, datasetname__, 'Teacher', 'Classification_'+datasetname__, str(100*auc_eval_teacher), '-', '-', '-', '-']
            with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)

    else:
        val_loss, auc_eval = evaluate_CLASSIFICATION(args, test_loader_cls_temp, model, criterion_CLS, log_writter_DETECTION, head_number=head_number_temp, task_cls_type=task_cls_type_temp)
        print( "Epoch {:04d}: {} Student Val Loss {:.5f} ".format(epoch, datasetname__, val_loss) )
        print( "Epoch {:04d}: {} Student Val Loss {:.5f} ".format(epoch, datasetname__, val_loss), file=log_writter_DETECTION )
        file1 = open(args.output_dir+'/export_csvFile.txt',"a")
        file1.write("Epoch {:04d}: {} Student Classification AUC {:.5f} \n".format(epoch, datasetname__, 100*auc_eval))
        file1.close()
        fields=[epoch, task_todo, datasetname__, 'Student', 'Classification_'+datasetname__, str(100*auc_eval), '-', '-', '-', '-']
        with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        if args.modelEMA is not None:
            val_loss, auc_eval = evaluate_CLASSIFICATION(args, test_loader_cls_temp, model_ema, criterion_CLS, log_writter_DETECTION, head_number=head_number_temp, task_cls_type=task_cls_type_temp)
            print( "Epoch {:04d}: {} Teacher Val Loss {:.5f} ".format(epoch, datasetname__, val_loss) )
            print( "Epoch {:04d}: {} Teacher Val Loss {:.5f} ".format(epoch, datasetname__, val_loss), file=log_writter_DETECTION )
            file1 = open(args.output_dir+'/export_csvFile.txt',"a")
            file1.write("Epoch {:04d}: {} Teacher Classification AUC {:.5f} \n\n".format(epoch, datasetname__, 100*auc_eval))
            file1.close()
            fields=[epoch, task_todo, datasetname__, 'Teacher', 'Classification_'+datasetname__, str(100*auc_eval), '-', '-', '-', '-']
            with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow(fields)
        end_time = time.time()
        hours, rem = divmod(end_time-start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Test Time {}: {:0>2}:{:0>2}:{:05.2f}".format(datasetname__, int(hours),int(minutes),seconds)) # log_writter
        # print("{} - Classification Test Time {}: {:0>2}:{:0>2}:{:05.2f}".format(args.resume, datasetname__, int(hours),int(minutes),seconds), file=log_writter_timer)

def evaluateLocSepFunc(epoch, datasetname__, task_todo, model, criterion, postprocessors, model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, TaskHead_number, DetHead_temp, wo_class_error, logger):
    print("Localization Testing Student: "+datasetname__)
    # model.task_DetHead = 0 ## Localization Heart
    test_stats, coco_evaluator, features_detectionList = evaluate(
        model, criterion, postprocessors, test_loader_loc_temp, base_ds_temp, device, args.output_dir,
        DetHead=DetHead_temp, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None))
    log_stats = { **{f'test_{k}': v for k, v in test_stats.items()} }
    print("Localization Testing Student: "+datasetname__)
    result_output_dir = args.output_dir + '/resultsTEST.txt'
    log_writer_detection = open(result_output_dir, 'a')
    log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Testing: "+datasetname__ + '\n')
    log_writer_detection.write('-- Testing --' + '\n')
    formatted_stats_test = {f'test_{k}': v for k, v in test_stats.items()}
    for key, value in formatted_stats_test.items():
        log_writer_detection.write(f'{key}: {value}\n')
    log_writer_detection.write('\n')
    log_writer_detection.write('\n')
    log_writer_detection.close()
    # export_csvFile = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Model', 'Task-Test', 'AUC', 'mAP40','mAP50','mAP50_95', 'DICE'])
    # export_csvFile.to_csv(args.output_dir+'/export_csvFile.csv', index=False)
    fields=[epoch, task_todo, datasetname__, 'Student', 'Localization_'+datasetname__, '-', 100*value[1], 100*value[2], 100*value[0], '-'] # AUC_SliceLevel_Res
    with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    if args.modelEMA:
        print("Localization Testing Teacher: "+datasetname__)
        # model_ema.task_DetHead = 0
        test_stats, coco_evaluator, features_detectionList = evaluate(
            model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, device, args.output_dir,
            DetHead=DetHead_temp, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None))
        log_stats = { **{f'test_{k}': v for k, v in test_stats.items()} }
        print("Localization Testing Teacher: "+datasetname__)
        result_output_dir = args.output_dir + '/resultsTEST.txt'
        log_writer_detection = open(result_output_dir, 'a')
        log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " TestingEMA: "+datasetname__ + '\n')
        log_writer_detection.write('-- Testing Teacher --' + '\n')
        formatted_stats_test = {f'testEMA_{k}': v for k, v in test_stats.items()}
        for key, value in formatted_stats_test.items():
            log_writer_detection.write(f'{key}: {value}\n')
        log_writer_detection.write('\n')
        log_writer_detection.write('\n')
        log_writer_detection.close()
        fields=[epoch, task_todo, datasetname__, 'Teacher', 'Localization_'+datasetname__, '-', 100*value[1], 100*value[2], 100*value[0], '-'] # AUC_SliceLevel_Res
        with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

def evaluateSegSepFunc(epoch, datasetname__, task_todo, model, model_ema, test_loader_loc_temp, head_number_temp, log_writter_DETECTION):
    test_y, test_p, _ = test_SEGMENTATION(model, test_loader_loc_temp, head_number=head_number_temp, log_writter_SEGMENTATION=log_writter_DETECTION)
    if datasetname__ != "ChestX-Det":
        temp_dice_score_res = 100.0 * dice_score(test_p, test_y)
        temp_mean_dice_score_res = 100.0 * mean_dice_coef(test_y > 0.5, test_p > 0.5)
    else: ## ChestX-Det Segmentation
        collection_res_DICE = []
        collection_res_mDICE = [] ## using this.
        for cls_value in range(0, 13):
            temp_test_p = test_p[:,cls_value,:,:]
            temp_test_y = test_y[:,cls_value,:,:]
            temp_res_DICE = 100.0 * dice_score(temp_test_p, temp_test_y)
            temp_res_mDICE = 100.0 * mean_dice_coef(np.expand_dims(temp_test_y, axis=1) > 0.5, np.expand_dims(temp_test_p, axis=1) > 0.5)
            print("[INFO] Epoch {} Student Test {} Class {} Dice = {:.5f}%".format(epoch, cls_value, datasetname__, temp_res_DICE), file=log_writter_DETECTION)
            print("[INFO] Epoch {} Student Test {} Class {} Mean Dice = {:.5f}%\n".format(epoch, cls_value, datasetname__, temp_res_mDICE), file=log_writter_DETECTION)
            collection_res_DICE.append(temp_res_DICE)
            collection_res_mDICE.append(temp_res_mDICE) ## using this.
        temp_dice_score_res = sum(collection_res_DICE) / len(collection_res_DICE)
        temp_mean_dice_score_res = sum(collection_res_mDICE) / len(collection_res_mDICE) ## using this.
    print("Epoch {}: Student {} Dice = {:.8f}%".format(epoch, datasetname__, temp_dice_score_res), file=log_writter_DETECTION)
    print("Epoch {}: Student {} Dice = {:.8f}%".format(epoch, datasetname__, temp_dice_score_res))
    print("Epoch {}: Student Mean {} Dice = {:.8f}%".format(epoch, datasetname__, temp_mean_dice_score_res), file=log_writter_DETECTION)
    print("Epoch {}: Student Mean {} Dice = {:.8f}%".format(epoch, datasetname__, temp_mean_dice_score_res))
    fields=[epoch, task_todo, datasetname__, 'Student', 'Segmentation_'+datasetname__, '-', '-', '-', '-', temp_mean_dice_score_res] # AUC_SliceLevel_Res
    with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
    if args.modelEMA:
        test_y, test_p, _ = test_SEGMENTATION(model_ema, test_loader_loc_temp, head_number=head_number_temp, log_writter_SEGMENTATION=log_writter_DETECTION)
        if datasetname__ != "ChestX-Det":
            temp_dice_score_res = 100.0 * dice_score(test_p, test_y)
            temp_mean_dice_score_res = 100.0 * mean_dice_coef(test_y > 0.5, test_p > 0.5)
        else: ## ChestX-Det Segmentation
            collection_res_DICE = []
            collection_res_mDICE = []
            for cls_value in range(0, 13):
                temp_test_p = test_p[:,cls_value,:,:]
                temp_test_y = test_y[:,cls_value,:,:]
                temp_res_DICE = 100.0 * dice_score(temp_test_p, temp_test_y)
                temp_res_mDICE = 100.0 * mean_dice_coef(np.expand_dims(temp_test_y, axis=1) > 0.5, np.expand_dims(temp_test_p, axis=1) > 0.5)
                print("[INFO] Epoch {} Teacher Test {} Class {} Dice = {:.5f}%".format(epoch, cls_value, datasetname__, temp_res_DICE), file=log_writter_DETECTION)
                print("[INFO] Epoch {} Teacher Test {} Class {} Mean Dice = {:.5f}%\n".format(epoch, cls_value, datasetname__, temp_res_mDICE), file=log_writter_DETECTION)
                collection_res_DICE.append(temp_res_DICE)
                collection_res_mDICE.append(temp_res_mDICE)
            temp_dice_score_res = sum(collection_res_DICE) / len(collection_res_DICE)
            temp_mean_dice_score_res = sum(collection_res_mDICE) / len(collection_res_mDICE)
        print("Epoch {}: Teacher {} Dice = {:.8f}%".format(epoch, datasetname__, temp_dice_score_res), file=log_writter_DETECTION)
        print("Epoch {}: Teacher {} Dice = {:.8f}%".format(epoch, datasetname__, temp_dice_score_res))
        print("Epoch {}: Teacher Mean {} Dice = {:.8f}%".format(epoch, datasetname__, temp_mean_dice_score_res), file=log_writter_DETECTION)
        print("Epoch {}: Teacher Mean {} Dice = {:.8f}%".format(epoch, datasetname__, temp_mean_dice_score_res))
        fields=[epoch, task_todo, datasetname__, 'Teacher', 'Segmentation_'+datasetname__, '-', '-', '-', '-', temp_mean_dice_score_res] # AUC_SliceLevel_Res
        with open(args.output_dir+'/export_csvFile.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(fields)

def train_one_epoch_SEGMENTATION(model, train_loader, optimizer, loss_scaler, epoch, head_number=None, log_writter_SEGMENTATION=None, model_ema=None, momen=None, coff=None, criterionMSE=None, train_type='uF'):
    model.train(True)
    if model_ema is not None:
        model_ema.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    lossesDICE = AverageMeter()
    lossesDICE_EMA = AverageMeter()
    lossesCONS1 = AverageMeter()
    lossesCONS2 = AverageMeter()
    criterion = torch_dice_coef_loss
    criterionMSE = torch.nn.MSELoss()
    end = time.time()

    # for idx, (img,mask, imgNonAug, maskNonAug) in enumerate(train_loader): # Disable Later - NAD
    for idx, (img,mask) in enumerate(train_loader):
        if args.debug:
            if idx == 25:
                print("Segmentation DEBUG BREAK!"*5)
                break
        if train_type == 'F':
            if idx == len(train_loader)//2:
                break
        elif train_type == 'F1':
            if idx == len(train_loader) * 1//3:
                break
        elif train_type == 'F2':
            if idx == len(train_loader) * 2//3:
                break

        data_time.update(time.time() - end)
        bsz = img.shape[0]

        img = img.cuda(non_blocking=True) 
        mask = mask.cuda(non_blocking=True) 

        img = img.float()
        mask = mask.float()

        # imgNonAug = imgNonAug.cuda(non_blocking=True)  # Disable Later - NAD
        # imgNonAug = imgNonAug.float() # Disable Later - NAD
        # maskNonAug = maskNonAug.float() # Disable Later - NAD

        # with torch.cuda.amp.autocast():
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            out_SegmentationHead, features_backbone, features_cons = model.module.backbone[0].extra_features_seg(img, head_n=head_number, model_to_train='student')
        else:
            out_SegmentationHead, features_backbone, features_cons = model.backbone[0].extra_features_seg(img, head_n=head_number, model_to_train='student')

        outputs = torch.sigmoid( out_SegmentationHead )

        del out_SegmentationHead

        loss = criterion(mask, outputs)
        loss_trainDice_temp = loss

        if model_ema is not None:
            # with torch.cuda.amp.autocast():
            # with torch.no_grad():
            if isinstance(model_ema, torch.nn.parallel.DistributedDataParallel):
                _, features_backbone_ema, features_cons_ema = model_ema.module.backbone[0].extra_features_seg(img, head_n=head_number, model_to_train='teacher')
                # _, features_backbone_ema, features_cons_ema = model_ema.module.backbone[0].extra_features_seg(imgNonAug, head_n=head_number, model_to_train='teacher')  # Disable Later - NAD
            else:
                _, features_backbone_ema, features_cons_ema = model_ema.backbone[0].extra_features_seg(img, head_n=head_number, model_to_train='teacher')
                # _, features_backbone_ema, features_cons_ema = model_ema.backbone[0].extra_features_seg(imgNonAug, head_n=head_number, model_to_train='teacher')  # Disable Later - NAD
            # loss_check_ema = criterion(mask, torch.sigmoid( out_SegmentationHead_ema ))
            # lossesDICE_EMA.update(loss_check_ema.item(), bsz)

            # features_backbone = features_backbone[-1]/features_backbone[-1].max()
            # features_backbone_ema = features_backbone_ema[-1]/features_backbone_ema[-1].max()
            # features_cons = features_cons/features_cons.max()
            # features_cons_ema = features_cons_ema/features_cons_ema.max()

            loss_cons1 = criterionMSE(features_backbone, features_backbone_ema)  ## 100x -- idea from Jiaxuan to make the MSE loss bigger.
            loss_cons2 = criterionMSE(features_cons, features_cons_ema)  ## 100x -- idea from Jiaxuan to make the MSE loss bigger.
            loss = (1-coff)*loss + coff*( (loss_cons1 + loss_cons2)/2 ) 
            del features_backbone, features_cons, features_backbone_ema, features_cons_ema
        else:
            loss_cons1 = 0
            loss_cons2 = 0

        # if model_ema is not None:
        ### Collecting SegmentationDecoder Weights for Regularizer
        # segmentation_block_weights = OrderedDict()
        # for name, param in model.named_parameters():
        #     # if "segmentation_PPN" in name or "segmentation_FPN" in name or "Segnorm" in name:
        #     if "backbone" in name or "segmentation_PPN" in name or "segmentation_FPN" in name:                
        #     # if "backbone" in name or "segmentation_PPN" in name or "segmentation_FPN" in name:
        #         segmentation_block_weights[name] = param
        # ### l2_regularizer with weight_decay implemented
        # loss = loss + ( 0.0001 * l2_regularizer(segmentation_block_weights) )
        # del segmentation_block_weights


        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), file=log_writter_SEGMENTATION)
            sys.exit(1)
        # update metric
        losses.update(loss.item(), bsz)
        lossesDICE.update(loss_trainDice_temp.item(), bsz)
        if loss_cons1 != 0:
            lossesCONS1.update(loss_cons1.item(), bsz)
        if loss_cons2 != 0:
            lossesCONS2.update(loss_cons2.item(), bsz)

        optimizer.zero_grad(set_to_none=True)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=None,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        # print("CHECK!!!CHECK!!!CHECK!!!CHECK!!!CHECK!!!CHECK!!!CHECK!!!CHECK!!!CHECK!!!CHECK!!!CHECK!!!")
        # for b in range(bsz):
        #     save_image(img[b].cpu().numpy().transpose(1, 2, 0), args.output_dir+"sample_images/{}_{}_input".format("train", b))
        #     save_image(mask[b].cpu().numpy(), args.output_dir+"sample_images/{}_{}_input".format("trainMASK", b))
        # exit(0)

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
                data_time=data_time, lr=optimizer.param_groups[0]['lr'], ttloss=losses))
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

    # if args.modelEMA == "True_Epoch" and train_type == 'uF': # Epoch based EMA update # ACTIVE
    # if args.modelEMA == "True_Epoch" and epoch > 36:
    if args.modelEMA == "True_Epoch":
        model_ema = ema_update_teacher_Seg(model, model_ema, momen, epoch, log_writter_SEGMENTATION)
        print("[Check] Teacher Model Updated....")
        print("[Check] Teacher Model Updated....", file=log_writter_SEGMENTATION)

    # print("Epoch {}: Loss_EMA = {}".format(epoch, lossesDICE_EMA.avg))
    # print("Epoch {}: Loss_EMA = {}".format(epoch, lossesDICE_EMA.avg), file=log_writter_SEGMENTATION)
    return losses.avg, lossesDICE.avg, lossesCONS1.avg,lossesCONS2.avg, model_ema
    # return losses.avg, lossesDICE.avg, lossesCONS1.avg,-1, model_ema


def evaluation_SEGMENTATION(model, val_loader, epoch, head_number=None, log_writter_SEGMENTATION=None):
    model.eval()
    losses = AverageMeter()
    criterion = torch_dice_coef_loss

    with torch.no_grad():
        for idx, (img, mask) in enumerate(val_loader):
            if args.debug:
                if idx == 50:
                    print("Segmentation Test Break!!"*5)
                    break
            bsz = img.shape[0]

            # img = img.double().cuda(non_blocking=True)
            # mask = mask.double().cuda(non_blocking=True)
            img = img.cuda(non_blocking=True) 
            mask = mask.cuda(non_blocking=True) 

            img = img.float()
            mask = mask.float()

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
                if idx == 25:
                    print("Segmentation Test Break!!"*5)
                    break
            bsz = img.shape[0]
            with torch.cuda.amp.autocast():
                # img = img.double().cuda(non_blocking=True)
                # mask = mask.cuda(non_blocking=True)
                img = img.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

                img = img.float()
                mask = mask.float()

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


def test_SEGMENTATION_3chVinDrCXR_OrganSeg(model, test_loader, head_number=None, log_writter_SEGMENTATION=None):
    features_segmentationList = []
    model.eval()
    with torch.no_grad():
        test_p0 = None
        test_y0 = None
        test_p1 = None
        test_y1 = None
        test_p2 = None
        test_y2 = None
        test_p_col = []
        test_y_col = []
        for idx, (img, mask) in enumerate(test_loader):

            if idx == 500:
                print("[INFO] Testing Break! only 500 Test Data as Validation.")
                print("[INFO] Testing Break! only 500 Test Data as Validation.", file=log_writter_SEGMENTATION)
                break

            if args.debug:
                if idx == 25:
                    print("Segmentation Test Break!!"*5)
                    break
            bsz = img.shape[0]
            with torch.cuda.amp.autocast():
                img = img.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

                img = img.float()
                mask = mask.float()

                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    out_SegmentationHead, _, _ = model.module.backbone[0].extra_features_seg(img, head_n=head_number)
                else:
                    out_SegmentationHead, _, _ = model.backbone[0].extra_features_seg(img, head_n=head_number)

                outputs0 = torch.sigmoid( out_SegmentationHead[0] )
                outputs1 = torch.sigmoid( out_SegmentationHead[1] )
                outputs2 = torch.sigmoid( out_SegmentationHead[2] )

                outputs0 = outputs0.cpu().detach()
                outputs1 = outputs1.cpu().detach()
                outputs2 = outputs2.cpu().detach()
                mask0 = mask[:,0,:].cpu().detach()
                mask0 = mask0.unsqueeze(dim=1)
                # mask0 = np.expand_dims(mask0, axis=1)
                mask1 = mask[:,1,:].cpu().detach()
                mask1 = mask1.unsqueeze(dim=1)
                # mask1 = np.expand_dims(mask1, axis=1)
                mask2 = mask[:,2,:].cpu().detach()
                mask2 = mask2.unsqueeze(dim=1)
                # mask2 = np.expand_dims(mask2, axis=1)

                # test_p_col = []
                # test_y_col = []

                if test_p0 is None and test_y0 is None:
                    test_p0 = outputs0
                    test_y0 = mask0
                else:
                    test_p0 = torch.cat((test_p0, outputs0), 0)
                    test_y0 = torch.cat((test_y0, mask0), 0)
                # test_p_col.append( test_p0.numpy() )
                # test_y_col.append( test_y0.numpy().reshape(test_p0.shape) )

                if test_p1 is None and test_y1 is None:
                    test_p1 = outputs1
                    test_y1 = mask1
                else:
                    test_p1 = torch.cat((test_p1, outputs1), 0)
                    test_y1 = torch.cat((test_y1, mask1), 0)
                # test_p_col.append( test_p1.numpy() )
                # test_y_col.append( test_y1.numpy().reshape(test_p1.shape) )

                if test_p2 is None and test_y2 is None:
                    test_p2 = outputs2
                    test_y2 = mask2
                else:
                    test_p2 = torch.cat((test_p2, outputs2), 0)
                    test_y2 = torch.cat((test_y2, mask2), 0)
                # test_p_col.append( test_p2.numpy() )
                # test_y_col.append( test_y2.numpy().reshape(test_p2.shape) )


                torch.cuda.empty_cache()
                if (idx + 1) % 100 == 0:
                    print("Testing Step[{}/{}] ".format(idx + 1, len(test_loader)), file=log_writter_SEGMENTATION)
                    log_writter_SEGMENTATION.flush()
                    # if conf.debug_mode:
                    #     break

        log_writter_SEGMENTATION.flush()
        # test_p = np.array(test_p_col)
        # test_y = np.array(test_y_col)

    test_p0 = test_p0.numpy()
    test_y0 = test_y0.numpy()
    test_y0 = test_y0.reshape(test_p0.shape)

    test_p1 = test_p1.numpy()
    test_y1 = test_y1.numpy()
    test_y1 = test_y1.reshape(test_p1.shape)

    test_p2 = test_p2.numpy()
    test_y2 = test_y2.numpy()
    test_y2 = test_y2.reshape(test_p2.shape)


    return test_y0, test_p0, test_y1, test_p1, test_y2, test_p2, features_segmentationList



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

def train_CLASSIFICATION(train_loader, model, criterion, optimizer, epoch, args, log_writter_CLASSIFICATION, head_number=None, model_ema=None, momen=None, coff=None, criterionMSE=None, task_cls_type=None):
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
        if args.debug and i == 100: # Debug
            break

        # measure data loading time
        data_time.update(time.time() - end)

        # if args.gpu is not None:
        #     images = images.cuda(args.gpu, non_blocking=True)
        # if torch.cuda.is_available():
        #     target = target.cuda(args.gpu, non_blocking=True)
        images, target = images.float().to(device), target.float().to(device) # NIH14 # int for CheXpert
        # print("[CHECK train_Classification] target value", target)
        # print("[CHECK train_Classification] images shape", images.shape)

        target = target.unsqueeze(1) ## For Binary Classification
        if args.taskcomponent == 'detection_vindrcxr_disease' or task_cls_type == 'nonBinary': ## For Multi-label Classification
          target = target.squeeze(1)


        # compute output
        # output = model(images)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            # _, out_classifierHead, _ = model.module.backbone(images)
            out_classifierHead, features_cons = model.module.backbone[0].extra_features(images, head_number)
        else:
            # _, out_classifierHead, _ = model.backbone(images)
            out_classifierHead, features_cons = model.backbone[0].extra_features(images, head_number)
        output = out_classifierHead
        # print(f"[CHECK Train_Classification] output{output.shape}, target shapes: { target.shape}")
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

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.debug and i == 200:
                break
            # if args.gpu is not None:
            #     images = images.cuda(args.gpu, non_blocking=True)
            # if torch.cuda.is_available():
            #     target = target.cuda(args.gpu, non_blocking=True)

            images, target = images.float().to(device), target.float().to(device) # NIH14 # int for CheXpert
            target = target.unsqueeze(1)
            if args.taskcomponent == 'detection_vindrcxr_disease' or task_cls_type == 'nonBinary':
              target = target.squeeze(1)
              
            # print("[CHECK Eval_Classification] images shape", images.shape)
            # print(target.shape)


            # compute output
            # if len(images.size()) == 4: # ## from TEST
            #     bs, c, h, w = images.size()
            #     n_crops = 1
            # elif len(images.size()) == 5: # ## from TEST
            #     bs, n_crops, c, h, w = images.size()
            #     images = torch.autograd.Variable(images.view(-1, c, h, w).to(device)) # ## from TEST

            # print("[Check Classification Test] images", images.shape)
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                # _, out_classifierHead, _ = model.module.backbone(images)
                out_classifierHead, _ = model.module.backbone[0].extra_features(images, head_number)
            else:
                # _, out_classifierHead, _ = model.backbone(images)
                out_classifierHead, _ = model.backbone[0].extra_features(images, head_number)
            output = out_classifierHead

            # output = torch.sigmoid( output )

            # output = output.view(bs, n_crops, -1).mean(1) ## from TEST
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

def test_CLASSIFICATION(dataset_NAME, data_loader_test, model, head_number, args):

  model.eval()

  y_test = torch.FloatTensor().cuda()
  p_test = torch.FloatTensor().cuda()

  with torch.no_grad():
    for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
      if args.debug and i == 250:
        break
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
        out_classifierHead, _ = model.module.backbone[0].extra_features(varInput, head_number)
      else:
        # _, out_classifierHead, _ = model.backbone(images)
        out_classifierHead, _ = model.backbone[0].extra_features(varInput, head_number)
      out = out_classifierHead
      # out = model(varInput)

      if dataset_NAME == "RSNApneumonia": ## RSNAPneumonia = Multi-class classification task
        out = torch.softmax(out, dim = 1)
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
    # Print the current random seed for PyTorch
    torch_seed = torch.initial_seed()
    print(f"PyTorch random seed: {torch_seed}")
    with open(args.output_dir+"/seed_val.txt", "w") as file:
        # Write the seeds to the file
        file.write(f"PyTorch random seed: {torch_seed}\n")

    # print("CHECK before utils.init_distributed_mode.")
    # utils.init_distributed_mode(args)
    # load cfg file and update the args
    print("Loading config file from {}".format(args.config_file))
    # time.sleep(args.rank * 0.02)
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
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)





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
    if args.taskcomponent in ["foundation_x_pretraining", "foundation_x2_pretraining"]: # Binary Class
        args.num_classes = 14
        args.dn_labelbook_size = 15
    if args.taskcomponent in ["foundation_x3_pretraining", "foundation_x3_FineTuning"]: # Binary Class
        args.num_classes = 14
        args.dn_labelbook_size = 15

    if args.taskcomponent in ["foundation_x3_pretraining", "foundation_x3_FineTuning"] and args.cyclictask == "locFT_TESTloc_objects365LOC": # Binary Class
        args.num_classes = 366
        args.dn_labelbook_size = 368


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
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    #     model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:'+str(n_parameters))
    # logger.info("params:\n"+json.dumps({n: p.numel() for n, p in model.named_parameters() if p.requires_grad}, indent=2))




    ### ----------------- Optimizer ----------------- ###
    if args.taskcomponent in ["detect_segmentation_cyclic", "detect_segmentation_cyclic_v4", 'detect_vindrcxr_heart_segTest', 'detect_vindrcxr_heart', 'detect_vindrcxr_leftlung', 'detect_vindrcxr_rightlung']:
        loss_scaler = NativeScaler()
        # args.lr_backbone = args.lr # 0.0001
        param_dicts = get_param_dict(args, model_without_ddp)
        if args.opt == "adamw":
            # optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
            optimizer = torch.optim.AdamW([
                {'params': [param for name, param in model.named_parameters() if 'backbone' in name and 'segmentation_' not in name], 'lr': args.lr_backbone},##Backbone
                {'params': [param for name, param in model.named_parameters() if 'segmentation_' in name], 'lr': args.lr_segmentor},##Segmentor
                {'params': [param for name, param in model.named_parameters() if 'transformer' in name], 'lr': args.lr_locDec},##Localizer
            ], lr=1e-3, weight_decay=args.weight_decay)

            args.lr_drop = 15
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
            print("[Info.] Lr Scheduler:", "StepLR", "Drop:", args.lr_drop)
        elif args.opt == "sgd":
            # optimizer = torch.optim.SGD(param_dicts, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=False)
            ### Backbone -- Segmentor -- Localizer
            optimizer = torch.optim.SGD([
                {'params': [param for name, param in model.named_parameters() if 'backbone' in name and 'segmentation_' not in name], 'lr': args.lr_backbone},##Backbone
                {'params': [param for name, param in model.named_parameters() if 'segmentation_' in name], 'lr': args.lr_segmentor},##Segmentor
                {'params': [param for name, param in model.named_parameters() if 'transformer' in name], 'lr': args.lr_locDec},##Localizer
            ], lr=1e-3, momentum=0.9, weight_decay=args.weight_decay)
            args.lr_drop = 20

        # args.num_classes = 3
        # args.dn_labelbook_size = 4

    if args.taskcomponent in ["detect_segmentation_cyclic_v2", "detect_segmentation_cyclic_v3", "segmentation_vindrcxr_organ_1h3ch", "segmentation"]:
        loss_scaler = NativeScaler()
        param_dicts = get_param_dict(args, model_without_ddp)
        if args.opt == "adamw":
            optimizer = torch.optim.AdamW([
                {'params': [param for name, param in model.named_parameters() if 'backbone' in name and 'segmentation_' not in name], 'lr': args.lr_backbone},##Backbone
                {'params': [param for name, param in model.named_parameters() if 'segmentation_' in name], 'lr': args.lr_segmentor},##Segmentor
                {'params': [param for name, param in model.named_parameters() if 'transformer.encoder' in name], 'lr': args.lr_locEnc},##Localizer Encoder
                {'params': [param for name, param in model.named_parameters() if 'transformer.decoder' in name], 'lr': args.lr_locDec},##Localizer Decoder
            ], lr=1e-3, weight_decay=args.weight_decay)

            args.lr_drop = 15
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
            print("[Info.] Lr Scheduler:", "StepLR", "Drop:", args.lr_drop)
        elif args.opt == "sgd":
            # optimizer = torch.optim.SGD(param_dicts, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=False)
            ### Backbone -- Segmentor -- Localizer
            optimizer = torch.optim.SGD([
                {'params': [param for name, param in model.named_parameters() if 'backbone' in name and 'segmentation_' not in name], 'lr': args.lr_backbone},##Backbone
                {'params': [param for name, param in model.named_parameters() if 'segmentation_' in name], 'lr': args.lr_segmentor},##Segmentor
                {'params': [param for name, param in model.named_parameters() if 'transformer.encoder' in name], 'lr': args.lr_locEnc},##Localizer Encoder
                {'params': [param for name, param in model.named_parameters() if 'transformer.decoder' in name], 'lr': args.lr_locDec},##Localizer Decoder
            ], lr=1e-3, momentum=0.9, weight_decay=args.weight_decay)
            args.lr_drop = 15

    # if args.taskcomponent == "segmentation" or args.taskcomponent == "segmentation_cyclic":
    if args.taskcomponent == "segmentation_cyclic":
        loss_scaler = NativeScaler()
        if args.opt == "adamw":
            # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            optimizer = torch.optim.AdamW([
                {'params': [param for name, param in model.named_parameters() if 'backbone' in name and 'segmentation_' not in name], 'lr': args.lr_backbone},##Backbone
                {'params': [param for name, param in model.named_parameters() if 'segmentation_' in name], 'lr': args.lr_segmentor},##Segmentor
                {'params': [param for name, param in model.named_parameters() if 'transformer' in name], 'lr': args.lr_locDec},##Localizer
            ], lr=args.lr_locDec, weight_decay=args.weight_decay)

            args.lr_drop = 15
        elif args.opt == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=False)

    if args.taskcomponent in ['detection', 'detection_baseline', 'detect_node21_nodule', 'detect_tbx11k_catagnostic', 'ClsLoc_tbx11k_catagnostic', 'detection_vindrcxr_disease', 'detection_vindrmammo_disease', 'detect_chestxdet_dataset']:
        # args.lr_backbone = args.lr
        # args.lr_backbone = 1e-5 # 0.0001
        loss_scaler = NativeScaler()
        param_dicts = get_param_dict(args, model_without_ddp)

        if args.opt == "adamw":
            optimizer = torch.optim.AdamW([
                {'params': [param for name, param in model.named_parameters() if 'backbone' in name and 'segmentation_' not in name], 'lr': args.lr_backbone},##Backbone
                {'params': [param for name, param in model.named_parameters() if 'segmentation_' in name], 'lr': args.lr_segmentor},##Segmentor
                {'params': [param for name, param in model.named_parameters() if 'transformer.encoder' in name], 'lr': args.lr_locEnc},##Localizer Encoder
                {'params': [param for name, param in model.named_parameters() if 'transformer.decoder' in name], 'lr': args.lr_locDec},##Localizer Decoder
            ], lr=args.lr_locDec, weight_decay=args.weight_decay)
            args.lr_drop = 20
        elif args.opt == "sgd":
            optimizer = torch.optim.SGD(param_dicts, lr=args.lr, weight_decay=args.weight_decay, momentum=0.9, nesterov=False)
            args.lr_drop = 20

    if args.taskcomponent in ["foundation_x_pretraining", "foundation_x2_pretraining", "foundation_x3_pretraining", "foundation_x3_FineTuning"]:
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
        elif args.opt == "sgd":
            optimizer = torch.optim.SGD([
                {'params': [param for name, param in model.named_parameters() if 'backbone' in name and 'segmentation_' not in name], 'lr': args.lr_backbone},##Backbone
                {'params': [param for name, param in model.named_parameters() if 'segmentation_' in name], 'lr': args.lr_segmentor},##Segmentor
                {'params': [param for name, param in model.named_parameters() if 'transformer.encoder' in name], 'lr': args.lr_locEnc},##Localizer Encoder
                {'params': [param for name, param in model.named_parameters() if 'transformer.decoder' in name], 'lr': args.lr_locDec},##Localizer Decoder
            ], lr=args.lr_locDec, momentum=0.9, weight_decay=args.weight_decay)
            args.lr_drop = 15
            print("[INFO] Using SGD Optimizer")
        elif args.opt == 'adamw_and_sgd':
            optimizer_adamw = torch.optim.AdamW([
                {'params': [param for name, param in model.named_parameters() if 'backbone' in name and 'segmentation_' not in name], 'lr': args.lr_backbone},##Backbone
                {'params': [param for name, param in model.named_parameters() if 'segmentation_' in name], 'lr': args.lr_segmentor},##Segmentor
                {'params': [param for name, param in model.named_parameters() if 'transformer.encoder' in name], 'lr': args.lr_locEnc},##Localizer Encoder
                {'params': [param for name, param in model.named_parameters() if 'transformer.decoder' in name], 'lr': args.lr_locDec},##Localizer Decoder
            ], lr=args.lr_locDec, weight_decay=args.weight_decay)

            optimizer_sgd = torch.optim.SGD([
                {'params': [param for name, param in model.named_parameters() if 'backbone' in name and 'segmentation_' not in name], 'lr': args.lr_backbone2},##Backbone
                {'params': [param for name, param in model.named_parameters() if 'segmentation_' in name], 'lr': args.lr_segmentor},##Segmentor
                {'params': [param for name, param in model.named_parameters() if 'transformer.encoder' in name], 'lr': args.lr_locEnc},##Localizer Encoder
                {'params': [param for name, param in model.named_parameters() if 'transformer.decoder' in name], 'lr': args.lr_locDec},##Localizer Decoder
            ], lr=args.lr_locDec, momentum=0.9, weight_decay=args.weight_decay)



    if args.taskcomponent == 'classification':
        if args.opt == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        elif args.opt == "sgd":
            optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=1e-4)

        if args.classification_dataset == 'imagenet':
            optimizer = create_optimizer(args, model)
            criterion = nn.CrossEntropyLoss().cuda(args.gpu)
        if args.classification_dataset == 'ChestXray14' or 'tbx11k':
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



    # if args.taskcomponent in ['ClsLoc_tbx11k_catagnostic']:
    #     logs_path = os.path.join(args.output_dir, "Logs")
    #     if not os.path.exists(logs_path):
    #         os.makedirs(logs_path)
    #     if os.path.exists(os.path.join(logs_path, "log.txt")):
    #         log_writter_DETECTION = open(os.path.join(logs_path, "log.txt"), 'a')
    #     else:
    #         log_writter_DETECTION = open(os.path.join(logs_path, "log.txt"), 'w')

    #     export_csvFile = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Model', 'Task-Test', 'AUC', 'mAP'])
    #     export_csvFile.to_csv(args.output_dir+'/export_csvFile.csv', index=False)

    #     print()
    #     print("-------------")
    #     print("[Information]  TASK:", args.taskcomponent)
    #     print("[Information]  Backbone:", args.backbonemodel) 
    #     print("[Information]  Backbone_INIT:", args.init)
    #     print("[Information]  Backbone Weights:", args.backbone_dir)
    #     print("[Information]  Dataset:", args.dataset_file)
    #     print("[Information]  Total Epoch:", args.total_epochs)
    #     print("[Information]  Batch Size:", args.batch_size)
    #     print("[Information]  Learning Rate:", args.lr)
    #     print("[Information]  Learning Rate Backbone:", args.lr_backbone)
    #     print("[Information]  Num Workers:", args.num_workers)
    #     print("[Information]  Optimizer:", args.opt)
    #     print("[Information]  Output Dir:", args.output_dir)
    #     print("-------------")
    #     print()

    #     print("-------------", file=log_writter_DETECTION)
    #     print("[Information]  TASK:", args.taskcomponent, file=log_writter_DETECTION)
    #     print("[Information]  Backbone:", args.backbonemodel, file=log_writter_DETECTION)
    #     print("[Information]  Backbone_INIT:", args.init, file=log_writter_DETECTION)
    #     print("[Information]  Backbone Weights:", args.backbone_dir, file=log_writter_DETECTION)
    #     print("[Information]  Dataset:", args.dataset_file, file=log_writter_DETECTION)
    #     print("[Information]  Total Epoch:", args.total_epochs, file=log_writter_DETECTION)
    #     print("[Information]  Batch Size:", args.batch_size, file=log_writter_DETECTION)
    #     print("[Information]  Learning Rate:", args.lr, file=log_writter_DETECTION)
    #     print("[Information]  Learning Rate Backbone:", args.lr_backbone, file=log_writter_DETECTION)
    #     print("[Information]  Num Workers:", args.num_workers, file=log_writter_DETECTION)
    #     print("[Information]  Optimizer:", args.opt, file=log_writter_DETECTION)
    #     print("[Information]  Output Dir:", args.output_dir, file=log_writter_DETECTION)
    #     print("-------------", file=log_writter_DETECTION)

        
    #     if args.backbone_dir is not None:
    #         model = load_weights(model, args)
    #     if args.modelEMA is not None:
    #         model_ema = copy.deepcopy(model)
    #         print("[Model Info.] Using Epoch-wise EMA Model for Teacher.")
    #         for p in model_ema.parameters():
    #             p.requires_grad = False

    #     criterion_CLS = torch.nn.BCEWithLogitsLoss()
    #     train_loader_Cls, val_loader_Cls, data_loader_train, data_loader_val, sampler_train, dataset_val= dataloader_return(args)
    #     base_ds = get_coco_api_from_dataset(dataset_val)


    if args.taskcomponent in ['detect_chestxdet_dataset']: # detect_chestxdet_dataset
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
        print("[Information]  Learning Rate Loc:", args.lr_locDec)
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

        # try:
        #     data_loader_train, data_loader_val, sampler_train, dataset_val = dataloader_return(args)
        #     base_ds = get_coco_api_from_dataset(dataset_val)
        # except:
        #     do_nothing = 1

        criterion_CLS = torch.nn.BCEWithLogitsLoss()
        if args.dataset_file == "chestxdetdataset":
            data_loader_train, data_loader_val, sampler_train, dataset_val, train_loader_seg, test_loader_seg, train_loader_Cls, val_loader_Cls = dataloader_return(args)
            base_ds = get_coco_api_from_dataset(dataset_val)
            for name, param in model.named_parameters():
                if ('decoder.1' in name) or ('decoder.2' in name) or ('segmentation' in name): ## Only Localization Encoder
                    param.requires_grad = False
            model_ema = copy.deepcopy(model)
            export_csvFile = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Model', 'Task-Test', 'AUC', 'mAP40','mAP50','mAP50_95', 'DICE'])
            export_csvFile.to_csv(args.output_dir+'/export_csvFile.csv', index=False)

        BEST_STUDENT_CLS_AUC = -1
        BEST_TEACHER_CLS_AUC = -1
        BEST_STUDENT_LOC_mAP = -1
        BEST_TEACHER_LOC_mAP = -1
        BEST_STUDENT_SEG_DICE = -1
        BEST_TEACHER_SEG_DICE = -1


    if args.taskcomponent in ['detection', 'detection_baseline', 'detect_node21_nodule', 'detect_tbx11k_catagnostic', 'detection_vindrcxr_disease', 'detection_vindrmammo_disease', 'ClsLoc_tbx11k_catagnostic']: # detect_chestxdet_dataset
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
        print("[Information]  Learning Rate Loc:", args.lr_locDec)
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

        if args.dataset_file == "vindrcxr_detect":
            data_loader_train, data_loader_val, sampler_train, dataset_val, train_loader_Cls, val_loader_Cls = dataloader_return(args)
            for name, param in model.named_parameters():
                if ('decoder.1' in name) or ('decoder.2' in name) or ('segmentation' in name): ## Only Localization Encoder
                    param.requires_grad = False
            model_ema = copy.deepcopy(model)
            if args.eval == False:
                export_csvFile = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Model', 'Task-Test', 'AUC', 'mAP40','mAP50','mAP50_95'])
                export_csvFile.to_csv(args.output_dir+'/export_csvFile.csv', index=False)

        if args.dataset_file == "detection_vindrmammo_disease":
            data_loader_train, data_loader_val, sampler_train, dataset_val, train_loader_Cls, val_loader_Cls = dataloader_return(args)
            for name, param in model.named_parameters():
                if ('decoder.1' in name) or ('decoder.2' in name) or ('segmentation' in name): ## Only Localization Encoder
                    param.requires_grad = False
            model_ema = copy.deepcopy(model)
            if args.eval == False:
                export_csvFile = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Model', 'Task-Test', 'AUC', 'mAP40','mAP50','mAP50_95'])
                export_csvFile.to_csv(args.output_dir+'/export_csvFile.csv', index=False)
            
        if args.dataset_file == "node21_noduleDataset":
            data_loader_train, data_loader_val, sampler_train, dataset_val, train_loader_Cls, val_loader_Cls = dataloader_return(args)
            for name, param in model.named_parameters():
                if ('decoder.1' in name) or ('decoder.2' in name) or ('segmentation' in name): ## Only Localization Encoder
                    param.requires_grad = False
            model_ema = copy.deepcopy(model)
            if args.eval == False:
                export_csvFile = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Model', 'Task-Test', 'AUC', 'mAP40','mAP50','mAP50_95'])
                export_csvFile.to_csv(args.output_dir+'/export_csvFile.csv', index=False)

        if args.dataset_file == "tbx11k_catagnostic": ####
            data_loader_train, data_loader_val, sampler_train, dataset_val, train_loader_Cls, val_loader_Cls = dataloader_return(args)
            for name, param in model.named_parameters():
                if ('decoder.1' in name) or ('decoder.2' in name) or ('segmentation' in name): ## Only Localization Encoder
                    param.requires_grad = False
            model_ema = copy.deepcopy(model)
            if args.eval == False:
                export_csvFile = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Model', 'Task-Test', 'AUC', 'mAP40','mAP50','mAP50_95'])
                export_csvFile.to_csv(args.output_dir+'/export_csvFile.csv', index=False)
            if args.foundationX is not None:
                print("Checkpoint Received from Foundation-x:", args.foundationX)

        BEST_STUDENT_CLS_AUC = -1
        BEST_TEACHER_CLS_AUC = -1
        BEST_STUDENT_LOC_mAP = -1
        BEST_TEACHER_LOC_mAP = -1

        if args.onecyclelr:
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr_locDec, steps_per_epoch=len(data_loader_train), epochs=args.total_epochs, pct_start=0.2)
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

        if args.eval:
            if args.pretrain_model_path:
                model, model_ema, _, epoch = load_weights_resume(model, model_ema, optimizer, args)

                model.task_DetHead = 0
                test_stats, _, _ = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, DetHead=0, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None))
                log_stats = { **{f'test_Heart_{k}': v for k, v in test_stats.items()} }
                result_output_dir = args.output_dir + '/results_EvaluationOnly.txt'
                log_writer_detection = open(result_output_dir, 'a')
                formatted_stats_test = {f'test_{k}': v for k, v in test_stats.items()}
                log_writer_detection.write('Epoch: ' + str(epoch) + '\n')
                log_writer_detection.write('Pretrained Model: ' + args.pretrain_model_path + '\n')
                log_writer_detection.write("Dataset: " + args.dataset_file + '\n')
                log_writer_detection.write('-- Student Model Testing --' + '\n')
                for key, value in formatted_stats_test.items():
                    log_writer_detection.write(f'{key}: {value}\n')
                log_writer_detection.write('\n')
                # str(100*value[1]), str(100*value[2]), str(100*value[0])   
                log_writer_detection.write('Localization mAP40: ' + str(100*value[1]) + '\n') # mAP40
                log_writer_detection.write('Localization mAP50: ' + str(100*value[2]) + '\n') # mAP50
                log_writer_detection.write('Localization mAP50:95: ' + str(100*value[0]) + '\n') # mAP50:95
                log_writer_detection.write('\n')
                log_writer_detection.write('\n')
                log_writer_detection.close()        

                model_ema.task_DetHead = 0
                test_stats, _, _ = evaluate(model_ema, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, DetHead=0, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None))
                log_stats = { **{f'test_Heart_{k}': v for k, v in test_stats.items()} }
                result_output_dir = args.output_dir + '/results_EvaluationOnly.txt'
                log_writer_detection = open(result_output_dir, 'a')
                formatted_stats_test = {f'test_{k}': v for k, v in test_stats.items()}
                log_writer_detection.write('Epoch: ' + str(epoch) + '\n')
                log_writer_detection.write('Pretrained Model: ' + args.pretrain_model_path + '\n')
                log_writer_detection.write("Dataset: " + args.dataset_file + '\n')
                log_writer_detection.write('-- Teacher Model Testing --' + '\n')
                for key, value in formatted_stats_test.items():
                    log_writer_detection.write(f'{key}: {value}\n')
                log_writer_detection.write('\n')
                # str(100*value[1]), str(100*value[2]), str(100*value[0])   
                log_writer_detection.write('Localization mAP40: ' + str(100*value[1]) + '\n') # mAP40
                log_writer_detection.write('Localization mAP50: ' + str(100*value[2]) + '\n') # mAP50
                log_writer_detection.write('Localization mAP50:95: ' + str(100*value[0]) + '\n') # mAP50:95
                log_writer_detection.write('\n')
                log_writer_detection.write('\n')
                log_writer_detection.close() 
                exit(0)
            else:
                print(" --- No pretrain_model_path given ---")
                exit(0)


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



    if args.taskcomponent == 'segmentation' or args.taskcomponent == 'segmentation_cyclic' or args.taskcomponent == "segmentation_vindrcxr_organ_1h3ch":
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
        print("[Information]  Learning Rate Backbone:", args.lr_backbone)
        print("[Information]  Learning Rate Segmentation:", args.lr_segmentor)
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
        print("[Information]  Learning Rate Backbone:", args.lr_backbone, file=log_writter_SEGMENTATION)
        print("[Information]  Learning Rate Segmentation:", args.lr_segmentor, file=log_writter_SEGMENTATION)
        print("[Information]  Num Workers:", num_workers, file=log_writter_SEGMENTATION)
        print("[Information]  Optimizer:", args.opt, file=log_writter_SEGMENTATION)
        print("[Information]  Patience:", patience_SEGMENTATION, file=log_writter_SEGMENTATION)
        print("[Information]  Output Dir:", args.output_dir, file=log_writter_SEGMENTATION)
        print("-------------", file=log_writter_SEGMENTATION)


        if args.segmentation_dataset == 'jsrt_lung':
            train_loader, val_loader, test_loader = dataloader_return(args)
            if args.resume is not None:
                # model = load_weights_resume(model, args)
                mode, model_ema = load_weights_foundationX(model, model_ema, optimizer, args, to_load='model')
        elif args.segmentation_dataset == 'jsrt_clavicle':
            train_loader, val_loader, test_loader = dataloader_return(args)
            if args.resume is not None:
                # model = load_weights_resume(model, args)
                mode, model_ema = load_weights_foundationX(model, model_ema, optimizer, args, to_load='model')
        elif args.segmentation_dataset == 'jsrt_heart': # 
            train_loader, val_loader, test_loader = dataloader_return(args)
            if args.resume is not None:
                # model = load_weights_resume(model, args)
                mode, model_ema = load_weights_foundationX(model, model_ema, optimizer, args, to_load='model')
        elif args.segmentation_dataset == 'jsrt_leftlung':
            train_loader, val_loader, test_loader = dataloader_return(args)
            if args.resume is not None:
                # model = load_weights_resume(model, args)
                mode, model_ema = load_weights_foundationX(model, model_ema, optimizer, args, to_load='model')
        elif args.segmentation_dataset == 'chestxdetdataset': # Disease Segmentation
            train_loader, test_loader = dataloader_return(args)
        elif args.segmentation_dataset == 'jsrt_lung_heart_clavicle': # For cyclic
            train_loader_jsrtLung, val_loader_jsrtLung, test_loader_jsrtLung, train_loader_jsrtClavicle, val_loader_jsrtClavicle, test_loader_jsrtClavicle, train_loader_jsrtHeart, val_loader_jsrtHeart, test_loader_jsrtHeart = dataloader_return(args)
        elif args.segmentation_dataset == 'vindrcxr_lung':
            train_loader, val_loader = dataloader_return(args)
        elif args.segmentation_dataset == 'vindrcxr_heart':
            train_loader, val_loader = dataloader_return(args)
        elif args.segmentation_dataset == 'vindrcxr_leftlung':
            train_loader, val_loader = dataloader_return(args)
        elif args.segmentation_dataset == 'vindrcxr_rightlung':
            train_loader, val_loader = dataloader_return(args)
        elif args.segmentation_dataset == 'vindrcxr_lung_heart':
            train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLung, val_loader_vindrcxrtLung = dataloader_return(args)
        elif args.segmentation_dataset == "segmentation_vindrcxr_organ_1h3ch":
            train_loader, val_loader = dataloader_return(args)
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


        args.total_epochs = args.total_epochs
        img_size = args.imgsize # 448 worked
        batch_size = args.batch_size
        num_workers = args.num_workers

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


        elif args.classification_dataset == "ChestXray14" or "tbx11k":
            args.total_epochs = 200
            img_size = args.imgsize # 448 worked
            batch_size = args.batch_size # 128
            num_workers = args.num_workers # 16 | 32
            # best_acc1_CLASSIFICATION = 100000
            best_val_CLASSIFICATION = 10000

            patience_counter_CLASSIFICATION = 0
            patience_CLASSIFICATION = 35

            train_loader, val_loader = dataloader_return(args)

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



    if args.taskcomponent == "detect_segmentation_cyclic_v2" or args.taskcomponent == "detect_segmentation_cyclic_v3": ## With EMA -- Teacher-Student Model
        if args.backbone_dir is not None or args.resume is not None: # 
            model = load_weights(model, args)
        if args.modelEMA is not None:
            model_ema = copy.deepcopy(model)
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

        if not os.path.exists(args.output_dir+'/export_csvFile.csv'):
            export_csvFile = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Model', 'Task-Test', 'mAP', 'DICE'])
            export_csvFile.to_csv(args.output_dir+'/export_csvFile.csv', index=False)

        if not os.path.exists(args.output_dir+'/export_csvFile_TRAIN.csv'):
            export_csvFile = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Model', 'Loc_Loss', 'Seg_Loss', 'Seg_DiceLoss', 'Seg_BackboneLoss', 'Seg_SegDecLoss'])
            export_csvFile.to_csv(args.output_dir+'/export_csvFile_TRAIN.csv', index=False)


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
        print("[Information]  Learning Rate Backbone:", args.lr_backbone)
        print("[Information]  Learning Rate Loc.Encoder:", args.lr_locEnc)
        print("[Information]  Learning Rate Loc.Decoder:", args.lr_locDec)
        print("[Information]  Num Workers:", num_workers)
        print("[Information]  Optimizer:", args.opt)
        print("[Information]  Output Dir:", args.output_dir)
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
        print("[Information]  Learning Rate Backbone:", args.lr_backbone, file=log_writter_SEGMENTATION)
        print("[Information]  Learning Rate Loc.Encoder:", args.lr_locEnc, file=log_writter_SEGMENTATION)
        print("[Information]  Learning Rate Loc.Decoder:", args.lr_locDec, file=log_writter_SEGMENTATION)
        print("[Information]  Num Workers:", num_workers, file=log_writter_SEGMENTATION)
        print("[Information]  Optimizer:", args.opt, file=log_writter_SEGMENTATION)
        print("[Information]  Output Dir:", args.output_dir, file=log_writter_SEGMENTATION)

        print("[Information]  L2 Regularizer for Loc.Encoder:", "Yes", file=log_writter_SEGMENTATION)
        print("[Information]  Multi-SeqLayers:", "Yes", file=log_writter_SEGMENTATION)
        print("[Information]  Multi-Decoders:", "Yes", file=log_writter_SEGMENTATION)
        print("[Information]  Localizer Component Loss:", "Encoder and Decoder Loss", file=log_writter_SEGMENTATION)
        # print("[Information]  Localizer Component Loss:", "Only Decoder Loss", file=log_writter_SEGMENTATION)

        print("-------------", file=log_writter_SEGMENTATION)

        loss_scaler = NativeScaler()

        # train_loader_vindrcxrHeart_A, train_loader_vindrcxrHeart_B, val_loader_vindrcxrtHeart, train_loader_vindrcxrLeftLung_A, train_loader_vindrcxrLeftLung_B, val_loader_vindrcxrtLeftLung, train_loader_vindrcxrRightLung_A, train_loader_vindrcxrRightLung_B, val_loader_vindrcxrtRightLung, \
        #     data_loader_trainHeartA, sampler_trainHeartA, data_loader_trainHeartB, sampler_trainHeartB, data_loader_valHeart, dataset_valHeart, \
        #     data_loader_trainLeftLungA, sampler_trainLeftLungA, data_loader_trainLeftLungB, sampler_trainLeftLungB, data_loader_valLeftLung, dataset_valLeftLung, \
        #     data_loader_trainRightLungA, sampler_trainRightLungA, data_loader_trainRightLungB, sampler_trainRightLungB, data_loader_valRightLung, dataset_valRightLung = dataloader_return(args)

        train_loader_vindrcxrHeart_A, val_loader_vindrcxrtHeart, train_loader_vindrcxrLeftLung_A, val_loader_vindrcxrtLeftLung, train_loader_vindrcxrRightLung_A, val_loader_vindrcxrtRightLung, val_loader_vindrcxrt3ch, \
            data_loader_trainHeartA, sampler_trainHeartA, data_loader_valHeart, dataset_valHeart, \
            data_loader_trainLeftLungA, sampler_trainLeftLungA, data_loader_valLeftLung, dataset_valLeftLung, \
            data_loader_trainRightLungA, sampler_trainRightLungA, data_loader_valRightLung, dataset_valRightLung = dataloader_return(args)
        

        base_ds_Heart = get_coco_api_from_dataset(dataset_valHeart)
        base_ds_LeftLung = get_coco_api_from_dataset(dataset_valLeftLung)
        base_ds_RightLung = get_coco_api_from_dataset(dataset_valRightLung)

        BEST_STUDENT_LOC_H_mAP = -1
        BEST_TEACHER_LOC_H_mAP = -1
        BEST_STUDENT_LOC_LL_mAP = -1
        BEST_TEACHER_LOC_LL_mAP = -1
        BEST_STUDENT_LOC_RL_mAP = -1
        BEST_TEACHER_LOC_RL_mAP = -1

        BEST_STUDENT_SEG_H_mAP = -1
        BEST_TEACHER_SEG_H_mAP = -1
        BEST_STUDENT_SEG_LL_mAP = -1
        BEST_TEACHER_SEG_LL_mAP = -1
        BEST_STUDENT_SEG_RL_mAP = -1
        BEST_TEACHER_SEG_RL_mAP = -1


    # if args.taskcomponent == "detect_segmentation_cyclic_v3": ## With EMA -- Teacher-Student Model
    #     if args.backbone_dir is not None or args.resume is not None: # 
    #         model = load_weights(model, args)
    #     if args.modelEMA is not None:
    #         model_ema = copy.deepcopy(model)
    #         print("[Model Info.] Using Epoch-wise EMA Model for Teacher.")
    #         for p in model_ema.parameters():
    #             p.requires_grad = False

    #     model_path_SEGMENTATION = args.output_dir
    #     if not os.path.exists(model_path_SEGMENTATION):
    #         os.makedirs(model_path_SEGMENTATION)
    #     logs_path = os.path.join(model_path_SEGMENTATION, "Logs")
    #     if not os.path.exists(logs_path):
    #         os.makedirs(logs_path)
    #     if os.path.exists(os.path.join(logs_path, "log.txt")):
    #         log_writter_SEGMENTATION = open(os.path.join(logs_path, "log.txt"), 'a')
    #     else:
    #         log_writter_SEGMENTATION = open(os.path.join(logs_path, "log.txt"), 'w')

    #     export_csvFile = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Model', 'Task-Test', 'mAP', 'DICE'])
    #     export_csvFile.to_csv(args.output_dir+'/export_csvFile.csv', index=False)


    #     args.total_epochs = args.total_epochs
    #     img_size = args.imgsize # 448 worked
    #     batch_size = args.batch_size
    #     num_workers = args.num_workers
    #     best_val_loss_SEGMENTATION = 100000
    #     patience_SEGMENTATION = 50
    #     patience_counter_SEGMENTATION = 0

    #     print()
    #     print("-------------")
    #     print("[Information]  TASK:", args.taskcomponent)
    #     print("[Information]  EMA:", args.modelEMA)
    #     print("[Information]  Backbone:", args.backbonemodel) 
    #     print("[Information]  Backbone_INIT:", args.init)
    #     print("[Information]  Backbone Weights:", args.backbone_dir)
    #     print("[Information]  Dataset:", args.segmentation_dataset)
    #     print("[Information]  Total Epoch:", args.total_epochs)
    #     print("[Information]  Image Size:", img_size)
    #     print("[Information]  Batch Size:", batch_size)
    #     print("[Information]  Learning Rate:", args.lr)
    #     print("[Information]  Learning Rate Backbone:", args.lr_backbone)
    #     print("[Information]  Num Workers:", num_workers)
    #     print("[Information]  Optimizer:", args.opt)
    #     print("[Information]  Output Dir:", args.output_dir)
    #     print("-------------")
    #     print() # log_writter_SEGMENTATION

    #     print("-------------", file=log_writter_SEGMENTATION)
    #     print("[Information]  TASK:", args.taskcomponent, file=log_writter_SEGMENTATION)
    #     print("[Information]  EMA:", args.modelEMA, file=log_writter_SEGMENTATION)
    #     print("[Information]  Backbone:", args.backbonemodel, file=log_writter_SEGMENTATION)
    #     print("[Information]  Backbone_INIT:", args.init, file=log_writter_SEGMENTATION)
    #     print("[Information]  Backbone Weights:", args.backbone_dir, file=log_writter_SEGMENTATION)
    #     print("[Information]  Dataset:", args.segmentation_dataset, file=log_writter_SEGMENTATION)
    #     print("[Information]  Total Epoch:", args.total_epochs, file=log_writter_SEGMENTATION)
    #     print("[Information]  Image Size:", img_size, file=log_writter_SEGMENTATION)
    #     print("[Information]  Batch Size:", batch_size, file=log_writter_SEGMENTATION)
    #     print("[Information]  Learning Rate Backbone:", args.lr_backbone, file=log_writter_SEGMENTATION)
    #     print("[Information]  Learning Rate Loc.Encoder:", args.lr_locEnc, file=log_writter_SEGMENTATION)
    #     print("[Information]  Learning Rate Loc.Decoder:", args.lr_locDec, file=log_writter_SEGMENTATION)
    #     print("[Information]  Num Workers:", num_workers, file=log_writter_SEGMENTATION)
    #     print("[Information]  Optimizer:", args.opt, file=log_writter_SEGMENTATION)
    #     print("[Information]  Output Dir:", args.output_dir, file=log_writter_SEGMENTATION)

    #     print("[Information]  L2 Regularizer for Loc.Encoder:", "Yes", file=log_writter_SEGMENTATION)
    #     print("[Information]  Multi-SeqLayers:", "Yes", file=log_writter_SEGMENTATION)
    #     print("[Information]  Multi-Decoders:", "Yes", file=log_writter_SEGMENTATION)
    #     print("[Information]  Localizer Component Loss:", "Encoder and Decoder Loss", file=log_writter_SEGMENTATION)
    #     # print("[Information]  Localizer Component Loss:", "Only Decoder Loss", file=log_writter_SEGMENTATION)
    #     print("-------------", file=log_writter_SEGMENTATION)

    #     loss_scaler = NativeScaler()

    #     train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLeftLung, val_loader_vindrcxrtLeftLung, train_loader_vindrcxrRightLung, val_loader_vindrcxrtRightLung, \
    #         data_loader_trainHeartA, sampler_trainHeartA, data_loader_trainHeartB, sampler_trainHeartB, data_loader_trainHeartC, sampler_trainHeartC, data_loader_valHeart, dataset_valHeart, \
    #         data_loader_trainLeftLungA, sampler_trainLeftLungA, data_loader_trainLeftLungB, sampler_trainLeftLungB, data_loader_trainLeftLungC, sampler_trainLeftLungC, data_loader_valLeftLung, dataset_valLeftLung, \
    #         data_loader_trainRightLungA, sampler_trainRightLungA, data_loader_trainRightLungB, sampler_trainRightLungB, data_loader_trainRightLungC, sampler_trainRightLungC, data_loader_valRightLung, dataset_valRightLung = dataloader_return(args)

    #     base_ds_Heart = get_coco_api_from_dataset(dataset_valHeart)
    #     base_ds_LeftLung = get_coco_api_from_dataset(dataset_valLeftLung)
    #     base_ds_RightLung = get_coco_api_from_dataset(dataset_valRightLung)


    if args.taskcomponent == "detect_segmentation_cyclic_v4": ## With EMA -- Teacher-Student Model
        if args.backbone_dir is not None or args.resume is not None: # 
            model = load_weights(model, args)
            # model = reinitialize_zero_weights(model)
        # if model_ema is not None:
        #     print("[Model Info.] Loading pre-trained model for Teacher-Model (EMA)!")
        #     model_ema = load_weights(model_ema, args)
        if args.modelEMA is not None:
            # model = load_weights(model, args)
            # model_ema = load_weights(model_ema, args)
            # print("[Model Info.] Loading pre-trained model for Teacher-Model (EMA):", args.resume)
            model_ema = copy.deepcopy(model)
            # model_ema = model_ema.cuda()

            # for param_q, param_k in zip(model.parameters(), model_ema.parameters()):
            #             param_k.data.copy_(param_q.detach().data)
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

    if args.taskcomponent in ['foundation_x_pretraining', 'foundation_x2_pretraining', 'foundation_x3_pretraining']: # detect_chestxdet_dataset
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
        print("[Information]  Learning Rate Backbone:", args.lr_backbone)
        print("[Information]  Learning Rate LocEnc:", args.lr_locEnc)
        print("[Information]  Learning Rate LocDec:", args.lr_locDec)
        print("[Information]  Num Workers:", args.num_workers)
        print("[Information]  Optimizer:", args.opt)
        print("[Information]  Output Dir:", args.output_dir)
        if args.foundationX is not None:
            print("[Information]  Foundation X Checkpoint:", args.foundationX)
            print("[Information]  Foundation X Checkpoint Model:", args.foundationXMODEL)
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
        if args.foundationX is not None:
            print("[Information]  Foundation X Checkpoint:", args.foundationX, file=log_writter_DETECTION)
            print("[Information]  Foundation X Checkpoint Model:", args.foundationXMODEL, file=log_writter_DETECTION)
        print("-------------", file=log_writter_DETECTION)

        
        if args.backbone_dir is not None:
            model = load_weights(model, args)
            # model = torch.nn.DataParallel(model)
            # model = nn.DataParallel(model, device_ids=[0, 1])
        if args.modelEMA is not None:
            model_ema = copy.deepcopy(model)
            # model_ema = torch.nn.DataParallel(model_ema)
            # model_ema = nn.DataParallel(model_ema, device_ids=[0, 1])

        try:
            data_loader_train, data_loader_val, sampler_train, dataset_val = dataloader_return(args)
            base_ds = get_coco_api_from_dataset(dataset_val)
        except:
            do_nothing = 1

        criterion_CLS = torch.nn.BCEWithLogitsLoss()

        if args.taskcomponent in ['foundation_x_pretraining'] and args.dataset_file == "foundation6_datasets":
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
                    export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_DiceLoss', 'Seg_ConsLoss'])
                    export_csvFile_train.to_csv(args.output_dir+'/export_csvFile_TRAIN.csv', index=False)

                export_csvFile = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Model', 'Task-Test', 'AUC', 'mAP40','mAP50','mAP50_95', 'DICE'])
                export_csvFile.to_csv(args.output_dir+'/export_csvFile.csv', index=False)

        if args.taskcomponent in ['foundation_x2_pretraining'] and args.dataset_file == "foundation6_datasets":
            train_loader_cls_TBX11k, test_loader_cls_TBX11k, train_loader_cls_NODE21, test_loader_cls_NODE21, train_loader_cls_ChestXDet, test_loader_cls_ChestXDet, \
            train_loader_cls_RSNApneumonia, test_loader_cls_RSNApneumonia, train_loader_cls_SIIMACRptx, test_loader_cls_SIIMACRptx, train_loader_cls_CANDIDptx, test_loader_cls_CANDIDptx, \
            train_loader_loc_TBX11k, test_loader_loc_TBX11k, dataset_val_loc_TBX11k, sampler_train_TBX11k, \
            train_loader_loc_Node21, test_loader_loc_Node21, dataset_val_loc_Node21, sampler_train_Node21, \
            train_loader_loc_CANDIDptx, test_loader_loc_CANDIDptx, dataset_val_loc_CANDIDptx, sampler_train_CANDIDptx, \
            train_loader_loc_ChestXDet, test_loader_loc_ChestXDet, dataset_val_loc_ChestXDet, sampler_train_ChestXDet, \
            train_loader_loc_RSNApneumonia, test_loader_loc_RSNApneumonia, dataset_val_loc_RSNApneumonia, sampler_train_RSNApneumonia, \
            train_loader_loc_SiimACR, test_loader_loc_SiimACR, dataset_val_loc_SiimACR, sampler_train_SiimACR, \
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
            # if args.eval == False:
            #     if args.resume == None:
            #         export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_DiceLoss', 'Seg_ConsLoss'])
            #         export_csvFile_train.to_csv(args.output_dir+'/export_csvFile_TRAIN.csv', index=False)

            if not os.path.exists(args.output_dir+'/export_csvFile_TRAIN.csv'):
                export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_DiceLoss', 'Seg_ConsLoss'])
                export_csvFile_train.to_csv(args.output_dir+'/export_csvFile_TRAIN.csv', index=False)
            if not os.path.exists(args.output_dir+'/export_csvFile.csv'):
                export_csvFile = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Model', 'Task-Test', 'AUC', 'mAP40','mAP50','mAP50_95', 'DICE'])
                export_csvFile.to_csv(args.output_dir+'/export_csvFile.csv', index=False)

        if args.taskcomponent in ['foundation_x3_pretraining'] and args.dataset_file == "foundation6Ark6_datasets":
            train_loader_cls_CheXpert, test_loader_cls_CheXpert, train_loader_cls_NIHChestXray14, test_loader_cls_NIHChestXray14, train_loader_cls_VinDRCXR, test_loader_cls_VinDRCXR, train_loader_cls_NIHShenzhen, test_loader_cls_NIHShenzhen, train_loader_cls_MIMICII, test_loader_cls_MIMICII, \
            train_loader_cls_TBX11k, test_loader_cls_TBX11k, train_loader_cls_NODE21, test_loader_cls_NODE21, train_loader_cls_ChestXDet, test_loader_cls_ChestXDet, \
            train_loader_cls_RSNApneumonia, test_loader_cls_RSNApneumonia, train_loader_cls_SIIMACRptx, test_loader_cls_SIIMACRptx, train_loader_cls_CANDIDptx, test_loader_cls_CANDIDptx, \
            train_loader_loc_TBX11k, test_loader_loc_TBX11k, dataset_val_loc_TBX11k, sampler_train_TBX11k, \
            train_loader_loc_Node21, test_loader_loc_Node21, dataset_val_loc_Node21, sampler_train_Node21, \
            train_loader_loc_CANDIDptx, test_loader_loc_CANDIDptx, dataset_val_loc_CANDIDptx, sampler_train_CANDIDptx, \
            train_loader_loc_ChestXDet, test_loader_loc_ChestXDet, dataset_val_loc_ChestXDet, sampler_train_ChestXDet, \
            train_loader_loc_RSNApneumonia, test_loader_loc_RSNApneumonia, dataset_val_loc_RSNApneumonia, sampler_train_RSNApneumonia, \
            train_loader_loc_SiimACR, test_loader_loc_SiimACR, dataset_val_loc_SiimACR, sampler_train_SiimACR, \
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

            if not os.path.exists(args.output_dir+'/export_csvFile_TRAIN.csv'):
                export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_DiceLoss', 'Seg_ConsLoss'])
                export_csvFile_train.to_csv(args.output_dir+'/export_csvFile_TRAIN.csv', index=False)
            if not os.path.exists(args.output_dir+'/export_csvFile.csv'):
                export_csvFile = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Model', 'Task-Test', 'AUC', 'mAP40','mAP50','mAP50_95', 'DICE'])
                export_csvFile.to_csv(args.output_dir+'/export_csvFile.csv', index=False)

        # BEST_STUDENT_CLS_AUC = -1
        # BEST_TEACHER_CLS_AUC = -1
        # BEST_STUDENT_LOC_mAP = -1
        # BEST_TEACHER_LOC_mAP = -1

        # if args.eval:
        #     if args.pretrain_model_path:
        #         model, model_ema, _, epoch = load_weights_resume(model, model_ema, optimizer, args)

        #         model.task_DetHead = 0
        #         test_stats, _, _ = evaluate(model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, DetHead=0, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None))
        #         log_stats = { **{f'test_Heart_{k}': v for k, v in test_stats.items()} }
        #         result_output_dir = args.output_dir + '/results_EvaluationOnly.txt'
        #         log_writer_detection = open(result_output_dir, 'a')
        #         formatted_stats_test = {f'test_{k}': v for k, v in test_stats.items()}
        #         log_writer_detection.write('Epoch: ' + str(epoch) + '\n')
        #         log_writer_detection.write('Pretrained Model: ' + args.pretrain_model_path + '\n')
        #         log_writer_detection.write("Dataset: " + args.dataset_file + '\n')
        #         log_writer_detection.write('-- Student Model Testing --' + '\n')
        #         for key, value in formatted_stats_test.items():
        #             log_writer_detection.write(f'{key}: {value}\n')
        #         log_writer_detection.write('\n')
        #         # str(100*value[1]), str(100*value[2]), str(100*value[0])   
        #         log_writer_detection.write('Localization mAP40: ' + str(100*value[1]) + '\n') # mAP40
        #         log_writer_detection.write('Localization mAP50: ' + str(100*value[2]) + '\n') # mAP50
        #         log_writer_detection.write('Localization mAP50:95: ' + str(100*value[0]) + '\n') # mAP50:95
        #         log_writer_detection.write('\n')
        #         log_writer_detection.write('\n')
        #         log_writer_detection.close()        

        #         model_ema.task_DetHead = 0
        #         test_stats, _, _ = evaluate(model_ema, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir, DetHead=0, wo_class_error=wo_class_error, args=args, logger=(logger if args.save_log else None))
        #         log_stats = { **{f'test_Heart_{k}': v for k, v in test_stats.items()} }
        #         result_output_dir = args.output_dir + '/results_EvaluationOnly.txt'
        #         log_writer_detection = open(result_output_dir, 'a')
        #         formatted_stats_test = {f'test_{k}': v for k, v in test_stats.items()}
        #         log_writer_detection.write('Epoch: ' + str(epoch) + '\n')
        #         log_writer_detection.write('Pretrained Model: ' + args.pretrain_model_path + '\n')
        #         log_writer_detection.write("Dataset: " + args.dataset_file + '\n')
        #         log_writer_detection.write('-- Teacher Model Testing --' + '\n')
        #         for key, value in formatted_stats_test.items():
        #             log_writer_detection.write(f'{key}: {value}\n')
        #         log_writer_detection.write('\n')
        #         # str(100*value[1]), str(100*value[2]), str(100*value[0])   
        #         log_writer_detection.write('Localization mAP40: ' + str(100*value[1]) + '\n') # mAP40
        #         log_writer_detection.write('Localization mAP50: ' + str(100*value[2]) + '\n') # mAP50
        #         log_writer_detection.write('Localization mAP50:95: ' + str(100*value[0]) + '\n') # mAP50:95
        #         log_writer_detection.write('\n')
        #         log_writer_detection.write('\n')
        #         log_writer_detection.close() 
        #         exit(0)
        #     else:
        #         print(" --- No pretrain_model_path given ---")
        #         exit(0)

    if args.taskcomponent in ['foundation_x3_FineTuning']: # detect_chestxdet_dataset
        logs_path = os.path.join(args.output_dir, "Logs")
        if not os.path.exists(logs_path):
            os.makedirs(logs_path)
        if os.path.exists(os.path.join(logs_path, "log.txt")):
            log_writter_DETECTION = open(os.path.join(logs_path, "log.txt"), 'a')
        else:
            log_writter_DETECTION = open(os.path.join(logs_path, "log.txt"), 'w')

        print()
        print("-----FINETUNING--------")
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
        if args.foundationX is not None:
            print("[Information]  Foundation X Checkpoint:", args.foundationX)
            print("[Information]  Foundation X Checkpoint Model:", args.foundationXMODEL)
        print("-------------")
        print()

        print("------FINETUNING-------", file=log_writter_DETECTION)
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
        if args.foundationX is not None:
            print("[Information]  Foundation X Checkpoint:", args.foundationX, file=log_writter_DETECTION)
            print("[Information]  Foundation X Checkpoint Model:", args.foundationXMODEL, file=log_writter_DETECTION)
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

        if args.taskcomponent in ['foundation_x3_FineTuning'] and args.dataset_file == "foundation6Ark6_datasets":
            train_loader, test_loader, dataset_val_ = dataloader_return(args)

            if 'locFT' in args.cyclictask:
                base_ds = get_coco_api_from_dataset(dataset_val_)
                del dataset_val_

            # for name, param in model.named_parameters(): ### WHY!!!!
            #     if ('decoder.1' in name) or ('decoder.2' in name) or ('segmentation' in name): ## Only Localization Encoder
            #         param.requires_grad = False
            model_ema = copy.deepcopy(model)

            if not os.path.exists(args.output_dir+'/export_csvFile_TRAIN.csv'):
                export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_DiceLoss', 'Seg_ConsLoss'])
                export_csvFile_train.to_csv(args.output_dir+'/export_csvFile_TRAIN.csv', index=False)
            if not os.path.exists(args.output_dir+'/export_csvFile.csv'):
                export_csvFile = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Model', 'Task-Test', 'AUC', 'mAP40','mAP50','mAP50_95', 'DICE'])
                export_csvFile.to_csv(args.output_dir+'/export_csvFile.csv', index=False)


















    ## T R A I N I N G   P H A S E ##
    print("Start training")
    start_time_ALL = time.time()
    best_map_holder = BestMetricHolder(use_ema=args.use_ema)
    args.start_epoch = 1

    if args.resume is not None:
        print("[CHECK] ---- Resuming Training -----")
        if args.opt == 'sgd' or args.opt == 'adamw':
            model, model_ema, optimizer, args.start_epoch = load_weights_resume(model, model_ema, optimizer, args)
            args.start_epoch = args.start_epoch + 1
        elif args.opt == 'adamw_and_sgd':
            model, model_ema, optimizer_adamw, optimizer_sgd, args.start_epoch = load_weights_resume2(model, model_ema, optimizer_adamw, optimizer_sgd, args)
            args.start_epoch = args.start_epoch + 1
        # exit(0)
    if args.foundationX is not None:
        print("[CHECK] ---- Loading Checkpoint from Foundation-X -----")
        model, model_ema = load_weights_foundationX(model, model_ema, optimizer, args, to_load=args.foundationXMODEL)
        # model, model_ema, optimizer = load_weights_foundationX(model, model_ema, optimizer, args, to_load=args.foundationXMODEL) # with Opt
        # args.lr_backbone = optimizer.param_groups[0]['lr']
        # args.lr_segmentor = optimizer.param_groups[1]['lr']
        # args.lr_locEnc = optimizer.param_groups[2]['lr']
        # args.lr_locDec = optimizer.param_groups[3]['lr']

    # for param in model.parameters():
    #     param.requires_grad = True

    # for name, param in model.named_parameters():
    #     if 'backbone' in name and 'segmentation_' not in name:
    #         param.requires_grad = False
    # old_value_layernorm = sum(model.module.backbone[0].layers[3].blocks[1].mlp.fc2.weight)

    if args.modelEMA == "True_Epoch":
        if args.taskcomponent == 'foundation_x_pretraining':
            Num_EPOCH_Iterative_Steps_MomentumSchduler = args.total_epochs
        elif args.taskcomponent == 'detect_segmentation_cyclic_v2' or args.taskcomponent == 'detect_segmentation_cyclic_v3':
            Num_EPOCH_Iterative_Steps_MomentumSchduler = args.total_epochs
        else: 
            Num_EPOCH_Iterative_Steps_MomentumSchduler = args.total_epochs
        momentum_schedule = cosine_scheduler(0.8, 1, Num_EPOCH_Iterative_Steps_MomentumSchduler, 1) # Localization EMA Update Epoch-wise
        momentum_schedule_SEG = cosine_scheduler(0.5, 1, Num_EPOCH_Iterative_Steps_MomentumSchduler, 1) # Segmentaiton EMA Update Epoch-wise
        # momentum_schedule = cosine_scheduler(0.9, 1, args.total_epochs, 1)
        print("[Model Info] EMA Epoch-wise Update.")
    else:
        momentum_schedule = None
        momentum_schedule_SEG = None

    # ## DEBUG ################:
    # print("Debug -- Layers Parameter 1 to ...")
    # value_temp_wb = 0
    # for (name_q, param_q), (name_k, param_k) in zip(model.named_parameters(), model_ema.named_parameters()):
    #     param_q.data.fill_(0)
    #     param_k.data.fill_(0)
    #     with open("/home/nuislam/projects/DINO_Detection/IntegratedModel_GitHub_V/zDebugTestFindings/Monitor_LayerWeights.txt", "a") as f:
    #         f.write(f"Layer from Student Model: {name_q} {param_q.detach().data.shape} | value {value_temp_wb} | {param_q.detach().data.sum()}\n")
    #         f.write(f"Layer from Teacher Model: {name_k} {param_k.data.shape} | value {value_temp_wb} | {param_k.data.sum()}\n")
    #         f.write("-" * 30 + "\n")
    # #     value_temp_wb += 1
    # if not os.path.exists(args.output_dir+'/export_csvFile_DEBUG_ema.csv'):
    #     export_csvFile = pd.DataFrame(columns=['Epoch', 'LayerName', 'Shape', 'Student', 'Teacher', 'Teacher_Final'])
    #     export_csvFile.to_csv(args.output_dir+'/export_csvFile_DEBUG_ema.csv', index=False)

    
    if args.opt != 'adamw_and_sgd':
        print("[Model Info.] Optimizer:", optimizer)
    else:
        print("[Model Info.] Optimizer AdamW:", optimizer_adamw)
        print("[Model Info.] Optimizer SGD:", optimizer_sgd)
    print("[Training Info.] Start_Epoch & Total_Epoch", args.start_epoch, args.total_epochs)
    print()
    MODEL_prev_parameters = {n: p.detach().cpu().numpy().tolist() for n, p in model.named_parameters() }
    if args.resume is not None:
        print(" ----------------------- ---- Resuming Training ----- ----------------------- ")
    




    # if isinstance(model, torch.nn.parallel.DistributedDataParallel):
    #     model._set_static_graph() # added by Nahid because of adding Classification & Segmentation component -- Forward/Backward pass issue
    for epoch in range(args.start_epoch, args.total_epochs):
        epoch_start_time = time.time()

        if args.taskcomponent == "foundation_x3_pretraining": ## Full Foundation-X training Ark6 Datasets + 6 ChestXray Datasets
            TaskHead_number = (epoch - 1) % 20
            store_best_performance_found = 0
            if args.modelEMA:
                model_ema.eval()
            # EMA Update Epoch-wise
            if args.modelEMA == "True_Epoch":
                if epoch >= Num_EPOCH_Iterative_Steps_MomentumSchduler:
                    coff = 0.5
                else:
                    coff = (momentum_schedule[epoch] - 0.9) * 5 # Epoch-wise
                coff = 0.5

            if args.opt == 'adamw' or args.opt == 'sgd':
                if TaskHead_number == 0:  ### LR update --->  Backbone -- Segmentor -- Localizer
                    lrBackbone_ = step_decay(epoch, args.lr_backbone, args.total_epochs, step_inc=100) ## step_inc was 20 --  100 (100/20=5) warm up upto 5 cycles
                    lrSegmentor_ = step_decay(epoch, args.lr_segmentor, args.total_epochs, step_inc=100) ## step_inc was 20 -- 100 (100/20=5) warm up upto 5 cycles
                    lrLocalizerEnc_ = step_decay(epoch, args.lr_locEnc, args.total_epochs, step_inc=100) ## step_inc was 20 -- 100 (100/20=5) warm up upto 5 cycles
                    lrLocalizerDec_ = step_decay(epoch, args.lr_locDec, args.total_epochs, step_inc=100) ## step_inc was 20 -- 100 (100/20=5) warm up upto 5 cycles
                    if len(optimizer.param_groups) == 2 or len(optimizer.param_groups) == 3 or len(optimizer.param_groups) == 4:
                        optimizer.param_groups[0]['lr'] = lrBackbone_
                        optimizer.param_groups[1]['lr'] = lrSegmentor_
                        optimizer.param_groups[2]['lr'] = lrLocalizerEnc_
                        optimizer.param_groups[3]['lr'] = lrLocalizerDec_
                        print('Epoch{} - TaskHead{} - learning_rateBackbone [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']))
                        print('Epoch{} - TaskHead{} - learning_rateSegmentor [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[1]['lr']))
                        print('Epoch{} - TaskHead{} - learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[2]['lr']))
                        print('Epoch{} - TaskHead{} - learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[3]['lr']))
                        # print('Epoch{} - TaskHead{} - learning_rateREST [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer['lr']))

                        print('Epoch{} - TaskHead{} - learning_rateBackbone [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_DETECTION)
                        print('Epoch{} - TaskHead{} - learning_rateSegmentor [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[1]['lr']), file=log_writter_DETECTION)
                        print('Epoch{} - TaskHead{} - learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[2]['lr']), file=log_writter_DETECTION)
                        print('Epoch{} - TaskHead{} - learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[3]['lr']), file=log_writter_DETECTION)
                    else:
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_
                        print('Epoch{} - TaskHead{} - learning_rate [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_DETECTION)
                else:
                    if len(optimizer.param_groups) == 2 or len(optimizer.param_groups) == 3 or len(optimizer.param_groups) == 4:
                        print('Epoch{} - TaskHead{} - learning_rateBackbone: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_DETECTION)
                        print('Epoch{} - TaskHead{} - learning_rateSegmentor: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[1]['lr']), file=log_writter_DETECTION)
                        print('Epoch{} - TaskHead{} - learning_rateLocalizer: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[2]['lr']), file=log_writter_DETECTION)
                        print('Epoch{} - TaskHead{} - learning_rateLocalizer: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[3]['lr']), file=log_writter_DETECTION)
                    else:
                        print('Epoch{} - TaskHead{} - learning_rate: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_DETECTION)
            elif args.opt == 'adamw_and_sgd':
                if TaskHead_number == 0:  ### LR update --->  Backbone -- Segmentor -- Localizer
                    lrBackbone_ = step_decay(epoch, args.lr_backbone, args.total_epochs, step_inc=20)
                    lrBackbone2_ = step_decay(epoch, args.lr_backbone2, args.total_epochs, step_inc=20)
                    lrSegmentor_ = step_decay(epoch, args.lr_segmentor, args.total_epochs, step_inc=20)
                    lrLocalizerEnc_ = step_decay(epoch, args.lr_locEnc, args.total_epochs, step_inc=20)
                    lrLocalizerDec_ = step_decay(epoch, args.lr_locDec, args.total_epochs, step_inc=20)

                    optimizer_sgd.param_groups[0]['lr'] = lrBackbone2_
                    optimizer_sgd.param_groups[1]['lr'] = lrSegmentor_
                    optimizer_sgd.param_groups[2]['lr'] = lrLocalizerEnc_
                    optimizer_sgd.param_groups[3]['lr'] = lrLocalizerDec_   

                    optimizer_adamw.param_groups[0]['lr'] = lrBackbone_
                    optimizer_adamw.param_groups[1]['lr'] = lrSegmentor_
                    optimizer_adamw.param_groups[2]['lr'] = lrLocalizerEnc_
                    optimizer_adamw.param_groups[3]['lr'] = lrLocalizerDec_              
                    print('Epoch{} - TaskHead{} - Opt_SGD learning_rateBackbone [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer_sgd.param_groups[0]['lr']), file=log_writter_DETECTION)
                    print('Epoch{} - TaskHead{} - Opt_SGD learning_rateSegmentor [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer_sgd.param_groups[1]['lr']), file=log_writter_DETECTION)
                    print('Epoch{} - TaskHead{} - Opt_SGD learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer_sgd.param_groups[2]['lr']), file=log_writter_DETECTION)
                    print('Epoch{} - TaskHead{} - Opt_SGD learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer_sgd.param_groups[3]['lr']), file=log_writter_DETECTION)

                    print('Epoch{} - TaskHead{} - Opt_AdamW learning_rateBackbone [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer_adamw.param_groups[0]['lr']), file=log_writter_DETECTION)
                    print('Epoch{} - TaskHead{} - Opt_AdamW learning_rateSegmentor [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer_adamw.param_groups[1]['lr']), file=log_writter_DETECTION)
                    print('Epoch{} - TaskHead{} - Opt_AdamW learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer_adamw.param_groups[2]['lr']), file=log_writter_DETECTION)
                    print('Epoch{} - TaskHead{} - Opt_AdamW learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer_adamw.param_groups[3]['lr']), file=log_writter_DETECTION)
                else:
                    print('Epoch{} - TaskHead{} - Opt_SGD learning_rateBackbone: {:.20f}'.format(epoch, TaskHead_number, optimizer_sgd.param_groups[0]['lr']), file=log_writter_DETECTION)
                    print('Epoch{} - TaskHead{} - Opt_SGD learning_rateSegmentor: {:.20f}'.format(epoch, TaskHead_number, optimizer_sgd.param_groups[1]['lr']), file=log_writter_DETECTION)
                    print('Epoch{} - TaskHead{} - Opt_SGD learning_rateLocalizer: {:.20f}'.format(epoch, TaskHead_number, optimizer_sgd.param_groups[2]['lr']), file=log_writter_DETECTION)
                    print('Epoch{} - TaskHead{} - Opt_SGD learning_rateLocalizer: {:.20f}'.format(epoch, TaskHead_number, optimizer_sgd.param_groups[3]['lr']), file=log_writter_DETECTION)

                    print('Epoch{} - TaskHead{} - Opt_AdamW learning_rateBackbone: {:.20f}'.format(epoch, TaskHead_number, optimizer_adamw.param_groups[0]['lr']), file=log_writter_DETECTION)
                    print('Epoch{} - TaskHead{} - Opt_AdamW learning_rateSegmentor: {:.20f}'.format(epoch, TaskHead_number, optimizer_adamw.param_groups[1]['lr']), file=log_writter_DETECTION)
                    print('Epoch{} - TaskHead{} - Opt_AdamW learning_rateLocalizer: {:.20f}'.format(epoch, TaskHead_number, optimizer_adamw.param_groups[2]['lr']), file=log_writter_DETECTION)
                    print('Epoch{} - TaskHead{} - Opt_AdamW learning_rateLocalizer: {:.20f}'.format(epoch, TaskHead_number, optimizer_adamw.param_groups[3]['lr']), file=log_writter_DETECTION)


            if args.modelEMA is None:
                coff = None
                momentum_schedule = None
                criterionMSE = None
                model_ema = None

            for param in model.parameters():
                param.requires_grad = True

            start_time = time.time()
            print('-- Epoch {} TaskHead {} --'.format(epoch, TaskHead_number), file=log_writter_DETECTION)
            task_todo = "None"

            #### Training Phase ####
            ## TaskHead_number  0 = CheXpert Classification
            ## TaskHead_number  1 = NIHChestXray14 Classification
            ## TaskHead_number  2 = VinDRCXR Classification
            ## TaskHead_number  3 = NIHShenzhen Classification
            ## TaskHead_number  4 = MIMICII Classification

            ## TaskHead_number  5 = TBX11k Classification  - ready
            ## TaskHead_number  6 = TBX11k Localization  - ready
            ## TaskHead_number  7 = NODE21 Classification  - ready
            ## TaskHead_number  8 = NODE21 Localization  - ready
            ## TaskHead_number  9 = CANDID-PTX Classification  - ready
            ## TaskHead_number  10 = CANDID-PTX Localization  - ready
            ## TaskHead_number  11 = CANDID-PTX Segmentation  - ready
            ## TaskHead_number  12 = RSNA_Pneumonia Classification  - ready
            ## TaskHead_number  13 = RSNA_Pneumonia Localization  - ready
            ## TaskHead_number  14 = ChestX-Det Classification  - ready
            ## TaskHead_number  15 = ChestX-Det Localization  - ready
            ## TaskHead_number  16 = ChestX-Det Segmentation  - ready
            ## TaskHead_number  17 = SIIM-ACR-Pneumothorax Classification  - ready
            ## TaskHead_number  18 = SIIM-ACR-Pneumothorax Localization  - ready
            ## TaskHead_number  19 = SIIM-ACR-Pneumothorax Segmentation  - ready


            if args.cyclictask == 'loc' or args.cyclictask == 'loc_TESTloc_tbx11kLOC_node21LOC_candidptxLOC_rsnapneumoniaLOC_chestxdetLOC_siimacrLOC': # 
                if TaskHead_number not in [6,8,10,13,15,18]: 
                    continue
            elif args.cyclictask == 'seg' or args.cyclictask == 'seg_TESTseg_candidptxSEG_chestxdetSEG_siimacrSEG':  #  args.cyclictask == 'TESTseg_candidptxSEG' or args.cyclictask == 'TESTseg_chestxdetSEG' or args.cyclictask == 'TESTseg_siimacrSEG':
                if TaskHead_number not in [11,16,19]:
                    continue
            elif args.cyclictask == 'cls' or args.cyclictask == 'cls_TESTcls_chexpertCLS_nihchestxray14CLS_vindrcxrCLS_nihshenzenCLS_mimic2CLS_tbx11kCLS_node21CLS_candidptxCLS_rsnapneumoniaCLS_chestxdetCLS_siimacrCLS':
                if TaskHead_number not in [0,1,2,3,4,5,7,9,12,14,17]: 
                # if TaskHead_number not in [3]: 
                    continue
            elif args.cyclictask == 'cls_seg_TESTcls_chexpertCLS_nihchestxray14CLS_vindrcxrCLS_nihshenzenCLS_mimic2CLS_tbx11kCLS_node21CLS_candidptxCLS_rsnapneumoniaCLS_chestxdetCLS_siimacrCLS_TESTseg_candidptxSEG_chestxdetSEG_siimacrSEG':
                if TaskHead_number not in [0,1,2,3,4,5,7,9,11,12,14,16,17,19]: 
                    continue
            elif args.cyclictask == 'cls_loc_TESTcls_chexpertCLS_nihchestxray14CLS_vindrcxrCLS_nihshenzenCLS_mimic2CLS_tbx11kCLS_node21CLS_candidptxCLS_rsnapneumoniaCLS_chestxdetCLS_siimacrCLS_TESTloc_tbx11kLOC_node21LOC_candidptxLOC_rsnapneumoniaLOC_chestxdetLOC_siimacrLOC':
                if TaskHead_number not in [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15,17,18]: 
                    continue
            elif args.cyclictask == 'cls_loc_seg' or args.cyclictask == 'cls_loc_seg_TESTcls_chexpertCLS_nihchestxray14CLS_vindrcxrCLS_nihshenzenCLS_mimic2CLS_tbx11kCLS_node21CLS_candidptxCLS_rsnapneumoniaCLS_chestxdetCLS_siimacrCLS_TESTloc_tbx11kLOC_node21LOC_candidptxLOC_rsnapneumoniaLOC_chestxdetLOC_siimacrLOC_TESTseg_candidptxSEG_chestxdetSEG_siimacrSEG':
                check = 0 ## Ignore nothing
            # # CLASSIFICATION FINE-TUNING
            elif args.cyclictask == 'clsFT_TESTcls_chexpertCLS': ## CheXpert Classification Finetuning
                if TaskHead_number not in [0]: 
                    continue
            elif args.cyclictask == 'clsFT_TESTcls_nihchestxray14CLS': ## NIHChestXray14 Classification Finetuning
                if TaskHead_number not in [1]: 
                    continue
            elif args.cyclictask == 'clsFT_TESTcls_vindrcxrCLS': ## VinDRCXR Classification Finetuning
                if TaskHead_number not in [2]: 
                    continue
            elif args.cyclictask == 'clsFT_TESTcls_nihshenzenCLS': ## NIHShenzhen Classification Finetuning
                if TaskHead_number not in [3]: 
                    continue
            elif args.cyclictask == 'clsFT_TESTcls_mimic2CLS': ## MIMICII Classification Finetuning
                if TaskHead_number not in [4]: 
                    continue
            elif args.cyclictask == 'clsFT_TESTcls_tbx11kCLS': ## TBX11k Classification Finetuning
                if TaskHead_number not in [5]: 
                    continue
            elif args.cyclictask == 'clsFT_TESTcls_node21CLS': ## NODE21 Classification Finetuning
                if TaskHead_number not in [7]: 
                    continue
            elif args.cyclictask == 'clsFT_TESTcls_candidptxCLS': ## CANDID-PTX Classification Finetuning
                if TaskHead_number not in [9]: 
                    continue
            elif args.cyclictask == 'clsFT_TESTcls_rsnapneumoniaCLS': ## RSNA Pneumonia Classification Finetuning
                if TaskHead_number not in [12]: 
                    continue
            elif args.cyclictask == 'clsFT_TESTcls_chestxdetCLS': ## ChestX-Det Classification Finetuning
                if TaskHead_number not in [14]: 
                    continue
            elif args.cyclictask == 'clsFT_TESTcls_siimacrCLS': ## SIIM-ACR Classification Finetuning
                if TaskHead_number not in [17]: 
                    continue
            # # LOCALIZATION FINE-TUNING 
            elif args.cyclictask == 'locFT_TESTloc_tbx11kLOC': ## TBX11k Localization Finetuning
                if TaskHead_number not in [6]: 
                    continue
            elif args.cyclictask == 'locFT_TESTloc_node21LOC': ## NODE21 Localization Finetuning
                if TaskHead_number not in [8]: 
                    continue
            elif args.cyclictask == 'locFT_TESTloc_candidptxLOC': ## CANDID-PTX Localization Finetuning
                if TaskHead_number not in [10]: 
                    continue
            elif args.cyclictask == 'locFT_TESTloc_rsnapneumoniaLOC': ## RSNA Pneumonia Localization Finetuning
                if TaskHead_number not in [13]: 
                    continue
            elif args.cyclictask == 'locFT_TESTloc_chestxdetLOC': ## ChestX-Det Localization Finetuning
                if TaskHead_number not in [15]: 
                    continue
            elif args.cyclictask == 'locFT_TESTloc_siimacrLOC': ## SIIM-ACR Localization Finetuning
                if TaskHead_number not in [18]: 
                    continue
            elif args.cyclictask == 'locFT_TESTloc_objects365LOC': ## Objects365 Localization Finetuning
                if TaskHead_number not in [6]: 
                    continue
            # # SEGMENTATION FINE-TUNING
            elif args.cyclictask == 'segFT_TESTseg_candidptxSEG': ## CANDID-PTX Segmentation Finetuning
                if TaskHead_number not in [11]: 
                    continue
            elif args.cyclictask == 'segFT_TESTseg_chestxdetSEG': ## ChestX-Det Segmentation Finetuning
                if TaskHead_number not in [16]: 
                    continue
            elif args.cyclictask == 'segFT_TESTseg_siimacrSEG': ## SIIM-ACR Segmentation Finetuning
                if TaskHead_number not in [19]: 
                    continue


            # else:
            #     print("[W A R N N I N G] Cyclic Task Error! ", args.cyclictask)
            #     exit(0)

            # model = Freeze_Backbone_and_Localization_Encoder(model)
            # model = unFreeze_Backbone_and_Localization_Encoder(model)

            if args.debugOnlyTest == False:
                ### ------ CheXpert ------ ###
                if TaskHead_number == 0: # Train for CheXpert Classification
                    print()
                    print("Classification_CheXpert_Train")
                    task_todo = "Classification_CheXpert_Train"
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_sgd
                    train_lossAVG, model_ema = train_CLASSIFICATION(train_loader_cls_CheXpert, model, criterion_CLS, optimizer, epoch, args, log_writter_DETECTION, head_number=6, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE, task_cls_type='nonBinary')
                    print( "Epoch {:04d}: Train Classification CheXpert Loss {:.5f} ".format(epoch, train_lossAVG) )
                    print( "Epoch {:04d}: Train Classification CheXpert Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_DETECTION )
                    fields=[epoch, 'CheXpert', task_todo, str(train_lossAVG), '-', '-']
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: 
                        writer = csv.writer(f)
                        writer.writerow(fields)

                ### ------ NIHChestXray14 ------ ###        
                if TaskHead_number == 1: # Train for NIH ChestX-ray14 Classification
                    print()
                    print("Classification_NIH_Train")
                    task_todo = "Classification_NIH_Train"
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_sgd
                    train_lossAVG, model_ema = train_CLASSIFICATION(train_loader_cls_NIHChestXray14, model, criterion_CLS, optimizer, epoch, args, log_writter_DETECTION, head_number=7, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE, task_cls_type='nonBinary')
                    print( "Epoch {:04d}: Train Classification NIHChestXray14 Loss {:.5f} ".format(epoch, train_lossAVG) )
                    print( "Epoch {:04d}: Train Classification NIHChestXray14 Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_DETECTION )
                    fields=[epoch, 'NIHChestXray14', task_todo, str(train_lossAVG), '-', '-']
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: 
                        writer = csv.writer(f)
                        writer.writerow(fields)

                ### ------ VinDRCXR ------ ###        
                if TaskHead_number == 2: # Train for VinDR-CXR Classification
                    print()
                    print("Classification_VinDRCXR_Train")
                    task_todo = "Classification_VinDRCXR_Train"
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_sgd
                    train_lossAVG, model_ema = train_CLASSIFICATION(train_loader_cls_VinDRCXR, model, criterion_CLS, optimizer, epoch, args, log_writter_DETECTION, head_number=8, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE, task_cls_type='nonBinary')
                    print( "Epoch {:04d}: Train Classification VinDRCXR Loss {:.5f} ".format(epoch, train_lossAVG) )
                    print( "Epoch {:04d}: Train Classification VinDRCXR Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_DETECTION )
                    fields=[epoch, 'VinDRCXR', task_todo, str(train_lossAVG), '-', '-']
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: 
                        writer = csv.writer(f)
                        writer.writerow(fields)

                ### ------ NIHShenzhen ------ ###        
                if TaskHead_number == 3: # Train for NIH Shenzhen CXR Classification
                    print()
                    print("Classification_NIHShenzhen_Train")
                    task_todo = "Classification_NIHShenzhen_Train"
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_sgd
                    train_lossAVG, model_ema = train_CLASSIFICATION(train_loader_cls_NIHShenzhen, model, criterion_CLS, optimizer, epoch, args, log_writter_DETECTION, head_number=9, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE, task_cls_type='nonBinary') ## Binary but for the dataloader setup
                    print( "Epoch {:04d}: Train Classification NIHShenzhen Loss {:.5f} ".format(epoch, train_lossAVG) )
                    print( "Epoch {:04d}: Train Classification NIHShenzhen Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_DETECTION )
                    fields=[epoch, 'NIHShenzhen', task_todo, str(train_lossAVG), '-', '-']
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: 
                        writer = csv.writer(f)
                        writer.writerow(fields)

                ### ------ MIMIC-II ------ ###        
                if TaskHead_number == 4: # Train for MIMIC-II Classification
                    print()
                    print("Classification_MIMICII_Train")
                    task_todo = "Classification_MIMICII_Train"
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_sgd
                    train_lossAVG, model_ema = train_CLASSIFICATION(train_loader_cls_MIMICII, model, criterion_CLS, optimizer, epoch, args, log_writter_DETECTION, head_number=10, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE, task_cls_type='nonBinary')
                    print( "Epoch {:04d}: Train Classification MIMICII Loss {:.5f} ".format(epoch, train_lossAVG) )
                    print( "Epoch {:04d}: Train Classification MIMICII Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_DETECTION )
                    fields=[epoch, 'MIMICII', task_todo, str(train_lossAVG), '-', '-']
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: 
                        writer = csv.writer(f)
                        writer.writerow(fields)


                ### ------ TBX11K ------ ###
                if TaskHead_number == 5: # Train for TBX11k Classification
                    print()
                    print("Classification_TBX11k_Train")
                    task_todo = "Classification_TBX11k_Train"
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_sgd
                    train_lossAVG, model_ema = train_CLASSIFICATION(train_loader_cls_TBX11k, model, criterion_CLS, optimizer, epoch, args, log_writter_DETECTION, head_number=0, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                    print( "Epoch {:04d}: Train Classification TBX11k Loss {:.5f} ".format(epoch, train_lossAVG) )
                    print( "Epoch {:04d}: Train Classification TBX11k Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_DETECTION )
                    fields=[epoch, 'TBX11k', task_todo, str(train_lossAVG), '-', '-'] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)

                if TaskHead_number == 6: # Train for TBX11k Localization
                    if 'locFT' not in args.cyclictask:
                        print()
                        print("Localization_TBX11k_A_Train") # Train A Set
                        task_todo = "Localization_TBX11k_A_Train"
                        model = Freeze_Backbone_and_Localization_Encoder(model)
                        model.task_DetHead = 0
                        if args.modelEMA:
                            model_ema.task_DetHead = 0
                        if args.opt == 'adamw_and_sgd':
                            optimizer = optimizer_adamw
                        train_stats, model_ema = train_one_epoch(
                            model, criterion, train_loader_loc_TBX11k, optimizer, device, epoch,
                            args.clip_max_norm, wo_class_error=wo_class_error, DetHead=0, train_type='F', lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                        log_stats = { **{f'train_TBX11k_A_{k}': v for k, v in train_stats.items()} }
                        result_output_dir = args.output_dir + '/results.txt'
                        log_writer_detection = open(result_output_dir, 'a')
                        log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: TBX11k_A " + '\n')
                        log_writer_detection.write('-- Training --' + '\n')
                        formatted_stats_train = {f'train_TBX11k_A_{k}': v for k, v in train_stats.items()}
                        for key, value in formatted_stats_train.items():
                            log_writer_detection.write(f'{key}: {value}\n')
                            if key == 'train_TBX11k_A_loss':
                                loss_localization_temp = value
                        log_writer_detection.write('\n')
                        log_writer_detection.close()
                        fields=[epoch, 'TBX11k', task_todo, '-', str(loss_localization_temp), '-'] # AUC_SliceLevel_Res
                        with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                            writer = csv.writer(f)
                            writer.writerow(fields)

                    print()
                    print("Localization_TBX11k_B_Train") # Train B Set
                    task_todo = "Localization_TBX11k_B_Train"
                    model = unFreeze_Backbone_and_Localization_Encoder(model)
                    model.task_DetHead = 0
                    if args.modelEMA:
                        model_ema.task_DetHead = 0
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_adamw
                    train_stats, model_ema = train_one_epoch(
                        model, criterion, train_loader_loc_TBX11k, optimizer, device, epoch,
                        args.clip_max_norm, wo_class_error=wo_class_error, DetHead=0,  train_type='uF', lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                    log_stats = { **{f'train_TBX11k_B_{k}': v for k, v in train_stats.items()} }
                    result_output_dir = args.output_dir + '/results.txt'
                    log_writer_detection = open(result_output_dir, 'a')
                    log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: TBX11k_B " + '\n')
                    log_writer_detection.write('-- Training --' + '\n')
                    formatted_stats_train = {f'train_TBX11k_B_{k}': v for k, v in train_stats.items()}
                    for key, value in formatted_stats_train.items():
                        log_writer_detection.write(f'{key}: {value}\n')
                        if key == 'train_TBX11k_B_loss':
                            loss_localization_temp = value
                    log_writer_detection.write('\n')
                    log_writer_detection.close()
                    fields=[epoch, 'TBX11k', task_todo, '-', str(loss_localization_temp), '-'] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)

                if args.cyclictask == 'locFT_TESTloc_objects365LOC' and TaskHead_number == 6:
                    if 'locFT' not in args.cyclictask:
                        print()
                        print("Localization_Objects365_A_Train") # Train A Set
                        task_todo = "Localization_Objects365_A_Train"
                        model = Freeze_Backbone_and_Localization_Encoder(model)
                        model.task_DetHead = 0
                        if args.modelEMA:
                            model_ema.task_DetHead = 0
                        if args.opt == 'adamw_and_sgd':
                            optimizer = optimizer_adamw
                        train_stats, model_ema = train_one_epoch(
                            model, criterion, train_loader_loc_TBX11k, optimizer, device, epoch,
                            args.clip_max_norm, wo_class_error=wo_class_error, DetHead=0, train_type='F', lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                        log_stats = { **{f'train_Objects365_A_{k}': v for k, v in train_stats.items()} }
                        result_output_dir = args.output_dir + '/results.txt'
                        log_writer_detection = open(result_output_dir, 'a')
                        log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: Objects365_A " + '\n')
                        log_writer_detection.write('-- Training --' + '\n')
                        formatted_stats_train = {f'train_Objects365_A_{k}': v for k, v in train_stats.items()}
                        for key, value in formatted_stats_train.items():
                            log_writer_detection.write(f'{key}: {value}\n')
                            if key == 'train_Objects365_A_loss':
                                loss_localization_temp = value
                        log_writer_detection.write('\n')
                        log_writer_detection.close()
                        fields=[epoch, 'Objects365', task_todo, '-', str(loss_localization_temp), '-'] # AUC_SliceLevel_Res
                        with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                            writer = csv.writer(f)
                            writer.writerow(fields)

                    print()
                    print("Localization_Objects365_B_Train") # Train B Set
                    task_todo = "Localization_Objects365_B_Train"
                    model = unFreeze_Backbone_and_Localization_Encoder(model)
                    model.task_DetHead = 0
                    if args.modelEMA:
                        model_ema.task_DetHead = 0
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_adamw
                    train_stats, model_ema = train_one_epoch(
                        model, criterion, train_loader_loc_TBX11k, optimizer, device, epoch,
                        args.clip_max_norm, wo_class_error=wo_class_error, DetHead=0,  train_type='uF', lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                    log_stats = { **{f'train_Objects365_B_{k}': v for k, v in train_stats.items()} }
                    result_output_dir = args.output_dir + '/results.txt'
                    log_writer_detection = open(result_output_dir, 'a')
                    log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: Objects365_B " + '\n')
                    log_writer_detection.write('-- Training --' + '\n')
                    formatted_stats_train = {f'train_Objects365_B_{k}': v for k, v in train_stats.items()}
                    for key, value in formatted_stats_train.items():
                        log_writer_detection.write(f'{key}: {value}\n')
                        if key == 'train_Objects365_B_loss':
                            loss_localization_temp = value
                    log_writer_detection.write('\n')
                    log_writer_detection.close()
                    fields=[epoch, 'Objects365', task_todo, '-', str(loss_localization_temp), '-'] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)


                ### ------ NODE21 ------ ###
                if TaskHead_number == 7: # Train for NODE21 Classification 
                    print()
                    print("Classification_NODE21_Train")
                    task_todo = "Classification_NODE21_Train"
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_sgd
                    train_lossAVG, model_ema = train_CLASSIFICATION(train_loader_cls_NODE21, model, criterion_CLS, optimizer, epoch, args, log_writter_DETECTION, head_number=1, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                    print( "Epoch {:04d}: Train Classification NODE21 Loss {:.5f} ".format(epoch, train_lossAVG) )
                    print( "Epoch {:04d}: Train Classification NODE21 Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_DETECTION )
                    fields=[epoch, 'NODE21', task_todo, str(train_lossAVG), '-', '-'] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)

                if TaskHead_number == 8: # Train for NODE21 Localization
                    if 'locFT' not in args.cyclictask:
                        print()
                        print("Localization_NODE21_A_Train") # Train A Set
                        task_todo = "Localization_NODE21_A_Train"
                        model = Freeze_Backbone_and_Localization_Encoder(model)
                        model.task_DetHead = 1
                        if args.modelEMA:
                            model_ema.task_DetHead = 1
                        if args.opt == 'adamw_and_sgd':
                            optimizer = optimizer_adamw
                        train_stats, model_ema = train_one_epoch(
                            model, criterion, train_loader_loc_Node21, optimizer, device, epoch,
                            args.clip_max_norm, wo_class_error=wo_class_error, DetHead=1, train_type='F', lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                        log_stats = { **{f'train_NODE21_A_{k}': v for k, v in train_stats.items()} }
                        result_output_dir = args.output_dir + '/results.txt'
                        log_writer_detection = open(result_output_dir, 'a')
                        log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: NODE21_A " + '\n')
                        log_writer_detection.write('-- Training --' + '\n')
                        formatted_stats_train = {f'train_NODE21_A_{k}': v for k, v in train_stats.items()}
                        for key, value in formatted_stats_train.items():
                            log_writer_detection.write(f'{key}: {value}\n')
                            if key == 'train_NODE21_A_loss':
                                loss_localization_temp = value
                        log_writer_detection.write('\n')
                        log_writer_detection.close()
                        fields=[epoch, 'NODE21', task_todo, '-', str(loss_localization_temp), '-'] # AUC_SliceLevel_Res
                        with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                            writer = csv.writer(f)
                            writer.writerow(fields)

                    print()
                    print("Localization_NODE21_B_Train") # Train B Set
                    task_todo = "Localization_NODE21_B_Train"
                    model = unFreeze_Backbone_and_Localization_Encoder(model)
                    model.task_DetHead = 1
                    if args.modelEMA:
                        model_ema.task_DetHead = 1
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_adamw
                    train_stats, model_ema = train_one_epoch(
                        model, criterion, train_loader_loc_Node21, optimizer, device, epoch,
                        args.clip_max_norm, wo_class_error=wo_class_error, DetHead=1, train_type='uF', lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                    log_stats = { **{f'train_NODE21_B_{k}': v for k, v in train_stats.items()} }
                    result_output_dir = args.output_dir + '/results.txt'
                    log_writer_detection = open(result_output_dir, 'a')
                    log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: NODE21_B " + '\n')
                    log_writer_detection.write('-- Training --' + '\n')
                    formatted_stats_train = {f'train_NODE21_B_{k}': v for k, v in train_stats.items()}
                    for key, value in formatted_stats_train.items():
                        log_writer_detection.write(f'{key}: {value}\n')
                        if key == 'train_NODE21_B_loss':
                            loss_localization_temp = value
                    log_writer_detection.write('\n')
                    log_writer_detection.close()
                    fields=[epoch, 'NODE21', task_todo, '-', str(loss_localization_temp), '-'] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)

                ### ------ CANDID-PTX ------ ###
                if TaskHead_number == 9: # Train for CANDID-PTX Classification
                    print()
                    print("Classification_CANDIDPTX_Train")
                    task_todo = "Classification_CANDIDPTX_Train"
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_sgd
                    train_lossAVG, model_ema = train_CLASSIFICATION(train_loader_cls_CANDIDptx, model, criterion_CLS, optimizer, epoch, args, log_writter_DETECTION, head_number=2, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE, task_cls_type='nonBinary')
                    print( "Epoch {:04d}: Train Classification CANDIDPTX Loss {:.5f} ".format(epoch, train_lossAVG) )
                    print( "Epoch {:04d}: Train Classification CANDIDPTX Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_DETECTION )
                    fields=[epoch, 'CANDID-PTX', task_todo, str(train_lossAVG), '-', '-'] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)

                if TaskHead_number == 10: # Train for CANDID-PTX Localization
                    if 'locFT' not in args.cyclictask:
                        print()
                        print("Localization_CANDIDPTX_A_Train") # Train A Set
                        task_todo = "Localization_CANDIDPTX_A_Train"
                        model = Freeze_Backbone_and_Localization_Encoder(model)
                        model.task_DetHead = 2
                        if args.modelEMA:
                            model_ema.task_DetHead = 2
                        if args.opt == 'adamw_and_sgd':
                            optimizer = optimizer_adamw
                        train_stats, model_ema = train_one_epoch(
                            model, criterion, train_loader_loc_CANDIDptx, optimizer, device, epoch,
                            args.clip_max_norm, wo_class_error=wo_class_error, DetHead=2, train_type='F', lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                        log_stats = { **{f'train_CANDIDPTX_A_{k}': v for k, v in train_stats.items()} }
                        result_output_dir = args.output_dir + '/results.txt'
                        log_writer_detection = open(result_output_dir, 'a')
                        log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: CANDIDPTX_A " + '\n')
                        log_writer_detection.write('-- Training --' + '\n')
                        formatted_stats_train = {f'train_CANDIDPTX_A_{k}': v for k, v in train_stats.items()}
                        for key, value in formatted_stats_train.items():
                            log_writer_detection.write(f'{key}: {value}\n')
                            if key == 'train_CANDIDPTX_A_loss':
                                loss_localization_temp = value
                        log_writer_detection.write('\n')
                        log_writer_detection.close()
                        fields=[epoch, 'CANDID-PTX', task_todo, '-', str(loss_localization_temp), '-'] # AUC_SliceLevel_Res
                        with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                            writer = csv.writer(f)
                            writer.writerow(fields)

                    print()
                    print("Localization_CANDIDPTX_B_Train") # Train B Set
                    task_todo = "Localization_CANDIDPTX_B_Train"
                    model = unFreeze_Backbone_and_Localization_Encoder(model)
                    model.task_DetHead = 2
                    if args.modelEMA:
                        model_ema.task_DetHead = 2
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_adamw
                    train_stats, model_ema = train_one_epoch(
                        model, criterion, train_loader_loc_CANDIDptx, optimizer, device, epoch,
                        args.clip_max_norm, wo_class_error=wo_class_error, DetHead=2, train_type='uF', lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                    log_stats = { **{f'train_CANDIDPTX_B_{k}': v for k, v in train_stats.items()} }
                    result_output_dir = args.output_dir + '/results.txt'
                    log_writer_detection = open(result_output_dir, 'a')
                    log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: CANDIDPTX_B " + '\n')
                    log_writer_detection.write('-- Training --' + '\n')
                    formatted_stats_train = {f'train_CANDIDPTX_B_{k}': v for k, v in train_stats.items()}
                    for key, value in formatted_stats_train.items():
                        log_writer_detection.write(f'{key}: {value}\n')
                        if key == 'train_CANDIDPTX_B_loss':
                            loss_localization_temp = value
                    log_writer_detection.write('\n')
                    log_writer_detection.close()
                    fields=[epoch, 'CANDID-PTX', task_todo, '-', str(loss_localization_temp), '-'] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)

                if TaskHead_number == 11: # Train for CANDID-PTX Segmentation
                    if 'segFT' not in args.cyclictask:
                        print()
                        print("Segmentation_CANDIDPTX_A_Train")
                        task_todo = "Segmentation_CANDIDPTX_A_Train"
                        model.task_DetHead = 2
                        if args.modelEMA:
                            model_ema.task_DetHead = 2
                        if args.opt == 'adamw_and_sgd':
                            optimizer = optimizer_sgd
                        model = Freeze_Backbone_SegmentationDecoder(model)
                        train_lossAVG, train_lossDiceAvg, train_lossConsAvg1, train_lossConsAvg2, model_ema = train_one_epoch_SEGMENTATION(model, train_loader_seg_CANDIDptx, optimizer, loss_scaler, epoch, head_number=2, log_writter_SEGMENTATION=log_writter_DETECTION, model_ema=model_ema, momen=momentum_schedule_SEG, coff=coff, criterionMSE=criterionMSE, train_type='F')
                        print( "Epoch {:04d}: Train Segmentation CANDIDPTX Loss {:.5f} {:.5f} {:.5f} {:.5f} ".format(epoch, train_lossAVG, train_lossDiceAvg, train_lossConsAvg1, train_lossConsAvg2), file=log_writter_DETECTION )
                        fields=[epoch, 'CANDID-PTX', task_todo, '-', '-', str(train_lossDiceAvg), str((train_lossConsAvg1+train_lossConsAvg2)/2) ] # AUC_SliceLevel_Res
                        with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                            writer = csv.writer(f)
                            writer.writerow(fields)

                    print()
                    print("Segmentation_CANDIDPTX_B_Train")
                    task_todo = "Segmentation_CANDIDPTX_B_Train"
                    model.task_DetHead = 2
                    if args.modelEMA:
                        model_ema.task_DetHead = 2
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_sgd
                    model = unFreeze_Backbone_SegmentationDecoder(model)
                    train_lossAVG, train_lossDiceAvg, train_lossConsAvg1, train_lossConsAvg2, model_ema = train_one_epoch_SEGMENTATION(model, train_loader_seg_CANDIDptx, optimizer, loss_scaler, epoch, head_number=2, log_writter_SEGMENTATION=log_writter_DETECTION, model_ema=model_ema, momen=momentum_schedule_SEG, coff=coff, criterionMSE=criterionMSE, train_type='uF')
                    print( "Epoch {:04d}: Train Segmentation CANDIDPTX Loss {:.5f} {:.5f} {:.5f} {:.5f} ".format(epoch, train_lossAVG, train_lossDiceAvg, train_lossConsAvg1, train_lossConsAvg2), file=log_writter_DETECTION )
                    fields=[epoch, 'CANDID-PTX', task_todo, '-', '-', str(train_lossDiceAvg), str((train_lossConsAvg1+train_lossConsAvg2)/2) ] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)
               
               
                ### ------ RSNA Pneumonia ------ ###
                if TaskHead_number == 12: # Train for RSNA Pneumonia Classification
                    print()
                    print("Classification_RSNAPneumonia_Train")
                    task_todo = "Classification_RSNAPneumonia_Train"
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_sgd
                    train_lossAVG, model_ema = train_CLASSIFICATION(train_loader_cls_RSNApneumonia, model, criterion_CLS, optimizer, epoch, args, log_writter_DETECTION, head_number=3, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE, task_cls_type='nonBinary')
                    print( "Epoch {:04d}: Train Classification RSNAPneumonia Loss {:.5f} ".format(epoch, train_lossAVG) )
                    print( "Epoch {:04d}: Train Classification RSNAPneumonia Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_DETECTION )
                    fields=[epoch, 'RSNAPneumonia', task_todo, str(train_lossAVG), '-', '-'] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)

                if TaskHead_number == 13:  # Train for RSNA Pneumonia Localization
                    if 'locFT' not in args.cyclictask:
                        print()
                        print("Localization_RSNAPneumonia_A_Train") # Train A Set
                        task_todo = "Localization_RSNAPneumonia_A_Train"
                        model = Freeze_Backbone_and_Localization_Encoder(model)
                        model.task_DetHead = 3
                        if args.modelEMA:
                            model_ema.task_DetHead = 3
                        if args.opt == 'adamw_and_sgd':
                            optimizer = optimizer_adamw
                        train_stats, model_ema = train_one_epoch(
                            model, criterion, train_loader_loc_RSNApneumonia, optimizer, device, epoch,
                            args.clip_max_norm, wo_class_error=wo_class_error, DetHead=3, train_type='F', lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                        log_stats = { **{f'train_RSNApneumonia_A_{k}': v for k, v in train_stats.items()} }
                        result_output_dir = args.output_dir + '/results.txt'
                        log_writer_detection = open(result_output_dir, 'a')
                        log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: RSNApneumonia_A " + '\n')
                        log_writer_detection.write('-- Training --' + '\n')
                        formatted_stats_train = {f'train_RSNApneumonia_A_{k}': v for k, v in train_stats.items()}
                        for key, value in formatted_stats_train.items():
                            log_writer_detection.write(f'{key}: {value}\n')
                            if key == 'train_RSNApneumonia_A_loss':
                                loss_localization_temp = value
                        log_writer_detection.write('\n')
                        log_writer_detection.close()
                        fields=[epoch, 'RSNAPneumonia', task_todo, '-', str(loss_localization_temp), '-'] # AUC_SliceLevel_Res
                        with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                            writer = csv.writer(f)
                            writer.writerow(fields)

                    print()
                    print("Localization_RSNAPneumonia_B_Train") # Train B Set
                    task_todo = "Localization_RSNAPneumonia_B_Train"
                    model = unFreeze_Backbone_and_Localization_Encoder(model)
                    model.task_DetHead = 3
                    if args.modelEMA:
                        model_ema.task_DetHead = 3
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_adamw
                    train_stats, model_ema = train_one_epoch(
                        model, criterion, train_loader_loc_RSNApneumonia, optimizer, device, epoch,
                        args.clip_max_norm, wo_class_error=wo_class_error, DetHead=3, train_type='uF', lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                    log_stats = { **{f'train_RSNApneumonia_B_{k}': v for k, v in train_stats.items()} }
                    result_output_dir = args.output_dir + '/results.txt'
                    log_writer_detection = open(result_output_dir, 'a')
                    log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: RSNApneumonia_B " + '\n')
                    log_writer_detection.write('-- Training --' + '\n')
                    formatted_stats_train = {f'train_RSNApneumonia_B_{k}': v for k, v in train_stats.items()}
                    for key, value in formatted_stats_train.items():
                        log_writer_detection.write(f'{key}: {value}\n')
                        if key == 'train_RSNApneumonia_B_loss':
                            loss_localization_temp = value
                    log_writer_detection.write('\n')
                    log_writer_detection.close()
                    fields=[epoch, 'RSNAPneumonia', task_todo, '-', str(loss_localization_temp), '-'] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)

                ### ------ ChestX-Det ------ ###
                if TaskHead_number == 14: # Train for ChestX-Det Classification
                    print()
                    print("Classification_ChestXDet_Train")
                    task_todo = "Classification_ChestXDet_Train"
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_sgd
                    train_lossAVG, model_ema = train_CLASSIFICATION(train_loader_cls_ChestXDet, model, criterion_CLS, optimizer, epoch, args, log_writter_DETECTION, head_number=4, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE, task_cls_type='nonBinary')
                    print( "Epoch {:04d}: Train Classification ChestXDet Loss {:.5f} ".format(epoch, train_lossAVG) )
                    print( "Epoch {:04d}: Train Classification ChestXDet Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_DETECTION )
                    fields=[epoch, 'ChestX-Det', task_todo, str(train_lossAVG), '-', '-'] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)

                if TaskHead_number == 15:  # Train for ChestX-Det Localization
                    if 'locFT' not in args.cyclictask:
                        print()
                        print("Localization_ChestXDet_A_Train") # Train A Set
                        task_todo = "Localization_ChestXDet_A_Train"
                        model = Freeze_Backbone_and_Localization_Encoder(model)
                        model.task_DetHead = 4
                        if args.modelEMA:
                            model_ema.task_DetHead = 4
                        if args.opt == 'adamw_and_sgd':
                            optimizer = optimizer_adamw
                        train_stats, model_ema = train_one_epoch(
                            model, criterion, train_loader_loc_ChestXDet, optimizer, device, epoch,
                            args.clip_max_norm, wo_class_error=wo_class_error, DetHead=4, train_type='F', lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                        log_stats = { **{f'train_ChestXDet_A_{k}': v for k, v in train_stats.items()} }
                        result_output_dir = args.output_dir + '/results.txt'
                        log_writer_detection = open(result_output_dir, 'a')
                        log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: ChestXDet_A " + '\n')
                        log_writer_detection.write('-- Training --' + '\n')
                        formatted_stats_train = {f'train_ChestXDet_A_{k}': v for k, v in train_stats.items()}
                        for key, value in formatted_stats_train.items():
                            log_writer_detection.write(f'{key}: {value}\n')
                            if key == 'train_ChestXDet_A_loss':
                                loss_localization_temp = value
                        log_writer_detection.write('\n')
                        log_writer_detection.close()
                        fields=[epoch, 'ChestX-Det', task_todo, '-', str(loss_localization_temp), '-'] # AUC_SliceLevel_Res
                        with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                            writer = csv.writer(f)
                            writer.writerow(fields)

                    print()
                    print("Localization_ChestXDet_B_Train") # Train B Set
                    task_todo = "Localization_ChestXDet_B_Train"
                    model = unFreeze_Backbone_and_Localization_Encoder(model)
                    model.task_DetHead = 4
                    if args.modelEMA:
                        model_ema.task_DetHead = 4
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_adamw
                    train_stats, model_ema = train_one_epoch(
                        model, criterion, train_loader_loc_ChestXDet, optimizer, device, epoch,
                        args.clip_max_norm, wo_class_error=wo_class_error, DetHead=4, train_type='uF', lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                    log_stats = { **{f'train_ChestXDet_B_{k}': v for k, v in train_stats.items()} }
                    result_output_dir = args.output_dir + '/results.txt'
                    log_writer_detection = open(result_output_dir, 'a')
                    log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: ChestXDet_B " + '\n')
                    log_writer_detection.write('-- Training --' + '\n')
                    formatted_stats_train = {f'train_ChestXDet_B_{k}': v for k, v in train_stats.items()}
                    for key, value in formatted_stats_train.items():
                        log_writer_detection.write(f'{key}: {value}\n')
                        if key == 'train_ChestXDet_B_loss':
                            loss_localization_temp = value
                    log_writer_detection.write('\n')
                    log_writer_detection.close()
                    fields=[epoch, 'ChestX-Det', task_todo, '-', str(loss_localization_temp), '-'] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)

                if TaskHead_number == 16: # Train for ChestX-Det Segmentation
                    if 'segFT' not in args.cyclictask:
                        print()
                        print("Segmentation_ChestXDet_A_Train")
                        task_todo = "Segmentation_ChestXDet_A_Train"
                        model.task_DetHead = 4
                        if args.modelEMA:
                            model_ema.task_DetHead = 4
                        model = Freeze_Backbone_SegmentationDecoder(model)
                        if args.opt == 'adamw_and_sgd':
                            optimizer = optimizer_sgd
                        train_lossAVG, train_lossDiceAvg, train_lossConsAvg1, train_lossConsAvg2, model_ema = train_one_epoch_SEGMENTATION(model, train_loader_seg_ChestXDet, optimizer, loss_scaler, epoch, head_number=4, log_writter_SEGMENTATION=log_writter_DETECTION, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE, train_type='F')
                        print( "Epoch {:04d}: Train Segmentation ChestXDet Loss {:.5f} {:.5f} {:.5f} {:.5f} ".format(epoch, train_lossAVG, train_lossDiceAvg, train_lossConsAvg1, train_lossConsAvg2), file=log_writter_DETECTION )
                        fields=[epoch, 'ChestX-Det', task_todo, '-', '-', str(train_lossDiceAvg), str((train_lossConsAvg1+train_lossConsAvg2)/2) ] # AUC_SliceLevel_Res
                        with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                            writer = csv.writer(f)
                            writer.writerow(fields)

                    print()
                    print("Segmentation_ChestXDet_B_Train")
                    task_todo = "Segmentation_ChestXDet_B_Train"
                    model.task_DetHead = 4
                    if args.modelEMA:
                        model_ema.task_DetHead = 4
                    model = unFreeze_Backbone_SegmentationDecoder(model)
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_sgd
                    train_lossAVG, train_lossDiceAvg, train_lossConsAvg1, train_lossConsAvg2, model_ema = train_one_epoch_SEGMENTATION(model, train_loader_seg_ChestXDet, optimizer, loss_scaler, epoch, head_number=4, log_writter_SEGMENTATION=log_writter_DETECTION, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE, train_type='uF')
                    print( "Epoch {:04d}: Train Segmentation ChestXDet Loss {:.5f} {:.5f} {:.5f} {:.5f} ".format(epoch, train_lossAVG, train_lossDiceAvg, train_lossConsAvg1, train_lossConsAvg2), file=log_writter_DETECTION )
                    fields=[epoch, 'ChestX-Det', task_todo, '-', '-', str(train_lossDiceAvg), str((train_lossConsAvg1+train_lossConsAvg2)/2) ] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)

                ### ------ SIIM_ACR_Pneumothorax ------ ###
                if TaskHead_number == 17: # Train for SIIM_ACR_Pneumothorax Classification
                    print()
                    print("Classification_SIIMACR_Train")
                    task_todo = "Classification_SIIMACR_Train"
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_sgd
                    train_lossAVG, model_ema = train_CLASSIFICATION(train_loader_cls_SIIMACRptx, model, criterion_CLS, optimizer, epoch, args, log_writter_DETECTION, head_number=5, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE, task_cls_type='nonBinary')
                    print( "Epoch {:04d}: Train Classification SIIMACR Loss {:.5f} ".format(epoch, train_lossAVG) )
                    print( "Epoch {:04d}: Train Classification SIIMACR Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_DETECTION )
                    fields=[epoch, 'SIIM-ACR', task_todo, str(train_lossAVG), '-', '-'] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)

                if TaskHead_number == 18:  # Train for SIIM_ACR_Pneumothorax Localization
                    if 'locFT' not in args.cyclictask:
                        print()
                        print("Localization_SIIMACR_A_Train") # Train A Set
                        task_todo = "Localization_SIIMACR_A_Train"
                        model = Freeze_Backbone_and_Localization_Encoder(model)
                        model.task_DetHead = 5
                        if args.modelEMA:
                            model_ema.task_DetHead = 5
                        if args.opt == 'adamw_and_sgd':
                            optimizer = optimizer_adamw
                        train_stats, model_ema = train_one_epoch(
                            model, criterion, train_loader_loc_SiimACR, optimizer, device, epoch,
                            args.clip_max_norm, wo_class_error=wo_class_error, DetHead=5, train_type='F', lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                        log_stats = { **{f'train_SIIMACR_A_{k}': v for k, v in train_stats.items()} }
                        result_output_dir = args.output_dir + '/results.txt'
                        log_writer_detection = open(result_output_dir, 'a')
                        log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: SIIMACR_A " + '\n')
                        log_writer_detection.write('-- Training --' + '\n')
                        formatted_stats_train = {f'train_SIIMACR_A_{k}': v for k, v in train_stats.items()}
                        for key, value in formatted_stats_train.items():
                            log_writer_detection.write(f'{key}: {value}\n')
                            if key == 'train_SIIMACR_A_loss':
                                loss_localization_temp = value
                        log_writer_detection.write('\n')
                        log_writer_detection.close()
                        fields=[epoch, 'SIIM-ACR', task_todo, '-', str(loss_localization_temp), '-'] # AUC_SliceLevel_Res
                        with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                            writer = csv.writer(f)
                            writer.writerow(fields)

                    print()
                    print("Localization_SIIMACR_B_Train") # Train B Set
                    task_todo = "Localization_SIIMACR_B_Train"
                    model = unFreeze_Backbone_and_Localization_Encoder(model)
                    model.task_DetHead = 5
                    if args.modelEMA:
                        model_ema.task_DetHead = 5
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_adamw
                    train_stats, model_ema = train_one_epoch(
                        model, criterion, train_loader_loc_SiimACR, optimizer, device, epoch,
                        args.clip_max_norm, wo_class_error=wo_class_error, DetHead=5, train_type='uF', lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                    log_stats = { **{f'train_SIIMACR_B_{k}': v for k, v in train_stats.items()} }
                    result_output_dir = args.output_dir + '/results.txt'
                    log_writer_detection = open(result_output_dir, 'a')
                    log_writer_detection.write('Epoch: ' + str(epoch) + "| TaskHead_number: " + str(TaskHead_number) + " Training: SIIMACR_B " + '\n')
                    log_writer_detection.write('-- Training --' + '\n')
                    formatted_stats_train = {f'train_SIIMACR_B_{k}': v for k, v in train_stats.items()}
                    for key, value in formatted_stats_train.items():
                        log_writer_detection.write(f'{key}: {value}\n')
                        if key == 'train_SIIMACR_B_loss':
                            loss_localization_temp = value
                    log_writer_detection.write('\n')
                    log_writer_detection.close()
                    fields=[epoch, 'SIIM-ACR', task_todo, '-', str(loss_localization_temp), '-'] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)

                if TaskHead_number == 19: # Train for SIIMACR Segmentation
                    if 'segFT' not in args.cyclictask:
                        print()
                        print("Segmentation_SIIMACR_A_Train")
                        task_todo = "Segmentation_SIIMACR_A_Train"
                        model.task_DetHead = 5
                        if args.modelEMA:
                            model_ema.task_DetHead = 5
                        model = Freeze_Backbone_SegmentationDecoder(model)
                        if args.opt == 'adamw_and_sgd':
                            optimizer = optimizer_sgd
                        train_lossAVG, train_lossDiceAvg, train_lossConsAvg1, train_lossConsAvg2, model_ema = train_one_epoch_SEGMENTATION(model, train_loader_seg_SIIM, optimizer, loss_scaler, epoch, head_number=5, log_writter_SEGMENTATION=log_writter_DETECTION, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE, train_type='F')
                        print( "Epoch {:04d}: Train Segmentation SIIMACR Loss {:.5f} {:.5f} {:.5f} {:.5f} ".format(epoch, train_lossAVG, train_lossDiceAvg, train_lossConsAvg1, train_lossConsAvg2), file=log_writter_DETECTION )
                        fields=[epoch, 'SIIM-ACR', task_todo, '-', '-', str(train_lossDiceAvg), str((train_lossConsAvg1+train_lossConsAvg1)/2)] # AUC_SliceLevel_Res
                        with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                            writer = csv.writer(f)
                            writer.writerow(fields)

                    print()
                    print("Segmentation_SIIMACR_B_Train")
                    task_todo = "Segmentation_SIIMACR_B_Train"
                    model.task_DetHead = 5
                    if args.modelEMA:
                        model_ema.task_DetHead = 5
                    model = unFreeze_Backbone_SegmentationDecoder(model)
                    if args.opt == 'adamw_and_sgd':
                        optimizer = optimizer_sgd
                    train_lossAVG, train_lossDiceAvg, train_lossConsAvg1, train_lossConsAvg2, model_ema = train_one_epoch_SEGMENTATION(model, train_loader_seg_SIIM, optimizer, loss_scaler, epoch, head_number=5, log_writter_SEGMENTATION=log_writter_DETECTION, model_ema=model_ema, momen=momentum_schedule_SEG, coff=coff, criterionMSE=criterionMSE, train_type='uF')
                    print( "Epoch {:04d}: Train Segmentation SIIMACR Loss {:.5f} {:.5f} {:.5f} {:.5f} ".format(epoch, train_lossAVG, train_lossDiceAvg, train_lossConsAvg1, train_lossConsAvg2), file=log_writter_DETECTION )
                    fields=[epoch, 'SIIM-ACR', task_todo, '-', '-', str(train_lossDiceAvg), str((train_lossConsAvg1+train_lossConsAvg1)/2) ] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print(' -- Train time for Epoch {} - TaskHead {}: {}'.format(epoch, TaskHead_number, total_time_str), file=log_writter_DETECTION)

            ## Model EMA/Teacher Update -- Fix: updating EMA_Model within the trainer function
            # if args.modelEMA == "True_Epoch":
            #     ema_update_teacher(model, model_ema, momentum_schedule, epoch)
            #     ### model_ema = model # should ignore later # DEBUG purpose

            # save_file = os.path.join(model_path_SEGMENTATION, 'ckpt_E'+str(epoch)+'_TH'+str(TaskHead_number)+'.pth')
            # save_model(model, optimizer, log_writter_SEGMENTATION, epoch, save_file, model_ema=model_ema)
            # print('\n', file=log_writter_SEGMENTATION)


            ### Checking which of the layers have been updated/modified
            if epoch < args.total_epochs:
                if not os.path.exists(args.output_dir + '/parameter_check'):
                    os.makedirs(args.output_dir + '/parameter_check')
                with open(args.output_dir + '/parameter_check/' +  f'parameters_epoch_{epoch}.txt', 'w') as file:
                    MODEL_current_parameters = {n: p.detach().cpu().numpy().tolist() for n, p in model.named_parameters() }
                    changed_params = {n for n, p in MODEL_current_parameters.items() if p != MODEL_prev_parameters.get(n)}
                    file.write(f"Epoch {epoch} - Parameters that have changed:\n")
                    file.write(json.dumps(list(changed_params), indent=2))
                    MODEL_prev_parameters = MODEL_current_parameters

            ### Testing Phase ###
            start_time = time.time()

            ### Test CLASSIFICATION ##
            if 'TESTcls' in args.cyclictask and 'chexpertCLS' in args.cyclictask:
                datasetname__ = "CheXpert"
                head_number_temp = 6
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader_cls_CheXpert
                multiclassClassificaitonTask = False #"multi-label classification"
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'nihchestxray14CLS' in args.cyclictask:
                datasetname__ = "NIHChestXray14"
                head_number_temp = 7
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader_cls_NIHChestXray14
                multiclassClassificaitonTask = False #"multi-label classification"
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'vindrcxrCLS' in args.cyclictask:
                datasetname__ = "VinDRCXR"
                head_number_temp = 8
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader_cls_VinDRCXR
                multiclassClassificaitonTask = False #"multi-label classification"
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'nihshenzenCLS' in args.cyclictask:
                datasetname__ = "NIHShenzhen"
                head_number_temp = 9
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader_cls_NIHShenzhen
                multiclassClassificaitonTask = False #"binary classification"
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'mimic2CLS' in args.cyclictask:
                datasetname__ = "MIMICII"
                head_number_temp = 10
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader_cls_MIMICII
                multiclassClassificaitonTask = False #"multi-label classification"
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'tbx11kCLS' in args.cyclictask:
                datasetname__ = "TBX11k"
                head_number_temp = 0
                task_cls_type_temp = None
                test_loader_cls_temp = test_loader_cls_TBX11k
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'node21CLS' in args.cyclictask:
                datasetname__ = "Node21"
                head_number_temp = 1
                task_cls_type_temp = None
                test_loader_cls_temp = test_loader_cls_NODE21
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'candidptxCLS' in args.cyclictask:
                datasetname__ = "CANDID-PTX"
                head_number_temp = 2
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader_cls_CANDIDptx
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'rsnapneumoniaCLS' in args.cyclictask:
                datasetname__ = "RSNApneumonia"
                head_number_temp = 3
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader_cls_RSNApneumonia
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'chestxdetCLS' in args.cyclictask:
                datasetname__ = "ChestX-Det"
                head_number_temp = 4
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader_cls_ChestXDet
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'siimacrCLS' in args.cyclictask:
                datasetname__ = "SIIM-ACRpneumothorax"
                head_number_temp = 5
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader_cls_SIIMACRptx
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)


            ### Test LOCALIZATION ##
            if 'TESTloc' in args.cyclictask and 'tbx11kLOC' in args.cyclictask:
                datasetname__ = "TBX11k"
                model.task_DetHead = 0
                if model_ema is not None:
                    model_ema.task_DetHead = 0
                DetHead_temp = 0
                base_ds_temp = base_ds_TBX11k
                test_loader_loc_temp = test_loader_loc_TBX11k
                evaluateLocSepFunc(epoch, datasetname__, task_todo, model, criterion, postprocessors, model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, TaskHead_number, DetHead_temp, wo_class_error, logger)
            if 'TESTloc' in args.cyclictask and 'node21LOC' in args.cyclictask:
                datasetname__ = "Node21"
                model.task_DetHead = 1
                if model_ema is not None:
                    model_ema.task_DetHead = 1
                DetHead_temp = 1
                base_ds_temp = base_ds_Node21
                test_loader_loc_temp = test_loader_loc_Node21
                evaluateLocSepFunc(epoch, datasetname__, task_todo, model, criterion, postprocessors, model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, TaskHead_number, DetHead_temp, wo_class_error, logger)
            if 'TESTloc' in args.cyclictask and 'candidptxLOC' in args.cyclictask:
                datasetname__ = "CANDID-PTX"
                model.task_DetHead = 2
                if model_ema is not None:
                    model_ema.task_DetHead = 2
                DetHead_temp = 2
                base_ds_temp = base_ds_CANDIDptx
                test_loader_loc_temp = test_loader_loc_CANDIDptx
                evaluateLocSepFunc(epoch, datasetname__, task_todo, model, criterion, postprocessors, model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, TaskHead_number, DetHead_temp, wo_class_error, logger)
            if 'TESTloc' in args.cyclictask and 'rsnapneumoniaLOC' in args.cyclictask:
                datasetname__ = "RSNApneumonia"
                model.task_DetHead = 3
                if model_ema is not None:
                    model_ema.task_DetHead = 3
                DetHead_temp = 3
                base_ds_temp = base_ds_RSNApneumonia
                test_loader_loc_temp = test_loader_loc_RSNApneumonia
                evaluateLocSepFunc(epoch, datasetname__, task_todo, model, criterion, postprocessors, model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, TaskHead_number, DetHead_temp, wo_class_error, logger)
            if 'TESTloc' in args.cyclictask and 'chestxdetLOC' in args.cyclictask:
                datasetname__ = "ChestX-Det"
                model.task_DetHead = 4
                if model_ema is not None:
                    model_ema.task_DetHead = 4
                DetHead_temp = 5
                base_ds_temp = base_ds_ChestXDet
                test_loader_loc_temp = test_loader_loc_ChestXDet
                evaluateLocSepFunc(epoch, datasetname__, task_todo, model, criterion, postprocessors, model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, TaskHead_number, DetHead_temp, wo_class_error, logger)
            if 'TESTloc' in args.cyclictask and 'siimacrLOC' in args.cyclictask:
                datasetname__ = "SIIM-ACRpneumothorax"
                model.task_DetHead = 5
                if model_ema is not None:
                    model_ema.task_DetHead = 5
                DetHead_temp = 5
                base_ds_temp = base_ds_SiimACR
                test_loader_loc_temp = test_loader_loc_SiimACR
                evaluateLocSepFunc(epoch, datasetname__, task_todo, model, criterion, postprocessors, model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, TaskHead_number, DetHead_temp, wo_class_error, logger)

            if 'TESTloc' in args.cyclictask and 'objects365LOC' in args.cyclictask: # locFT_TESTloc_objects365LOC
                datasetname__ = "Objects365"
                model.task_DetHead = 0
                if model_ema is not None:
                    model_ema.task_DetHead = 0
                DetHead_temp = 0
                base_ds_temp = base_ds # base_ds_TBX11k
                test_loader_loc_temp = test_loader_loc_TBX11k
                evaluateLocSepFunc(epoch, datasetname__, task_todo, model, criterion, postprocessors, model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, TaskHead_number, DetHead_temp, wo_class_error, logger)


            ### Test SEGMENTATION ##
            if 'TESTseg' in args.cyclictask and 'candidptxSEG' in args.cyclictask:
                datasetname__ = "CANDID-PTX"
                head_number_temp = 2
                test_loader_loc_temp = test_loader_seg_CANDIDptx
                model.task_DetHead = 2
                if model_ema is not None:
                    model_ema.task_DetHead = 2
                evaluateSegSepFunc(epoch, datasetname__, task_todo, model, model_ema, test_loader_loc_temp, head_number_temp, log_writter_DETECTION)
            if 'TESTseg' in args.cyclictask and 'chestxdetSEG' in args.cyclictask:
                datasetname__ = "ChestX-Det"
                head_number_temp = 4
                test_loader_loc_temp = test_loader_seg_ChestXDet
                model.task_DetHead = 4
                if model_ema is not None:
                    model_ema.task_DetHead = 4
                evaluateSegSepFunc(epoch, datasetname__, task_todo, model, model_ema, test_loader_loc_temp, head_number_temp, log_writter_DETECTION)
            if 'TESTseg' in args.cyclictask and 'siimacrSEG' in args.cyclictask:
                datasetname__ = "SIIM-ACRpneumothorax"
                head_number_temp = 5
                test_loader_loc_temp = test_loader_seg_SIIM
                model.task_DetHead = 5
                if model_ema is not None:
                    model_ema.task_DetHead = 5
                evaluateSegSepFunc(epoch, datasetname__, task_todo, model, model_ema, test_loader_loc_temp, head_number_temp, log_writter_DETECTION)


            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Test time Epoch {} TaskHead {} -- {}\n\n'.format(epoch, TaskHead_number, total_time_str), file=log_writter_DETECTION)
            log_writter_DETECTION.flush()

            if args.debugOnlyTest == True:
                exit()

            if args.saveAllModel == False and store_best_performance_found == 1:
                save_file = os.path.join(args.output_dir, 'ckpt_E'+str(epoch)+'_TH'+str(TaskHead_number)+'.pth')
                save_model(model, optimizer, log_writter_DETECTION, epoch, save_file, model_ema=model_ema)
                print('\n', file=log_writter_DETECTION)
                log_writter_DETECTION.flush()
            elif args.saveAllModel == True:
                save_file = os.path.join(args.output_dir, 'ckpt_E'+str(epoch)+'_TH'+str(TaskHead_number)+'.pth')
                if args.opt != 'adamw_and_sgd':
                    save_model(model, optimizer, log_writter_DETECTION, epoch, save_file, model_ema=model_ema)
                else:
                    save_model2(model, optimizer_adamw, optimizer_sgd, log_writter_DETECTION, epoch, save_file, model_ema=model_ema)
                print('\n', file=log_writter_DETECTION)
                log_writter_DETECTION.flush()




                

        if args.taskcomponent == "foundation_x3_FineTuning": ## Full Foundation-X Finetuning
            if args.modelEMA:
                model_ema.eval()
            # EMA Update Epoch-wise
            if args.modelEMA == "True_Epoch":
                if epoch >= Num_EPOCH_Iterative_Steps_MomentumSchduler:
                    coff = 0.5
                else:
                    coff = (momentum_schedule[epoch] - 0.9) * 5 # Epoch-wise
                coff = 0.5

            ## Optimizer and Learning Rate Update
            if args.opt == 'adamw' or args.opt == 'sgd':
                lrBackbone_ = step_decay(epoch, args.lr_backbone, args.total_epochs, step_inc=10)
                lrSegmentor_ = step_decay(epoch, args.lr_segmentor, args.total_epochs, step_inc=10)
                lrLocalizerEnc_ = step_decay(epoch, args.lr_locEnc, args.total_epochs, step_inc=10) 
                lrLocalizerDec_ = step_decay(epoch, args.lr_locDec, args.total_epochs, step_inc=10) 
                if len(optimizer.param_groups) == 2 or len(optimizer.param_groups) == 3 or len(optimizer.param_groups) == 4:
                    optimizer.param_groups[0]['lr'] = lrBackbone_
                    optimizer.param_groups[1]['lr'] = lrSegmentor_
                    optimizer.param_groups[2]['lr'] = lrLocalizerEnc_
                    optimizer.param_groups[3]['lr'] = lrLocalizerDec_
                    print('Epoch{} - learning_rateBackbone [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[0]['lr']))
                    print('Epoch{} - learning_rateSegmentor [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[1]['lr']))
                    print('Epoch{} - learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[2]['lr']))
                    print('Epoch{} - learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[3]['lr']))
                    # print('Epoch{} - TaskHead{} - learning_rateREST [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer['lr']))

                    print('Epoch{} - learning_rateBackbone [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[0]['lr']), file=log_writter_DETECTION)
                    print('Epoch{} - learning_rateSegmentor [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[1]['lr']), file=log_writter_DETECTION)
                    print('Epoch{} - learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[2]['lr']), file=log_writter_DETECTION)
                    print('Epoch{} - learning_rateLocalizer [Updated]: {:.20f}'.format(epoch, optimizer.param_groups[3]['lr']), file=log_writter_DETECTION)
                else:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_
                    print('Epoch{} - TaskHead{} - learning_rate [Updated]: {:.20f}'.format(epoch, TaskHead_number, optimizer.param_groups[0]['lr']), file=log_writter_DETECTION)

            ## Preparation:
            if 'chexpertCLS' in args.cyclictask:
                datasetname__ = "CheXpert"
                head_number_temp = 6
                task_cls_type_temp = 'nonBinary'
            if 'nihchestxray14CLS' in args.cyclictask:
                datasetname__ = "NIHChestXray14"
                head_number_temp = 7
                task_cls_type_temp = 'nonBinary'
            if 'vindrcxrCLS' in args.cyclictask:
                datasetname__ = "VinDRCXR"
                head_number_temp = 8
                task_cls_type_temp = 'nonBinary'
            if 'nihshenzenCLS' in args.cyclictask:
                datasetname__ = "NIHShenzhen"
                head_number_temp = 9
                task_cls_type_temp = 'nonBinary'
            if 'mimic2CLS' in args.cyclictask:
                datasetname__ = "MIMICII"
                head_number_temp = 10
                task_cls_type_temp = 'nonBinary'
            if 'tbx11kCLS' in args.cyclictask:
                datasetname__ = "TBX11k"
                head_number_temp = 0
                task_cls_type_temp = None
            if 'node21CLS' in args.cyclictask:
                datasetname__ = "Node21"
                head_number_temp = 1
                task_cls_type_temp = None
            if 'candidptxCLS' in args.cyclictask:
                datasetname__ = "CANDID-PTX"
                head_number_temp = 2
                task_cls_type_temp = 'nonBinary'
            if 'rsnapneumoniaCLS' in args.cyclictask:
                datasetname__ = "RSNApneumonia"
                head_number_temp = 3
                task_cls_type_temp = 'nonBinary'
            if 'chestxdetCLS' in args.cyclictask:
                datasetname__ = "ChestX-Det"
                head_number_temp = 4
                task_cls_type_temp = 'nonBinary'
            if 'siimacrCLS' in args.cyclictask:
                datasetname__ = "SIIM-ACRpneumothorax"
                head_number_temp = 5
                task_cls_type_temp = 'nonBinary'
            if 'tbx11kLOC' in args.cyclictask:
                datasetname__ = "TBX11k"
                head_number_temp = 0
            if 'node21LOC' in args.cyclictask:
                datasetname__ = "Node21"
                head_number_temp = 1
            if 'candidptxLOC' in args.cyclictask:
                datasetname__ = "CANDID-PTX"
                head_number_temp = 2
            if 'rsnapneumoniaLOC' in args.cyclictask:
                datasetname__ = "RSNApneumonia"
                head_number_temp = 3
            if 'chestxdetLOC' in args.cyclictask:
                datasetname__ = "ChestX-Det"
                head_number_temp = 4
            if 'siimacrLOC' in args.cyclictask:
                datasetname__ = "SIIM-ACRpneumothorax"
                head_number_temp = 5
            if 'candidptxSEG' in args.cyclictask:
                datasetname__ = "CANDID-PTX"
                head_number_temp = 2
            if 'chestxdetSEG' in args.cyclictask:
                datasetname__ = "ChestX-Det"
                head_number_temp = 4
            if 'siimacrSEG' in args.cyclictask:
                datasetname__ = "SIIM-ACRpneumothorax"
                head_number_temp = 5

            if 'objects365LOC' in args.cyclictask:
                datasetname__ = "Objects365"
                head_number_temp = 0

            if args.modelEMA is None:
                coff = None
                momentum_schedule = None
                criterionMSE = None
                model_ema = None
            for param in model.parameters():
                param.requires_grad = True    
            
            ### Training Phase ###
            if args.debugOnlyTest == False:
                start_time = time.time()
                if 'clsFT' in args.cyclictask:
                    print()
                    print("Classification_Train")
                    task_todo = "Classification_Train"
                    train_lossAVG, model_ema = train_CLASSIFICATION(train_loader, model, criterion_CLS, optimizer, epoch, args, log_writter_DETECTION, head_number=head_number_temp, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE, task_cls_type=task_cls_type_temp)
                    print( "Epoch {:04d}: Train Classification Loss {:.5f} ".format(epoch, train_lossAVG) )
                    print( "Epoch {:04d}: Train Classification Loss {:.5f} ".format(epoch, train_lossAVG), file=log_writter_DETECTION )
                    fields=[epoch, datasetname__, task_todo, str(train_lossAVG), '-', '-'] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)

                if 'locFT' in args.cyclictask:
                    print()
                    print("Localization_Train") # Train B Set
                    task_todo = "Localization_Train"
                    # model = unFreeze_Backbone_and_Localization_Encoder(model)
                    model.task_DetHead = head_number_temp
                    if args.modelEMA:
                        model_ema.task_DetHead = head_number_temp
                    train_stats, model_ema = train_one_epoch(
                        model, criterion, train_loader, optimizer, device, epoch,
                        args.clip_max_norm, wo_class_error=wo_class_error, DetHead=head_number_temp, train_type='uF', lr_scheduler=None, args=args, logger=(logger if args.save_log else None), model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE)
                    log_stats = { **{f'train_{k}': v for k, v in train_stats.items()} }
                    result_output_dir = args.output_dir + '/results.txt'
                    log_writer_detection = open(result_output_dir, 'a')
                    log_writer_detection.write('Epoch: ' + str(epoch) + " Training: Dataset " + '\n')
                    log_writer_detection.write('-- Training --' + '\n')
                    formatted_stats_train = {f'train_{k}': v for k, v in train_stats.items()}
                    for key, value in formatted_stats_train.items():
                        log_writer_detection.write(f'{key}: {value}\n')
                        if key == 'train_loss':
                            loss_localization_temp = value
                    log_writer_detection.write('\n')
                    log_writer_detection.close()
                    fields=[epoch, datasetname__, task_todo, '-', str(loss_localization_temp), '-'] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)

                if 'segFT' in args.cyclictask:
                    print()
                    print("Segmentation_Train")
                    task_todo = "Segmentation_Train"
                    model.task_DetHead = head_number_temp
                    if args.modelEMA:
                        model_ema.task_DetHead = head_number_temp
                    # model = unFreeze_Backbone_SegmentationDecoder(model)
                    train_lossAVG, train_lossDiceAvg, train_lossConsAvg1, train_lossConsAvg2, model_ema = train_one_epoch_SEGMENTATION(model, train_loader, optimizer, loss_scaler, epoch, head_number=head_number_temp, log_writter_SEGMENTATION=log_writter_DETECTION, model_ema=model_ema, momen=momentum_schedule, coff=coff, criterionMSE=criterionMSE, train_type='uF')
                    print( "Epoch {:04d}: Train Segmentation Loss {:.5f} {:.5f} {:.5f} {:.5f} ".format(epoch, train_lossAVG, train_lossDiceAvg, train_lossConsAvg1, train_lossConsAvg2), file=log_writter_DETECTION )
                    fields=[epoch, datasetname__, task_todo, '-', '-', str(train_lossDiceAvg), str((train_lossConsAvg1+train_lossConsAvg2)/2) ] # AUC_SliceLevel_Res
                    with open(args.output_dir+'/export_csvFile_TRAIN.csv', 'a') as f: # export_csvFile_train = pd.DataFrame(columns=['Epoch', 'Dataset', 'Task-Train', 'Cls_Loss', 'Loc_Loss','Seg_Loss'])
                        writer = csv.writer(f)
                        writer.writerow(fields)
                
                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                print('Training time Epoch {} -- {}\n\n'.format(epoch, total_time_str), file=log_writter_DETECTION)
                log_writter_DETECTION.flush()


            ### Testing Phase ###
            start_time = time.time()

            ### Test CLASSIFICATION ##
            if 'TESTcls' in args.cyclictask and 'chexpertCLS' in args.cyclictask:
                datasetname__ = "CheXpert"
                head_number_temp = 6
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader
                multiclassClassificaitonTask = False #"multi-label classification"
                task_todo = "Classification_Train" + "_" + datasetname__
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'nihchestxray14CLS' in args.cyclictask:
                datasetname__ = "NIHChestXray14"
                head_number_temp = 7
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader
                multiclassClassificaitonTask = False #"multi-label classification"
                task_todo = "Classification_Train" + "_" + datasetname__
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'vindrcxrCLS' in args.cyclictask:
                datasetname__ = "VinDRCXR"
                head_number_temp = 8
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader
                multiclassClassificaitonTask = False #"multi-label classification"
                task_todo = "Classification_Train" + "_" + datasetname__
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'nihshenzenCLS' in args.cyclictask:
                datasetname__ = "NIHShenzhen"
                head_number_temp = 9
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader
                multiclassClassificaitonTask = False #"binary classification"
                task_todo = "Classification_Train" + "_" + datasetname__
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'mimic2CLS' in args.cyclictask:
                datasetname__ = "MIMICII"
                head_number_temp = 10
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader
                multiclassClassificaitonTask = False #"multi-label classification"
                task_todo = "Classification_Train" + "_" + datasetname__
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'tbx11kCLS' in args.cyclictask:
                datasetname__ = "TBX11k"
                head_number_temp = 0
                task_cls_type_temp = None
                test_loader_cls_temp = test_loader
                task_todo = "Classification_Train" + "_" + datasetname__
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'node21CLS' in args.cyclictask:
                datasetname__ = "Node21"
                head_number_temp = 1
                task_cls_type_temp = None
                test_loader_cls_temp = test_loader
                task_todo = "Classification_Train" + "_" + datasetname__
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'candidptxCLS' in args.cyclictask:
                datasetname__ = "CANDID-PTX"
                head_number_temp = 2
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader
                task_todo = "Classification_Train" + "_" + datasetname__
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'rsnapneumoniaCLS' in args.cyclictask:
                datasetname__ = "RSNApneumonia"
                head_number_temp = 3
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader
                task_todo = "Classification_Train" + "_" + datasetname__
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'chestxdetCLS' in args.cyclictask:
                datasetname__ = "ChestX-Det"
                head_number_temp = 4
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader
                task_todo = "Classification_Train" + "_" + datasetname__
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)
            if 'TESTcls' in args.cyclictask and 'siimacrCLS' in args.cyclictask:
                datasetname__ = "SIIM-ACRpneumothorax"
                head_number_temp = 5
                task_cls_type_temp = 'nonBinary'
                test_loader_cls_temp = test_loader
                task_todo = "Classification_Train" + "_" + datasetname__
                evaluateClsSepFunc(epoch, datasetname__, task_todo, model, model_ema, criterion_CLS, log_writter_DETECTION, head_number_temp, task_cls_type_temp, test_loader_cls_temp)


            ### Test LOCALIZATION ##
            if 'TESTloc' in args.cyclictask and 'tbx11kLOC' in args.cyclictask:
                datasetname__ = "TBX11k"
                model.task_DetHead = 0
                if model_ema is not None:
                    model_ema.task_DetHead = 0
                DetHead_temp = 0
                base_ds_temp = base_ds
                test_loader_loc_temp = test_loader
                task_todo = "Localization_Train" + "_" + datasetname__
                evaluateLocSepFunc(epoch, datasetname__, task_todo, model, criterion, postprocessors, model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, 0, DetHead_temp, wo_class_error, logger)
            if 'TESTloc' in args.cyclictask and 'node21LOC' in args.cyclictask:
                datasetname__ = "Node21"
                model.task_DetHead = 1
                if model_ema is not None:
                    model_ema.task_DetHead = 1
                DetHead_temp = 1
                base_ds_temp = base_ds
                test_loader_loc_temp = test_loader
                task_todo = "Localization_Train" + "_" + datasetname__
                evaluateLocSepFunc(epoch, datasetname__, task_todo, model, criterion, postprocessors, model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, 0, DetHead_temp, wo_class_error, logger)
            if 'TESTloc' in args.cyclictask and 'candidptxLOC' in args.cyclictask:
                datasetname__ = "CANDID-PTX"
                model.task_DetHead = 2
                if model_ema is not None:
                    model_ema.task_DetHead = 2
                DetHead_temp = 2
                base_ds_temp = base_ds
                test_loader_loc_temp = test_loader
                task_todo = "Localization_Train" + "_" + datasetname__
                evaluateLocSepFunc(epoch, datasetname__, task_todo, model, criterion, postprocessors, model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, 0, DetHead_temp, wo_class_error, logger)
            if 'TESTloc' in args.cyclictask and 'rsnapneumoniaLOC' in args.cyclictask:
                datasetname__ = "RSNApneumonia"
                model.task_DetHead = 3
                if model_ema is not None:
                    model_ema.task_DetHead = 3
                DetHead_temp = 3
                base_ds_temp = base_ds
                test_loader_loc_temp = test_loader
                task_todo = "Localization_Train" + "_" + datasetname__
                evaluateLocSepFunc(epoch, datasetname__, task_todo, model, criterion, postprocessors, model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, 0, DetHead_temp, wo_class_error, logger)
            if 'TESTloc' in args.cyclictask and 'chestxdetLOC' in args.cyclictask:
                datasetname__ = "ChestX-Det"
                model.task_DetHead = 4
                if model_ema is not None:
                    model_ema.task_DetHead = 4
                DetHead_temp = 5
                base_ds_temp = base_ds
                test_loader_loc_temp = test_loader
                task_todo = "Localization_Train" + "_" + datasetname__
                evaluateLocSepFunc(epoch, datasetname__, task_todo, model, criterion, postprocessors, model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, 0, DetHead_temp, wo_class_error, logger)
            if 'TESTloc' in args.cyclictask and 'siimacrLOC' in args.cyclictask:
                datasetname__ = "SIIM-ACRpneumothorax"
                model.task_DetHead = 5
                if model_ema is not None:
                    model_ema.task_DetHead = 5
                DetHead_temp = 5
                base_ds_temp = base_ds
                test_loader_loc_temp = test_loader
                task_todo = "Localization_Train" + "_" + datasetname__
                evaluateLocSepFunc(epoch, datasetname__, task_todo, model, criterion, postprocessors, model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, 0, DetHead_temp, wo_class_error, logger)

            if 'TESTloc' in args.cyclictask and 'objects365LOC' in args.cyclictask: # locFT_TESTloc_objects365LOC
                datasetname__ = "Objects365"
                model.task_DetHead = 0
                if model_ema is not None:
                    model_ema.task_DetHead = 0
                DetHead_temp = 0
                base_ds_temp = base_ds
                test_loader_loc_temp = test_loader
                task_todo = "Localization_Train" + "_" + datasetname__
                evaluateLocSepFunc(epoch, datasetname__, task_todo, model, criterion, postprocessors, model_ema, criterion_ema, postprocessors_ema, test_loader_loc_temp, base_ds_temp, 0, DetHead_temp, wo_class_error, logger)


            ### Test SEGMENTATION ##
            if 'TESTseg' in args.cyclictask and 'candidptxSEG' in args.cyclictask:
                datasetname__ = "CANDID-PTX"
                head_number_temp = 2
                test_loader_loc_temp = test_loader
                model.task_DetHead = 2
                if model_ema is not None:
                    model_ema.task_DetHead = 2
                task_todo = "Segmentation_Train" + "_" + datasetname__
                evaluateSegSepFunc(epoch, datasetname__, task_todo, model, model_ema, test_loader_loc_temp, head_number_temp, log_writter_DETECTION)
            if 'TESTseg' in args.cyclictask and 'chestxdetSEG' in args.cyclictask:
                datasetname__ = "ChestX-Det"
                head_number_temp = 4
                test_loader_loc_temp = test_loader
                model.task_DetHead = 4
                if model_ema is not None:
                    model_ema.task_DetHead = 4
                task_todo = "Segmentation_Train" + "_" + datasetname__
                evaluateSegSepFunc(epoch, datasetname__, task_todo, model, model_ema, test_loader_loc_temp, head_number_temp, log_writter_DETECTION)
            if 'TESTseg' in args.cyclictask and 'siimacrSEG' in args.cyclictask:
                datasetname__ = "SIIM-ACRpneumothorax"
                head_number_temp = 5
                test_loader_loc_temp = test_loader
                model.task_DetHead = 5
                if model_ema is not None:
                    model_ema.task_DetHead = 5
                task_todo = "Segmentation_Train" + "_" + datasetname__
                evaluateSegSepFunc(epoch, datasetname__, task_todo, model, model_ema, test_loader_loc_temp, head_number_temp, log_writter_DETECTION)


            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Test time Epoch {} -- {}\n\n'.format(epoch, total_time_str), file=log_writter_DETECTION)
            log_writter_DETECTION.flush()

            if args.debugOnlyTest == True:
                exit()









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
