#!/bin/bash
#SBATCH --job-name=FoundationX_run104_RCons_1LocDec

#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -c 12
##SBATCH --exclusive
#SBATCH --mem=80G
#SBATCH -p general
#SBATCH -t 7-00:00:00
#SBATCH -q public

#SBATCH -o %x_slurm_%j.out     
#SBATCH -e %xslurm_%j.err      
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=nuislam@asu.edu

# # A100 SOL
module load mamba/latest
module load cuda-11.6.2-gcc-12.1.0
source activate tf-tnt-gpu2

# # Grace Hopper SOL
# module load mamba/latest
# module load cuda-12.4.1-gcc-11.4.1
# mamba activate gh_gpu1


# # Swin-L + DINO
# CONFIGFILE=config/DINO/DINO_4scale_swinBASE.py
CONFIGFILE=config/DINO/DINO_4scale_swinLARGE384.py
LOGFILE=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run104_Ark6F6_ClsLocSeg_b24_AdamW_LockReleaseAll_RCons_1LocDec_TESTrun
# backbone_dir=/data/jliang12/dongaoma/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar ## Swin-B
backbone_dir=/data/jliang12/shared/pretrained_models/Ark_models/Ark6_swinLarge768_ep50.pth.tar ## Swin-L

BACKBONEMODEL=Swin-L # Swin-T, Swin-B, Swin-L
IMGSIZE=768
coco_path=/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/
DATASETFILE=foundation6Ark6_datasets

### 1e-1 = 0.1
### 1e-2 = 0.01
### 1e-3 = 0.001
### 1e-4 = 0.0001
### 1e-5 = 0.00001

## ADAMW
lr_backbone=1e-5
lr_locEnc=1e-4
lr_locDec=1e-4
lr_segmentor=1e-4

## SGD
# lr_backbone=1e-3
# lr_locEnc=1e-2
# lr_locDec=1e-2
# # LR=1e-4 # 0.5
# lr_segmentor=1e-2

## AdamW and SGD
# lr_backbone=1e-5 ## FOR Localization AdamW
# lr_backbone2=1e-3 ## FOR Classification & Segmentation SGD
# lr_locEnc=1e-4
# lr_locDec=1e-4
# lr_segmentor=1e-2

# lr_backbone=1e-5 ## FOR Localization AdamW
# lr_backbone2=1e-5 ## FOR Classification & Segmentation SGD
# lr_locEnc=1e-4
# lr_locDec=1e-4
# lr_segmentor=1e-3

# lr_backbone=1e-5 ## FOR Localization AdamW
# lr_backbone2=1e-5 ## FOR Classification & Segmentation SGD
# lr_locEnc=1e-4
# lr_locDec=1e-4
# lr_segmentor=1e-4


BATCHSIZE=24
num_workers=12
INIT=ark
total_epochs=2000
opt=adamw # sgd adamw
SERVER=SOL
EMAMODE=True_Epoch

# cyclictask=chexpertCLS_nihchestxray14CLS_vindrcxrCLS_nihshenzenCLS_mimic2CLS_tbx11kCLS_node21CLS_candidptxCLS_rsnapneumoniaCLS_chestxdetCLS_siimacrCLS_tbx11kLOC_node21LOC_candidptxLOC_rsnapneumoniaLOC_chestxdetLOC_siimacrLOC_candidptxSEG_chestxdetSEG_siimacrSEG
cyclictask=nihchestxray14CLS_node21CLS_node21LOC_candidptxCLS_candidptxLOC_candidptxSEG
# cls_loc_seg | seg | loc | loc_seg | cls_loc | cls
# --debugOnlyTest --saveAllModel --lockrelease --debug

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29501

# RESUME=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run104_Ark6F6_ClsLocSeg_b24_AdamW_LockReleaseAll_RCons_1LocDec/ckpt_E930_TH9.pth
# --resume $RESUME

# ~/.conda/envs/tf-tnt-gpu2/bin/python 
~/.conda/envs/tf-tnt-gpu2/bin/python main_Consolidated.py --taskcomponent foundation_x5_pretraining --train --numClasses 1 --dataset_file $DATASETFILE --classification_dataset $DATASETFILE --num_workers $num_workers --coco_path $coco_path --weight-decay 0.0001 \
	--output_dir $LOGFILE -c $CONFIGFILE --imgsize $IMGSIZE --backbonemodel $BACKBONEMODEL --init $INIT --total_epochs $total_epochs --batch_size $BATCHSIZE --opt $opt \
	--finetune_ignore label_enc.weight class_embed \
	--backbone_dir $backbone_dir --lr_backbone $lr_backbone --lr_locEnc $lr_locEnc --lr_locDec $lr_locDec  --lr_segmentor $lr_segmentor \
	--cyclictask $cyclictask --serverC $SERVER --modelEMA $EMAMODE --lockrelease --saveAllModel \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0

