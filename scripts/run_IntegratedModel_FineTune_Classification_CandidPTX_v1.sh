#!/bin/bash
#SBATCH --job-name=SLURM_SwinB_IntegratedModel_run107_F6_allLocSeg_b20_

#SBATCH -N 1
#SBATCH -G a100:1
#SBATCH -c 12
##SBATCH --exclusive
#SBATCH --mem=80G
#SBATCH -p general
#SBATCH -t 3-00:00:00
#SBATCH -q public

#SBATCH -o %x_slurm_%j.out     
#SBATCH -e %xslurm_%j.err      
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=nuislam@asu.edu

module load mamba/latest
module load cuda-11.7.0-gcc-11.2.0
source activate tf-tnt-gpu2


# # Swin-L + DINO
CONFIGFILE=config/DINO/DINO_4scale_swinBASE.py
LOGFILE=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/FineTune_CANDIDptx/FineTune_e632Teacher_CANDIDptx_Classification/FineTune_e632Teacher_CANDIDptx_Classification_run01_1e-5
backbone_dir=/data/jliang12/dongaoma/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar

BACKBONEMODEL=Swin-B # Swin-T, Swin-B, Swin-L
IMGSIZE=224 # 448
coco_path=/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/
DATASETFILE=foundation6Ark6_datasets

### 1e-1 = 0.1
### 1e-2 = 0.01
### 1e-3 = 0.001
### 1e-4 = 0.0001
### 1e-5 = 0.00001

## ADAMW
# lr_backbone=1e-5
# lr_locEnc=1e-4
# lr_locDec=1e-4
# lr_segmentor=1e-4

## ADAMW -- FOR FINETUNING
lr_backbone=1e-5
lr_locEnc=1e-5
lr_locDec=1e-5
lr_segmentor=1e-5



## SGD - FOR FINETUNING
# lr_backbone=1e-4
# lr_locEnc=1e-4
# lr_locDec=1e-4
# lr_segmentor=1e-4

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
num_workers=6
INIT=ark
total_epochs=25
opt=sgd # sgd adamw
SERVER=SOL
# EMAMODE=True_Epoch
# cls_loc_seg | seg | loc | loc_seg | cls_loc | cls
# --resume $RESUME --debugOnlyTest --saveAllModel --debug



# ### 'model' or 'teacher_model'
# ### CLASSIFICATION TASK FINETUNING
### ----------------------------------- ###
# cyclictask=clsFT_TESTcls_chexpertCLS
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E224_TH3.pth
# FXMODEL=teacher_model

# cyclictask=clsFT_TESTcls_nihchestxray14CLS
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E224_TH3.pth
# FXMODEL=teacher_model
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E477_TH16.pth  # ckpt_E362_TH1.pth
# FXMODEL=model
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E362_TH1.pth 
# FXMODEL=model
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E364_TH3.pth
# FXMODEL=teacher_model

# cyclictask=clsFT_TESTcls_vindrcxrCLS
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E224_TH3.pth
# FXMODEL=teacher_model

# cyclictask=clsFT_TESTcls_nihshenzenCLS
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E224_TH3.pth
# FXMODEL=teacher_model

# cyclictask=clsFT_TESTcls_mimic2CLS
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E224_TH3.pth
# FXMODEL=teacher_model

# cyclictask=clsFT_TESTcls_tbx11kCLS
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E224_TH3.pth # ckpt_E844_TH3.pth
# FXMODEL=teacher_model
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E844_TH3.pth # ckpt_E844_TH3.pth
# FXMODEL=model

# cyclictask=clsFT_TESTcls_node21CLS
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E224_TH3.pth
# FXMODEL=teacher_model

cyclictask=clsFT_TESTcls_candidptxCLS
foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E632_TH11.pth #ckpt_E632_TH11.pth
FXMODEL=teacher_model

# cyclictask=clsFT_TESTcls_rsnapneumoniaCLS
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E224_TH3.pth
# FXMODEL=teacher_model

# cyclictask=clsFT_TESTcls_chestxdetCLS
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E224_TH3.pth
# FXMODEL=teacher_model
# foundationX=/mnt/dfs/nuislam/Projects/IntegratedModel_GitHubV/Model_Checkpoints/Pretrained_Checkpoints/IntegratedModel_FoundationX_Checkpoint/ckpt_E477_TH16.pth 
# FXMODEL=model

# cyclictask=clsFT_TESTcls_siimacrCLS
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E224_TH3.pth
# FXMODEL=teacher_model


# ### LOCALIZATION TASK FINETUNING
### ----------------------------------- ###
# cyclictask=locFT_TESTloc_tbx11kLOC
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E308_TH7.pth
# FXMODEL=teacher_model

# cyclictask=locFT_TESTloc_node21LOC
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E270_TH9.pth
# FXMODEL=teacher_model

# cyclictask=locFT_TESTloc_candidptxLOC
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E224_TH3.pth
# FXMODEL=teacher_model

# cyclictask=locFT_TESTloc_rsnapneumoniaLOC
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E280_TH19.pth
# FXMODEL=model

# cyclictask=locFT_TESTloc_chestxdetLOC
# # foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E317_TH16.pth
# # FXMODEL=teacher_model
# # foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E424_TH3.pth
# foundationX=/mnt/dfs/nuislam/Projects/IntegratedModel_GitHubV/Model_Checkpoints/Pretrained_Checkpoints/IntegratedModel_FoundationX_Checkpoint/ckpt_E477_TH16.pth 
# FXMODEL=model

# cyclictask=locFT_TESTloc_siimacrLOC
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E187_TH6.pth
# FXMODEL=teacher_model


# ### SEGMENTATION TASK FINETUNING
### ----------------------------------- ###
# cyclictask=segFT_TESTseg_candidptxSEG
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E224_TH3.pth
# FXMODEL=teacher_model

# cyclictask=segFT_TESTseg_chestxdetSEG
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E224_TH3.pth
# FXMODEL=teacher_model

# cyclictask=segFT_TESTseg_siimacrSEG
# foundationX=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/run101_Ark6F6_ClsLocSeg_b24_AdamW/ckpt_E224_TH3.pth
# FXMODEL=teacher_model




export MASTER_ADDR=127.0.0.2
export MASTER_PORT=29501

sleep 5
LOGFILE=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/IntegratedModel_FoundationX3/FineTune_CANDIDptx/FineTune_e632Teacher_CANDIDptx_Classification/FineTune_e632Teacher_CANDIDptx_Classification_run03_1e-5
~/.conda/envs/tf-tnt-gpu2/bin/python main_NAD_finetune.py --taskcomponent foundation_x3_FineTuning --train --numClasses 1 --dataset_file $DATASETFILE --classification_dataset $DATASETFILE --num_workers $num_workers --coco_path $coco_path --weight-decay 0.0001 \
	--output_dir $LOGFILE -c $CONFIGFILE --imgsize $IMGSIZE --backbonemodel $BACKBONEMODEL --init $INIT --total_epochs $total_epochs --batch_size $BATCHSIZE --opt $opt \
	--finetune_ignore label_enc.weight class_embed \
	--backbone_dir $backbone_dir --lr_backbone $lr_backbone --lr_locEnc $lr_locEnc --lr_locDec $lr_locDec  --lr_segmentor $lr_segmentor \
	--cyclictask $cyclictask --serverC $SERVER --foundationX $foundationX --foundationXMODEL $FXMODEL \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0
