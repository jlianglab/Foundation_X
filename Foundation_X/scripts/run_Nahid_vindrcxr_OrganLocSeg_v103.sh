#!/bin/bash
#SBATCH --job-name=SLURM_SwinB_IntegratedModel_run101_Nahid_VinDrCXR_HLLRLsegmentation

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
#SBATCH --mail-user=ssiingh@asu.edu

module load mamba/latest
source activate tf-tnt-gpu2
# module load gcc/7.2.0
# module load anaconda/py3
# source activate tf-detr_gpu


# # Swin-L + DINO
CONFIGFILE=config/DINO/DINO_4scale_swinBASE.py
LOGFILE=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/Integrated_Model_vindrcxr_OrganSeg_/run104_Nahid_VinDrCXR_HLLRL_LocSeg_FuF_m0.80_noLayerNorm_yesFeatureNorm_noL2Reg
backbone_dir=/data/jliang12/dongaoma/Ark_models/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar

BACKBONEMODEL=Swin-B # Swin-T, Swin-B, Swin-L
IMGSIZE=224 # 448
coco_path=/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/
# coco_path=/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/

### 1e-1 = 0.1
### 1e-2 = 0.01
### 1e-3 = 0.001
### 1e-4 = 0.0001
### 1e-5 = 0.00001

lr_backbone=1e-5
lr_locEnc=1e-4
lr_locDec=1e-4
# LR=1e-4 # 0.5
lr_segmentor=1e-4

BATCHSIZE=2
num_workers=8
INIT=ark
total_epochs=500
opt=adamw
SERVER=SOL
cyclictask=heart_leftlung_rightlung
EMAMODE=True_Epoch
# --resume $RESUME --debugOnlyTest --saveAllModel --debug

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29503

# RESUME=/scratch/nuislam/Model_Checkpoints/IntegratedModel_DINOpipeline/Swin-B_224_Ark6_UperNet_FineTune_vindrcxr_OrganLocSeg_/run101_SOL_LocSeg_TrStg1_LocMultiHead_L2Reg_MultiSeq_EncDecLoss_HLLR/ckpt_E197_TH7.pth

# export CUDA_VISIBLE_DEVICES=1 && python main.py --taskcomponent detect_segmentation_cyclic_v2 --train --numClasses 1 --num_workers $num_workers --coco_path $coco_path --weight-decay 0.0001 \
# 	--output_dir $LOGFILE -c $CONFIGFILE --imgsize $IMGSIZE --backbonemodel $BACKBONEMODEL --init $INIT --total_epochs $total_epochs --batch_size $BATCHSIZE --opt $opt \
# 	--finetune_ignore label_enc.weight class_embed \
# 	--backbone_dir $backbone_dir --lr_backbone $lr_backbone --lr_locEnc $lr_locEnc --lr_locDec $lr_locDec  --lr_segmentor $lr_segmentor \
# 	--cyclictask $cyclictask --serverC $SERVER --modelEMA $EMAMODE --resume $RESUME \
# 	--options dn_scalar=100 embed_init_tgt=TRUE \
# 	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
# 	dn_box_noise_scale=1.0


# export CUDA_VISIBLE_DEVICES=0 &&
# ~/.conda/envs/tf-tnt-gpu/bin/python
# ~/.conda/envs/tf-detr_gpu/bin/python
export CUDA_VISIBLE_DEVICES=0 && ~/.conda/envs/tf-tnt-gpu2/bin/python main.py --taskcomponent detect_segmentation_cyclic_v2 --train --numClasses 1 --num_workers $num_workers --coco_path $coco_path --weight-decay 0.0001 \
	--output_dir $LOGFILE -c $CONFIGFILE --imgsize $IMGSIZE --backbonemodel $BACKBONEMODEL --init $INIT --total_epochs $total_epochs --batch_size $BATCHSIZE --opt $opt \
	--finetune_ignore label_enc.weight class_embed \
	--backbone_dir $backbone_dir --lr_backbone $lr_backbone --lr_locEnc $lr_locEnc --lr_locDec $lr_locDec  --lr_segmentor $lr_segmentor \
	--cyclictask $cyclictask --serverC $SERVER --modelEMA $EMAMODE --saveAllModel \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0

