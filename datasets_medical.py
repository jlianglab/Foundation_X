# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved. ##
from email.mime import image
import torch.nn as nn
from os.path import isfile, join
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import random
import pydicom as dicom
import cv2
from einops import rearrange
from torchvision import transforms
import os
import json
import csv
import copy
# import SimpleITK as sitk
from tqdm import tqdm

import albumentations
from albumentations import Compose, HorizontalFlip, Normalize, VerticalFlip, Rotate, Resize, ShiftScaleRotate, OneOf, GridDistortion, OpticalDistortion, \
    ElasticTransform, IAAAdditiveGaussianNoise, GaussNoise, MedianBlur,  Blur, CoarseDropout,RandomBrightnessContrast,RandomGamma,RandomSizedCrop, ToFloat
from datasets import build_dataset
from albumentations.pytorch import ToTensorV2

from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomBrightnessContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise,CenterCrop,
    IAAAdditiveGaussianNoise,GaussNoise,OpticalDistortion,RandomSizedCrop, RandomResizedCrop, Normalize
)

## https://github.com/jlianglab/BenchmarkTransformers/blob/main/dataloader.py
def build_transform_classification(normalize, crop_size=224, resize=256, mode="train", test_augment=True):
    transformations_list = []

    if normalize.lower() == "imagenet":
      normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    elif normalize.lower() == "chestx-ray":
      normalize = transforms.Normalize([0.5056, 0.5056, 0.5056], [0.252, 0.252, 0.252])
    elif normalize.lower() == "none":
      normalize = None
    else:
      print("mean and std for [{}] dataset do not exist!".format(normalize))
      exit(-1)
    if mode == "train":
      transformations_list.append(transforms.RandomResizedCrop(crop_size))
      transformations_list.append(transforms.RandomHorizontalFlip())
      transformations_list.append(transforms.RandomRotation(7))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "valid":
      transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.CenterCrop(crop_size))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    elif mode == "test":
      if test_augment:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
          transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
          transformations_list.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
      else:
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.CenterCrop(crop_size))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
          transformations_list.append(normalize)
    elif mode == "test2":
    #   transformations_list.append(transforms.Resize((resize, resize)))
      transformations_list.append(transforms.Resize((crop_size, crop_size)))
      transformations_list.append(transforms.ToTensor())
      if normalize is not None:
        transformations_list.append(normalize)
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence

def build_ts_transformations():
    AUGMENTATIONS = Compose([
      RandomResizedCrop(height=224, width=224),
      ShiftScaleRotate(rotate_limit=10),
      OneOf([
          RandomBrightnessContrast(),
          RandomGamma(),
           ], p=0.3),
    ])
    return AUGMENTATIONS


# def build_transform_segmentation():
#   AUGMENTATIONS_TRAIN = Compose([
#     # HorizontalFlip(p=0.5),
#     OneOf([
#         RandomBrightnessContrast(),
#         RandomGamma(),
#          ], p=0.3),
#     OneOf([
#         ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
#         GridDistortion(),
#         OpticalDistortion(distort_limit=2, shift_limit=0.5),
#         ], p=0.3),
#     RandomSizedCrop(min_max_height=(315, 448), height=448, width=448,p=0.25),
#     ToFloat(max_value=1)
#     ],p=1)
#
#   return AUGMENTATIONS_TRAIN
def build_transform_segmentation():
    pass

#__________________________________________SIIM Pneumothorax segmentation dataset --------------------------------------------------
class PXSDataset(Dataset):

    def __init__(self, image_path_file, image_size=(448,448),mode= "train"):
        self.img_list = []
        self.img_label = []
        self.image_size = image_size

        self.mode = mode
        self.transformSequence = {
            'train': Compose([
                # HorizontalFlip(p=0.5),
                OneOf([
                    RandomBrightnessContrast(),
                    RandomGamma(),
                ], p=0.3),
                OneOf([
                    ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                    GridDistortion(),
                    OpticalDistortion(distort_limit=2, shift_limit=0.5),
                ], p=0.3),
                RandomSizedCrop(min_max_height=(int(0.7*self.image_size[0]), self.image_size[1]),
                                height=self.image_size[0], width=self.image_size[1],p=0.25),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
                ToTensorV2()
            ],p=1),
            'val': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
                ToTensorV2()
            ])
        }

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory, line.split(',')[0]))
                    self.img_label.append(join(pathImageDirectory, line.split(',')[1]))
                    line = fileDescriptor.readline().strip()


    def __len__(self):
        return len(self.img_list)

    def rle2mask(self,rle, width, height):
        mask = np.zeros(width * height)
        if rle == "-1":
            return mask.reshape(width, height)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position + lengths[index]] = 255
            current_position += lengths[index]

        return mask.reshape(width, height).T

    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]

        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        maskData = cv2.resize(cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)


        dic = self.transformSequence[self.mode](image=imageData, mask=maskData)
        img = dic['image']
        mask = (dic['mask'])/255.

        return img, mask



class MontgomeryDataset(Dataset):

    def __init__(self, image_path_file, image_size=(448,448), mode= "train"):

        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.mode = mode
        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'val': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }
        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory + "/CXR_png", line))
                    self.img_label.append(
                        (join(pathImageDirectory+"/ManualMask/leftMask", line),(join(pathImageDirectory+"/ManualMask/rightMask", line)))
                         )
                    line = fileDescriptor.readline().strip()


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]

        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        leftMaskData = cv2.resize(cv2.imread(maskPath[0],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)
        rightMaskData = cv2.resize(cv2.imread(maskPath[1],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)

        maskData = leftMaskData + rightMaskData
        maskData[maskData>0] =255
        maskData = maskData/255
        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode](image=imageData, mask=maskData)
        img = dic['image']
        mask = (dic['mask'])

        return img, mask


class JSRTLeftLungDataset(Dataset):
    def __init__(self, image_path_file , image_size=(448,448), mode="train"):
        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.mode = mode

        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'val': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory + "/images", line+".IMG.png"))
                    self.img_label.append(
                        (join(pathImageDirectory+"/masks/left_lung_png", line+".png"))
                         )
                    line = fileDescriptor.readline().strip()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]

        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        leftMaskData = cv2.resize(cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)
        # rightMaskData = cv2.resize(cv2.imread(maskPath[1],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)

        # maskData = leftMaskData + rightMaskData
        maskData = leftMaskData
        maskData[maskData>0] =255
        maskData = maskData/255

        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode](image=imageData, mask=maskData)
        img = dic['image']
        mask = (dic['mask'])

        return img, mask


class JSRTLungDataset(Dataset):

    def __init__(self, image_path_file , image_size=(448,448), mode="train"):

        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.mode = mode


        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'val': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory + "/images", line+".IMG.png"))
                    self.img_label.append(
                        (join(pathImageDirectory+"/masks/left_lung_png", line+".png"),(join(pathImageDirectory+"/masks/right_lung_png", line+".png")))
                         )
                    line = fileDescriptor.readline().strip()


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]



        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255



        leftMaskData = cv2.resize(cv2.imread(maskPath[0],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)
        rightMaskData = cv2.resize(cv2.imread(maskPath[1],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)

        maskData = leftMaskData + rightMaskData
        maskData[maskData>0] =255
        maskData = maskData/255

        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode](image=imageData, mask=maskData)
        img = dic['image']
        mask = (dic['mask'])

        return img, mask


class JSRTClavicleDataset(Dataset):

    def __init__(self, image_path_file , image_size=(448,448), mode="train", ann=None):

        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.mode = mode
        self.ann = ann
        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'val': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }



        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory + "/images", line+".IMG.png"))
                    self.img_label.append(
                        (join(pathImageDirectory+"/masks/left_clavicle_png/", line+".png"),(join(pathImageDirectory+"/masks/right_clavicle_png/", line+".png")))
                         )
                    line = fileDescriptor.readline().strip()

        if self.mode == "train" and self.ann != None:  ## For few shot learning
            # Shuffle indices to randomly pick samples
            indices = list(range(len(self.img_list)))
            random.shuffle(indices)
            
            # Select 'ann' random indices from shuffled indices
            selected_indices = indices[:ann]
            
            # Extract corresponding samples from ListA and ListB
            selected_samplesA = [self.img_list[i] for i in selected_indices]
            selected_samplesB = [self.img_label[i] for i in selected_indices]
            self.img_list = selected_samplesA
            self.img_label = selected_samplesB


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]

        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        leftMaskData = cv2.resize(cv2.imread(maskPath[0],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)
        rightMaskData = cv2.resize(cv2.imread(maskPath[1],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)

        maskData = leftMaskData + rightMaskData
        maskData[maskData>0] =255
        maskData = maskData/255
        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode](image=imageData, mask=maskData)
        img = dic['image']
        mask = (dic['mask'])

        return img, mask

class JSRTHeartDataset(Dataset):

    def __init__(self, image_path_file , image_size=(448,448), mode="train"):
        self.img_list = []
        self.img_label = []
        self.image_size = image_size

        self.mode = mode
        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'val': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }



        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory + "/images", line+".IMG.png"))
                    self.img_label.append(join(pathImageDirectory+"/masks/heart_png/", line+".png"))
                    line = fileDescriptor.readline().strip()


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]

        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        maskData = cv2.resize(cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)

        maskData[maskData>0] =255
        maskData = maskData/255
        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode](image=imageData, mask=maskData)
        img = dic['image']
        mask = (dic['mask'])

        return img, mask


class VindrCXRHLLRL_3chDataset(Dataset): ## Segmentation Dataloader for 1H 3Ch Heart-LeftLung-RightLung

    def __init__(self, image_path_file , image_size=(448,448), mode="train", ann=-1):

        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.mode = mode
        self.ann = ann

        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'test': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }

        if mode == "train":
            mode_images = "/train_jpeg"
            mode_masksL = "/masks_organ_train/vindrcxr_mask_leftLung/"
            mode_masksR = "/masks_organ_train/vindrcxr_mask_rightLung/"
            mode_masksH = "/masks_organ_train/vindrcxr_mask_heart/"

        elif mode == "test":
            mode_images = "/test_jpeg"
            mode_masksL = "/masks_organ_test/vindrcxr_mask_leftLung/"
            mode_masksR = "/masks_organ_test/vindrcxr_mask_rightLung/"
            mode_masksH = "/masks_organ_test/vindrcxr_mask_heart/"

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory + mode_images, line+".jpeg"))
                    self.img_label.append(
                        ( join(pathImageDirectory+mode_masksL, line+".png"), (join(pathImageDirectory+mode_masksR, line+".png")), (join(pathImageDirectory+mode_masksH, line+".png")) )
                         )
                    line = fileDescriptor.readline().strip()

        if self.mode == "train" and self.ann == 7500: ## last 7500 data for segmentation and first to 7500 is for localization
            self.img_list.sort()
            self.img_label.sort()
            self.img_list = self.img_list[7500:15000]
            self.img_label = self.img_label[7500:15000]

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]

        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        leftMaskData = cv2.resize(cv2.imread(maskPath[0],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)
        rightMaskData = cv2.resize(cv2.imread(maskPath[1],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)
        heartMaskData = cv2.resize(cv2.imread(maskPath[2],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)

        ## Add all mask together -- OLD IDEA
        # maskData = leftMaskData + rightMaskData + heartMaskData
        # maskData[maskData>0] = 255
        # maskData = maskData/255

        maskData = np.zeros( (3,224,224), dtype = np.uint8 )
        maskData[0,:,:] = heartMaskData
        maskData[1,:,:] = leftMaskData
        maskData[2,:,:] = rightMaskData
        maskData = maskData/255

        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode](image=imageData, mask=maskData)
        img = dic['image']
        mask = (dic['mask'])

        return img, mask ## mask should be CH x H x W


class VindrCXRLungDataset(Dataset):

    def __init__(self, image_path_file , image_size=(448,448), mode="train"):

        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.mode = mode


        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'test': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }


        if mode == "train":
            mode_images = "/train_jpeg"
            mode_masksL = "/masks_organ_train/vindrcxr_mask_leftLung/"
            mode_masksR = "/masks_organ_train/vindrcxr_mask_rightLung/"
        elif mode == "test":
            mode_images = "/test_jpeg"
            mode_masksL = "/masks_organ_test/vindrcxr_mask_leftLung/"
            mode_masksR = "/masks_organ_test/vindrcxr_mask_rightLung/"

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory + mode_images, line+".jpeg"))
                    self.img_label.append(
                        (join(pathImageDirectory+mode_masksL, line+".png"),(join(pathImageDirectory+mode_masksR, line+".png")))
                         )
                    line = fileDescriptor.readline().strip()


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]



        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255



        leftMaskData = cv2.resize(cv2.imread(maskPath[0],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)
        rightMaskData = cv2.resize(cv2.imread(maskPath[1],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)

        maskData = leftMaskData + rightMaskData
        maskData[maskData>0] =255
        maskData = maskData/255

        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode](image=imageData, mask=maskData)
        img = dic['image']
        mask = (dic['mask'])

        return img, mask

class VindrCXRLeftLungDataset(Dataset):

    def __init__(self, image_path_file , image_size=(448,448), ann=-1, mode="train"):

        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.mode = mode
        self.ann = ann


        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'test': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }


        if self.mode == "train" or self.mode == "trainA" or self.mode == "trainB":
            mode_images = "/train_jpeg"
            mode_masksL = "/masks_organ_train/vindrcxr_mask_leftLung/"
        elif self.mode == "test":
            mode_images = "/test_jpeg"
            mode_masksL = "/masks_organ_test/vindrcxr_mask_leftLung/"

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory + mode_images, line+".jpeg"))
                    self.img_label.append(
                        (join(pathImageDirectory+mode_masksL, line+".png"))
                         )
                    line = fileDescriptor.readline().strip()

        if self.mode == "train" and self.ann == 2500:
            self.img_list.sort()
            self.img_label.sort()
            self.img_list = self.img_list[10000:12500]
            self.img_label = self.img_label[10000:12500]
        if self.mode == "trainA" and self.ann == 1250:
            self.img_list.sort()
            self.img_label.sort()
            self.img_list = self.img_list[10000:11250]
            self.img_label = self.img_label[10000:11250]
        if self.mode == "trainB" and self.ann == 1250:
            self.img_list.sort()
            self.img_label.sort()
            self.img_list = self.img_list[11250:12500]
            self.img_label = self.img_label[11250:12500]


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]



        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255



        leftMaskData = cv2.resize(cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)

        maskData = leftMaskData
        maskData[maskData>0] =255
        maskData = maskData/255

        imageData = imageData.transpose((1, 2, 0))
        if self.mode == "train" or self.mode == "trainA" or self.mode == "trainB":
            dic = self.transformSequence["train"](image=imageData, mask=maskData)
            dic_nonAug = self.transformSequence["test"](image=imageData, mask=maskData)
        else:
            dic = self.transformSequence[self.mode](image=imageData, mask=maskData) # TEST
        img = dic['image']
        mask = (dic['mask'])

        if self.mode == "train":
            return img, mask, dic_nonAug['image'], (dic_nonAug['mask'])
        else:
            return img, mask
        # return img, mask

class VindrCXRRightLungDataset(Dataset):

    def __init__(self, image_path_file , image_size=(448,448), ann=-1, mode="train"):

        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.mode = mode
        self.ann = ann


        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'test': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }


        if self.mode == "train" or self.mode == "trainA" or self.mode == "trainB":
            mode_images = "/train_jpeg"
            mode_masksR = "/masks_organ_train/vindrcxr_mask_rightLung/"
        elif self.mode == "test":
            mode_images = "/test_jpeg"
            mode_masksR = "/masks_organ_test/vindrcxr_mask_rightLung/"

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory + mode_images, line+".jpeg"))
                    self.img_label.append(
                        ((join(pathImageDirectory+mode_masksR, line+".png")))
                         )
                    line = fileDescriptor.readline().strip()
        # print(self.mode, self.ann)
        # exit(0)
        if self.mode == "train" and self.ann == 2500:
            self.img_list.sort()
            self.img_label.sort()
            self.img_list = self.img_list[12500:15000]
            self.img_label = self.img_label[12500:15000]
        if self.mode == "trainA" and self.ann == 1250:
            self.img_list.sort()
            self.img_label.sort()
            self.img_list = self.img_list[12500:13750]
            self.img_label = self.img_label[12500:13750]
        if self.mode == "trainB" and self.ann == 1250:
            self.img_list.sort()
            self.img_label.sort()
            self.img_list = self.img_list[13750:15000]
            self.img_label = self.img_label[13750:15000]


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]



        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        rightMaskData = cv2.resize(cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)

        maskData = rightMaskData
        maskData[maskData>0] =255
        maskData = maskData/255

        imageData = imageData.transpose((1, 2, 0))
        if self.mode == "train" or self.mode == "trainA" or self.mode == "trainB":
            dic = self.transformSequence["train"](image=imageData, mask=maskData)
            dic_nonAug = self.transformSequence["test"](image=imageData, mask=maskData)
        else:
            dic = self.transformSequence[self.mode](image=imageData, mask=maskData) # TEST
        img = dic['image']
        mask = (dic['mask'])

        if self.mode == "train":
            return img, mask, dic_nonAug['image'], (dic_nonAug['mask'])
        else:
            return img, mask
        # return img, mask



class VindrCXRHeartDataset(Dataset):

    def __init__(self, image_path_file , image_size=(448,448), ann=-1, mode="train"):
        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.ann = ann

        self.mode = mode
        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'test': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }


        if self.mode == "train" or self.mode == "trainA" or self.mode == "trainB":
            mode_images = "/train_jpeg"
            mode_masks = "/masks_organ_train/vindrcxr_mask_heart/"
        elif self.mode == "test":
            mode_images = "/test_jpeg"
            mode_masks = "/masks_organ_test/vindrcxr_mask_heart/"

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory + mode_images, line + ".jpeg"))
                    self.img_label.append(join(pathImageDirectory+ mode_masks, line + ".png"))
                    line = fileDescriptor.readline().strip()
        if self.mode == "train" and self.ann == 2500:
            self.img_list.sort()
            self.img_label.sort()
            self.img_list = self.img_list[7500:10000]
            self.img_label = self.img_label[7500:10000]
        if self.mode == "trainA" and self.ann == 1250:
            self.img_list.sort()
            self.img_label.sort()
            self.img_list = self.img_list[7500:8750]
            self.img_label = self.img_label[7500:8750]
        if self.mode == "trainB" and self.ann == 1250:
            self.img_list.sort()
            self.img_label.sort()
            self.img_list = self.img_list[8750:10000]
            self.img_label = self.img_label[8750:10000]


    def __len__(self):
        return len(self.img_list)


    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]

        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        maskData = cv2.resize(cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)

        maskData[maskData>0] =255
        maskData = maskData/255
        imageData = imageData.transpose((1, 2, 0))
        if self.mode == "train" or self.mode == "trainA" or self.mode == "trainB":
            dic = self.transformSequence["train"](image=imageData, mask=maskData)
            dic_nonAug = self.transformSequence["test"](image=imageData, mask=maskData)
        else:
            dic = self.transformSequence[self.mode](image=imageData, mask=maskData) # TEST
        img = dic['image']
        mask = (dic['mask'])

        if self.mode == "train":
            return img, mask, dic_nonAug['image'], (dic_nonAug['mask'])
        else:
            return img, mask
        # return img, mask

class ChestXDetDataset(Dataset): # only Segmentation

    def __init__(self, image_path_file , image_size=(448,448), mode="train"):

        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.mode = mode


        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'val': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ]),
            'test': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }

        if mode == "train":
            mode_images = "/train"
            mode_masks = "/train_masks"
            ext = ".png"
        elif mode == "val":
            mode_images = "/train"
            mode_masks = "/train_masks"
            ext = ".png"
        elif mode == "test":
            mode_images = "/test"
            mode_masks = "/test_masks"
            ext = ".png"

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory + mode_images, line))
                    self.img_label.append( ( join(pathImageDirectory + mode_masks, line) ) )
                    line = fileDescriptor.readline().strip()

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]

        # print("[Dataloader Info.] Image:", self.img_list[idx])
        # print("[Dataloader Info.] Image:", self.img_label[idx])

        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255

        leftMaskData = cv2.resize(cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)
        # rightMaskData = cv2.resize(cv2.imread(maskPath[1],cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)

        maskData = leftMaskData
        maskData[maskData>0] =255
        maskData = maskData/255

        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode](image=imageData, mask=maskData)
        img = dic['image']
        mask = (dic['mask'])
        mask = mask.unsqueeze(0)

        return img, mask

class ChestXDet_13Diseases(Dataset): ## From DongAo
    def _init_(self, images_path, split, augment, image_size=(224,224), anno_percent=100, normalization=None):
        self.augmentation = augment
        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.normalization = normalization
        self.disease_labels =  ["Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "Diffuse Nodule", "Effusion", 
                                "Emphysema", "Fibrosis", "Fracture", "Mass", "Nodule", "Pleural Thickening", "Pneumothorax"]
        
        gt_file = os.path.join(images_path, f"ChestX_Det_{split}.json")
        images_path = os.path.join(images_path, split)
        with open(gt_file, 'r') as json_file:
           gt_data = json.load(json_file)
           for d in gt_data:
                fname = d['file_name']
                self.img_list.append(os.path.join(images_path, fname))
                self.img_label.append(d)
        
        if anno_percent < 100:
          raise NotImplementedError
    

    def _getitem_(self, index):
        imagePath = self.img_list[index]
        imageData = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        labelAll = [np.zeros((imageData.shape[:2]), dtype=np.uint8) for _ in range(len(self.disease_labels))]

        imageLabel = self.img_label[index]
        polygons = imageLabel['polygons']
        for i, sym in enumerate(imageLabel['syms']):
           pts = np.array(polygons[i], np.int32)
           label = labelAll[self.disease_labels.index(sym)]
           labelAll[self.disease_labels.index(sym)] = cv2.fillPoly(label, [pts], 1)
        labelAll =  np.stack([cv2.resize(label, self.image_size,interpolation=cv2.INTER_AREA) for label in labelAll])
        mask = labelAll.transpose((1, 2, 0))

        image = cv2.resize(imageData,self.image_size, interpolation=cv2.INTER_AREA)
        
        if self.augmentation:
            augmented = self.augmentation(image=image, mask=mask)
            image=augmented['image']
            mask=augmented['mask']
            image=np.array(image) / 255.
        else:
            image = np.array(image) / 255.
        if self.normalization == "imagenet":
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            image = (image-mean)/std

        image = image.transpose((2, 0, 1)).astype('float32')
        mask = mask.transpose((2, 0, 1))

        return image, mask

    def _len_(self):
        return len(self.img_list)

class chestxdet_dataset(Dataset):  # only Segmentation ## From Anni
    def __init__(self,image_path,masks_path,image_size=(224,224), transforms=None, mode='train'):
        self.image_path = image_path
        self.masks_path = masks_path
        self.image_size = image_size
        self.mode = mode
        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'valid': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ]),
            'test': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }
        # self.transformSequence = {
        #     'train': Compose([
        #         # HorizontalFlip(p=0.5),
        #         OneOf([
        #             RandomBrightnessContrast(),
        #             RandomGamma(),
        #         ], p=0.3),
        #         OneOf([
        #             ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        #             GridDistortion(),
        #             OpticalDistortion(distort_limit=2, shift_limit=0.5),
        #         ], p=0.3),
        #         RandomSizedCrop(min_max_height=(int(0.7*self.image_size[0]), self.image_size[1]),
        #                         height=self.image_size[0], width=self.image_size[1],p=0.25),
        #         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
        #         ToTensorV2()
        #     ],p=1),
        #     'valid': Compose([
        #         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
        #         ToTensorV2()
        #     ]),
        #     'test': Compose([
        #         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
        #         ToTensorV2()
        #     ])
        # }

        images = os.listdir(self.image_path)
        masks = os.listdir(self.masks_path)
        
        self.img_dict = {}
        for i in images:
            temp = []
            for idx,j in enumerate(masks):
                if j.startswith(i.split(".")[0]) :
                    temp.append(j)
            self.img_dict[i] = temp

    def __getitem__(self,index):
        image_name = list(self.img_dict.keys())[index]
        image_path = os.path.join(self.image_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        masks_list = self.img_dict[image_name]
        mask = self.combine_mask(img,masks_list)

        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_AREA)
        # img = img/255
        # img = rearrange(img, 'h w c-> c h w')/255
        # mask = rearrange(mask, 'h w c-> c h w')
        

        # print(" -- [CHECK img]", self.mode, img.min(), img.max(), img.shape)
        # print(" -- [CHECK mask]", self.mode, mask.min(), mask.max(), mask.shape)
        dic = self.transformSequence[self.mode](image=img, mask=mask)
        img = dic['image']
        mask = (dic['mask'])
        # mask = mask.unsqueeze(0)
        mask = rearrange(mask, 'h w c-> c h w')
        img = img/255
        mask = mask/255
        # print("[CHECK ChestXDet Train Set] img", self.mode, img.min(), img.max(), "mask", mask.min(), mask.max(), mask.shape)
        return img, mask


    def combine_mask(self,img,masks_list):
        combined_mask = np.zeros( (224,224,len(masks_list)), dtype = np.uint8 )
        # combined_mask = np.zeros( (len(masks_list),img.shape[0],img.shape[1]), dtype = np.uint8 )
        # print("[CHECK] combined_mask", combined_mask.shape)

        for i, mask_file in enumerate(masks_list):
            mask_path = os.path.join(self.masks_path,mask_file)
            mask = cv2.resize(cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)
            # mask[mask==255]=1 ## was enable
            combined_mask[:,:,i] = mask
            
        return combined_mask

    def __len__(self):
        return(len(os.listdir(self.image_path)))
    


class ChestXDet_cls(Dataset): ## For Classification
  def __init__(self, images_path, file_path, augment, num_class=13, anno_percent=100): ## ChestX_det_train_NAD_v2  ChestX_det_test_NAD_v2
    self.img_list = []
    self.img_label = []
    self.disease_labels =  ["Atelectasis", "Calcification", "Cardiomegaly", "Consolidation", "Diffuse Nodule", "Effusion", "Emphysema", "Fibrosis", "Fracture", "Mass", "Nodule", "Pleural Thickening", "Pneumothorax"]
    self.augment = augment
    split = "train" if "train" in file_path else "test"
    gt_file = os.path.join(images_path, f"ChestX_det_{split}_NAD_v2.json")

    images_path = os.path.join(images_path, split)
    with open(gt_file, 'r') as json_file:
        gt_data = json.load(json_file)
        for d in gt_data["images"]:
            fname = d["file_name"]
            imageLabel = [0 for _ in range(num_class)]
            for anno in gt_data["annotations"]:
                if d["id"] == anno["image_id"]:
                    lb = anno["category_id"] - 1
                    imageLabel[lb] = 1

            self.img_list.append(os.path.join(images_path, fname))
            self.img_label.append(imageLabel)
    # print("-- CHECK -- ChestXDet Dataset Length:", len(self.img_list))

    if anno_percent < 100:
        raise NotImplementedError

  def __getitem__(self, index):
    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):
    return len(self.img_list)
  

## RSNA_Pneumonia Challenge Dataset Classification Dataloader
class RSNAPneumonia(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=3):
    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.diseases_LIST = ['Normal', 'No Lung Opacity/Not Normal', 'Lung Opacity']

    with open(file_path, "r") as fileDescriptor:
      line = True
      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.strip().split(' ')
          imagePath = os.path.join(images_path, lineItems[0])

          self.img_list.append(imagePath)
          self.img_label.append(int(lineItems[-1]))

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = np.zeros(3)
    imageLabel[self.img_label[index]] = 1
    imageLabel = torch.FloatTensor(imageLabel)
    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):
    return len(self.img_list)


## SIIM-ACR_PTX Classification Dataloader
class SIIMPTX(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=1):
    self.img_list = []
    self.img_label = []
    self.augment = augment

    split_folder = "test_jpeg" if "test" in file_path else "train_jpeg"

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, split_folder, lineItems[0]+'.dcm.jpeg')
          imageLabel = [int(lineItems[1])]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    # im_array = dicom.dcmread(imagePath).pixel_array
    # imageData = Image.fromarray(im_array).convert('RGB')
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):
    return len(self.img_list)
  

## CANDID-PTX Classification Dataloader
class CANDIDPTX(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=1):
    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(file_path, "r") as fileDescriptor:
      line = True
      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = [int(lineItems[1])]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    im_array = dicom.dcmread(imagePath).pixel_array
    imageData = Image.fromarray(im_array).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):
    return len(self.img_list)
  
## NIH Shenzhen CXR Dataset Classification Dataloader
class ShenzhenCXR(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=1, annotation_percent=100):
    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.train_augment = build_ts_transformations()
    self.diseases_LIST = ['TB']

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split(',')

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    return student_img, imageLabel
    return student_img, teacher_img, imageLabel

  def __len__(self):

    return len(self.img_list)


class VinDrRibCXRDataset(Dataset):
    def __init__(self, image_path_file, image_size, mode):
        self.pathImageDirectory, pathDatasetFile = image_path_file
        self.image_size = image_size
        self.mode = mode
        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                # HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'val': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }
        self.rib_labels =  ['R1','R2','R3','R4','R5','R6','R7','R8','R9','R10',
                           'L1','L2','L3','L4','L5','L6','L7','L8','L9','L10']
        f = open(pathDatasetFile)
        data= json.load(f)

        self.img_list = data['img']

        self.label_list = data

    def __getitem__(self, index):
        imagePath = self.img_list[str(index)]
        imageData = cv2.imread(os.path.join(self.pathImageDirectory, imagePath), cv2.IMREAD_COLOR)
        label0 = []
        for name in self.rib_labels:
            pts = self.label_list[name][str(index)]
            label = np.zeros((imageData.shape[:2]), dtype=np.uint8)
            if pts != 'None':
                pts = np.array([[[int(pt['x']), int(pt['y'])]] for pt in pts])
                label = cv2.fillPoly(label, [pts], 1)
                label = cv2.resize(label, self.image_size,interpolation=cv2.INTER_AREA)
            label0.append(label)
        label0 = np.stack(label0)
        label0 = label0.transpose((1, 2, 0))

        imageData = cv2.resize(imageData,self.image_size, interpolation=cv2.INTER_AREA)
        imageData = rearrange(imageData, 'h w c-> c h w')/255
        imageData = imageData.transpose((1, 2, 0))
        dic = self.transformSequence[self.mode] (image=imageData, mask=label0)
        img = dic['image']
        mask = (dic['mask'].permute(2, 0, 1))

        return img, mask



    def __len__(self):

        return len(self.img_list)




class ShenzhenDataset(Dataset):
    def __init__(self, pathImageDirectory, pathDatasetFile, cxr_embedding,image_size):
        self.img_list = []
        self.img_label = []
        self.cxr_embedding = cxr_embedding
        self.image_size = image_size
        self.diseases_LIST = ['TB']
        self.transformSequence_img = transforms.Compose([
            torch.from_numpy,
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.cxr_embedding:
            self.img_list = np.load(pathImageDirectory)
            with open(pathDatasetFile, "r") as fr:
                line = fr.readline().strip()
                while line:
                    lineItems = line.split()
                    imageLabel = lineItems[1:]
                    imageLabel = [int(i) for i in imageLabel]
                    self.img_label.append(imageLabel)
                    line = fr.readline()
            np.random.shuffle(self.img_list)
        else:
            with open(pathDatasetFile, "r") as fr:
                line = fr.readline().strip()
                while line:
                    lineItems = line.split()
                    imagePath = os.path.join(pathImageDirectory, lineItems[0])
                    imageLabel = lineItems[1:]
                    imageLabel = [int(i) for i in imageLabel]
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)
                    line = fr.readline()


    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor(self.img_label[index])
        if not self.cxr_embedding:
            imageData = cv2.resize(cv2.imread(imagePath, cv2.IMREAD_COLOR),( self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            imageData = rearrange(imageData, 'h w c-> c h w')/255
            imageData = self.transformSequence_img(imageData)
        else:
            imageData = imagePath
        return imageData, imageLabel
    def __len__(self):

        return len(self.img_list)


class VinDrCXRDataset(Dataset):
    def __init__(self, pathImageDirectory, pathDatasetFile, cxr_embedding,image_size):
        self.img_list = []
        self.img_label = []
        self.cxr_embedding = cxr_embedding
        self.image_size = image_size
        self.transformSequence_img = transforms.Compose([
            torch.from_numpy,
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.cxr_embedding:
            self.img_list = np.load(pathImageDirectory)
            with open(pathDatasetFile, "r") as fr:
                line = fr.readline().strip()
                while line:
                    lineItems = line.split()
                    imageLabel = lineItems[1:]
                    imageLabel = [int(i) for i in imageLabel]
                    self.img_label.append(imageLabel)
                    line = fr.readline()
            np.random.shuffle(self.img_list)
        else:
            with open(pathDatasetFile, "r") as fr:
                line = fr.readline().strip()
                while line:
                    lineItems = line.split()
                    imagePath = os.path.join(pathImageDirectory, lineItems[0]+".jpeg")
                    imageLabel = lineItems[1:]
                    imageLabel = [int(i) for i in imageLabel]
                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)
                    line = fr.readline()


    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageLabel = torch.FloatTensor(self.img_label[index])
        if not self.cxr_embedding:
            imageData = cv2.resize(cv2.imread(imagePath, cv2.IMREAD_COLOR),( self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            imageData = rearrange(imageData, 'h w c-> c h w')/255
            imageData = self.transformSequence_img(imageData)
        else:
            imageData = imagePath
        return imageData, imageLabel
    def __len__(self):

        return len(self.img_list)

## Classification on NODE21 Dataset
class NODE21(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=2, annotation_percent=100):
    self.img_list = []
    self.img_label = []
    self.augment = augment
    with open(file_path, "r") as fileDescriptor:
      line = True
      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, "images_png", lineItems[0])
          imageLabel = int(lineItems[1])
        #   print("CHECK", imagePath, imageLabel)
        #   imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    # imageLabel = torch.FloatTensor(self.img_label[index])
    imageLabel = self.img_label[index]
    # print("CHECK", imageData.shape, imageLabel.shape)

    if self.augment != None: imageData = self.augment(imageData)
    return imageData, imageLabel

  def __len__(self):
    return len(self.img_list)

## VinDr-CXR Dataset Classification Dataloader
class VinDrCXR(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=6, annotation_percent=100):
    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.train_augment = build_ts_transformations()
    self.diseases_LIST = ['PE', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']

    with open(file_path, "r") as fr:
      line = fr.readline().strip()
      while line:
        lineItems = line.split()
        imagePath = os.path.join(images_path, lineItems[0]+".jpeg")
        imageLabel = lineItems[1:]
        imageLabel = [int(i) for i in imageLabel]
        self.img_list.append(imagePath)
        self.img_label.append(imageLabel)
        line = fr.readline()

    if annotation_percent < 100:
      indexes = np.arange(len(self.img_list))
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]
    imageLabel = torch.FloatTensor(self.img_label[index])
    imageData = Image.open(imagePath).convert('RGB').resize((224,224))

    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    # return student_img, teacher_img, imageLabel
    return student_img, imageLabel

  def __len__(self):
    return len(self.img_list)
  

## MIMIC-II Dataset Classification Dataloader  
class MIMIC(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=14,
               uncertain_label="Ones", unknown_label=0, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.train_augment = build_ts_transformations()
    self.diseases_LIST = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    self.diseases_LIST_test = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              if self.uncertain_label == "Ones":
                label[i] = 1
              elif self.uncertain_label == "Zeros":
                label[i] = 0
              elif self.uncertain_label == "LSR-Ones":
                label[i] = random.uniform(0.55, 0.85)
              elif self.uncertain_label == "LSR-Zeros":
                label[i] = random.uniform(0, 0.3)
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        self.img_label.append(label)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index): 

    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])     

    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    # return student_img, teacher_img, imageLabel
    return student_img, imageLabel
  def __len__(self):
    return len(self.img_list)
  
## Classification on VinDr Dataset 
class VindrCXRClass(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=2, annotation_percent=100):
    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.diseases_LIST = ['PE', 'Lung tumor', 'Pneumonia', 'Tuberculosis', 'Other diseases', 'No finding']

    with open(file_path, "r") as fileDescriptor:
      line = True
      while line:
        line = fileDescriptor.readline()
        if line:
          lineItems = line.split()
         
          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:]
          imageLabel = [int(i) for i in imageLabel]
          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    # indexes = np.arange(len(self.img_list))
    # if annotation_percent < 100:
    #   random.Random(99).shuffle(indexes)
    #   num_data = int(indexes.shape[0] * annotation_percent / 100.0)
    #   indexes = indexes[:num_data]

    #   _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
    #   self.img_list = []
    #   self.img_label = []

    #   for i in indexes:
    #     self.img_list.append(_img_list[i])
    #     self.img_label.append(_img_label[i])

      

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])
    #imageLabel = self.img_label[index]
    # print("CHECK", imageData.shape, imageLabel.shape)

    if self.augment != None: 
      imageData = self.augment(imageData)
    return imageData,imageLabel

  def __len__(self):
    return len(self.img_list)


## Segmentation SIIM_Pneumothorax
class SIIM_PXSDataset(Dataset):
    def __init__(self, image_path_file, image_size=(448,448), mode= "train"):
        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.mode = mode
        self.transformSequence = {
            'train': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                HorizontalFlip(),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                # RandomGamma(), # new
                # ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03), # new
                # GridDistortion(), # new
                # OpticalDistortion(distort_limit=2, shift_limit=0.5), # new
                ToTensorV2()
            ]),
            'valid': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ]),
            'test': Compose([
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }
        # self.transformSequence = {
        #     'train': Compose([
        #         # HorizontalFlip(p=0.5),
        #         OneOf([
        #             RandomBrightnessContrast(),
        #             RandomGamma(),
        #         ], p=0.3),
        #         OneOf([
        #             ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        #             GridDistortion(),
        #             OpticalDistortion(distort_limit=2, shift_limit=0.5),
        #         ], p=0.3),
        #         RandomSizedCrop(min_max_height=(int(0.7*self.image_size[0]), self.image_size[1]),
        #                         height=self.image_size[0], width=self.image_size[1],p=0.25),
        #         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
        #         ToTensorV2()
        #     ],p=1),
        #     'valid': Compose([
        #         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
        #         ToTensorV2()
        #     ]),
        #     'test': Compose([
        #         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
        #         ToTensorV2()
        #     ])
        # }

        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory, line.split(',')[0]))
                    self.img_label.append(join(pathImageDirectory, line.split(',')[1]))
                    line = fileDescriptor.readline().strip()

    def __len__(self):
        return len(self.img_list)

    # used to generate mask from rle. All mask are pre-generated, not used in here
    def rle2mask(self,rle, width, height):
        mask = np.zeros(width * height)
        if rle == "-1":
            return mask.reshape(width, height)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position + lengths[index]] = 255
            current_position += lengths[index]
        return mask.reshape(width, height).T

    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskPath = self.img_label[idx]

        imageData = cv2.resize(cv2.imread(imagePath,cv2.IMREAD_COLOR),self.image_size, interpolation=cv2.INTER_AREA)
        #imageData = rearrange(imageData, 'h w c-> c h w')/255
        # imageData = imageData/255
        maskData = cv2.resize(cv2.imread(maskPath,cv2.IMREAD_GRAYSCALE),self.image_size, interpolation=cv2.INTER_AREA)
        # maskData = maskData/255

        # print(" -- [CHECK img]", self.mode, imageData.min(), imageData.max(), imageData.shape) ##  224x224x3
        # print(" -- [CHECK mask]", self.mode, maskData.min(), maskData.max(), maskData.shape) ## 224x224
        dic = self.transformSequence[self.mode](image=imageData, mask=maskData)
        img = dic['image']
        mask = (dic['mask'])
        mask = mask.unsqueeze(0)
        img = img/255
        mask = mask/255

        # print("image data:", img.shape, torch.min(img), torch.max(img))
        # print("maskdata:",mask.shape, torch.min(mask), torch.max(mask))
        # img = rearrange(img, 'c h w-> h w c')
        # img = img.numpy()
        # img = ((img - img.min()) * ( 1 / (img.max() - img.min()) * 255)).astype('uint8')
        # img = Image.fromarray(img)
        # img.save("{}.png".format(idx))
        #
        # maskData = Image.fromarray(mask.numpy()*255).convert("L")
        # maskData.save("{}_phx.png".format(idx))
        #
        # exit(0)
        # print(" -- [CHECK img] ---", self.mode, imageData.min(), imageData.max(), imageData.shape) ##  224x224x3
        # print(" -- [CHECK mask] --", self.mode, maskData.min(), maskData.max(), maskData.shape) ## 224x224
        return img, mask


## Segmentation CANDID-PTX dataset
class Candid_PTX_PXSDataset(Dataset):
    def __init__(self, image_path_file, image_size=(448,448), mode= "train"):
        self.img_list = []
        self.img_label = []
        self.image_size = image_size
        self.mode = mode
        self.transformSequence = {
            'train': Compose([
                Resize(self.image_size[0], self.image_size[1], cv2.INTER_AREA),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ShiftScaleRotate(rotate_limit=10),
                RandomBrightnessContrast(),
                ToTensorV2()
            ]),
            'valid': Compose([
                Resize(self.image_size[0], self.image_size[1], cv2.INTER_AREA),

                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ]),
            'test': Compose([
                Resize(self.image_size[0], self.image_size[1], cv2.INTER_AREA),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
                ToTensorV2()
            ])
        }
        # self.transformSequence = {
        #     'train': Compose([
        #         # HorizontalFlip(p=0.5),
        #         OneOf([
        #             RandomBrightnessContrast(),
        #             RandomGamma(),
        #         ], p=0.3),
        #         OneOf([
        #             ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        #             GridDistortion(),
        #             OpticalDistortion(distort_limit=2, shift_limit=0.5),
        #         ], p=0.3),
        #         RandomSizedCrop(min_max_height=(int(0.7*self.image_size[0]), self.image_size[1]),
        #                         height=self.image_size[0], width=self.image_size[1],p=0.25),
        #         Resize(self.image_size[0], self.image_size[1], cv2.INTER_AREA),
        #         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
        #         ToTensorV2()
        #     ],p=1),
        #     'valid': Compose([
        #         Resize(self.image_size[0], self.image_size[1], cv2.INTER_AREA),
        #         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
        #         ToTensorV2()
        #     ]),
        #     'test': Compose([
        #         Resize(self.image_size[0], self.image_size[1], cv2.INTER_AREA),
        #         Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255),
        #         ToTensorV2()
        #     ])
        # }
        for pathImageDirectory, pathDatasetFile in image_path_file:
            with open(pathDatasetFile, "r") as fileDescriptor:
                line = fileDescriptor.readline().strip()
                while line:
                    self.img_list.append(join(pathImageDirectory, line.split(',')[0]))
                    self.img_label.append(line.split(',')[1])
                    line = fileDescriptor.readline().strip()

    def __len__(self):
        return len(self.img_list)

    # used to generate mask from rle. USED in here
    def rle2mask(self,rle, width, height):
        mask = np.zeros(width * height)
        if rle == "-1":
            return mask.reshape(width, height)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]

        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position:current_position + lengths[index]] = 255
            current_position += lengths[index]

        return mask.reshape(width, height).T

    def __getitem__(self, idx):
        imagePath = self.img_list[idx]
        maskrle = self.img_label[idx]

        imageData = dicom.dcmread(imagePath).pixel_array
        imageData = ((imageData - imageData.min()) * ( 1 / (imageData.max() - imageData.min()) * 255)).astype('uint8')
        w, h = imageData.shape
        # imageData = np.repeat(np.expand_dims(imageData,-1),3,axis=-1)/255
        imageData = np.repeat(np.expand_dims(imageData,-1),3,axis=-1)
        # maskData = self.rle2mask(maskrle, w, h)/255
        maskData = self.rle2mask(maskrle, w, h)

        dic = self.transformSequence[self.mode](image=imageData, mask=maskData)
        img = dic['image']
        mask = (dic['mask'])
        mask = mask.unsqueeze(0)
        img = img/255
        mask = mask/255

        # print("image data:", img.shape, torch.min(img), torch.max(img))
        # print("maskdata:",mask.shape, torch.min(mask), torch.max(mask))
        # img = rearrange(img, 'c h w-> h w c')
        # img = img.numpy()
        # img = ((img - img.min()) * ( 1 / (img.max() - img.min()) * 255)).astype('uint8')
        # img = Image.fromarray(img)
        # img.save("{}.png".format(idx))
        #
        # maskData = Image.fromarray(mask.numpy()*255).convert("L")
        # maskData.save("{}_phx.png".format(idx))
        #
        # exit(0)
        return img, mask
    
    
### Classification on Vindr-Mammo Dataset:
class VindrmammoClass(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=2, annotation_percent=100):
    self.img_list = []
    self.img_label = []
    self.augment = augment
    with open(file_path, "r") as fileDescriptor:
      line = True
      while line:
        line = fileDescriptor.readline()
        if line and 'image_path' not in line:
          lineItems = line.split()
         
          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:]
          imageLabel = [int(i) for i in imageLabel]
          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    # indexes = np.arange(len(self.img_list))
    # if annotation_percent < 100:
    #   random.Random(99).shuffle(indexes)
    #   num_data = int(indexes.shape[0] * annotation_percent / 100.0)
    #   indexes = indexes[:num_data]

    #   _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
    #   self.img_list = []
    #   self.img_label = []

    #   for i in indexes:
    #     self.img_list.append(_img_list[i])
    #     self.img_label.append(_img_label[i])

      

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])
    #imageLabel = self.img_label[index]
    # print("CHECK", imageData.shape, imageLabel.shape)

    if self.augment != None: 
      imageData = self.augment(imageData)
    return imageData,imageLabel

  def __len__(self):
    return len(self.img_list)
    
## Classification on TBX11K Dataset
# file_path 'lists/TBX11K_train.txt'
# images_path "/scratch/jliang12/data/tbx11k/tbx11k/TBX11K/"
class TBX11KDataset(Dataset): ## Need to FIX

  def __init__(self, images_path, file_path, augment, num_class=1, annotation_percent=100):
    self.img_list = []
    self.img_label = []
    self.augment = augment

    with open(images_path+file_path, "r") as fileDescriptor:
      line = True
      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path+"imgs/", lineItems[0])
          if lineItems[0].startswith("tb"):
            imageLabel = 1
          else:
            imageLabel = 0
        #   imageLabel = lineItems[1:num_class + 1]
        #   imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)
        #   print("[CHECK]  Image:", imagePath, imageLabel)
        #   print("[CHECK]  ImageLabel:", imageLabel)
        #   exit(0)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]
    # print("CHECK Image", imagePath)

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = self.img_label[index]
    # print("CHECK ImageLabel", imageLabel)

    if self.augment != None: imageData = self.augment(imageData)
    
    return imageData, imageLabel

  def __len__(self):

    return len(self.img_list)
  


## Classification on NIH14 ChestXray Dataset
class ChestXray14Dataset(Dataset):

  def __init__(self, images_path, file_path, augment, num_class=14, annotation_percent=100):

    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.diseases_LIST = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']

    with open(file_path, "r") as fileDescriptor:
      line = True

      while line:
        line = fileDescriptor.readline()

        if line:
          lineItems = line.split()

          imagePath = os.path.join(images_path, lineItems[0])
          imageLabel = lineItems[1:num_class + 1]
          imageLabel = [int(i) for i in imageLabel]

          self.img_list.append(imagePath)
          self.img_label.append(imageLabel)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):

    imagePath = self.img_list[index]

    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: imageData = self.augment(imageData)

    return imageData, imageLabel

  def __len__(self):
    return len(self.img_list)
  

## Classification on CheXpert Dataset
class CheXpert(Dataset):
  def __init__(self, images_path, file_path, augment, num_class=14,
               uncertain_label="LSR-Ones", unknown_label=0, annotation_percent=100):
    self.img_list = []
    self.img_label = []
    self.augment = augment
    self.train_augment = build_ts_transformations()
    self.diseases_LIST = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
    self.diseases_LIST_test = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
    
    assert uncertain_label in ["Ones", "Zeros", "LSR-Ones", "LSR-Zeros"]
    self.uncertain_label = uncertain_label

    with open(file_path, "r") as fileDescriptor:
      csvReader = csv.reader(fileDescriptor)
      next(csvReader, None)
      for line in csvReader:
        imagePath = os.path.join(images_path, line[0])
        label = line[5:]
        for i in range(num_class):
          if label[i]:
            a = float(label[i])
            if a == 1:
              label[i] = 1
            elif a == 0:
              label[i] = 0
            elif a == -1: # uncertain label
              if self.uncertain_label == "Ones":
                label[i] = 1
              elif self.uncertain_label == "Zeros":
                label[i] = 0
              elif self.uncertain_label == "LSR-Ones":
                label[i] = random.uniform(0.55, 0.85)
              elif self.uncertain_label == "LSR-Zeros":
                label[i] = random.uniform(0, 0.3)
          else:
            label[i] = unknown_label # unknown label

        self.img_list.append(imagePath)
        self.img_label.append(label)

    indexes = np.arange(len(self.img_list))
    if annotation_percent < 100:
      random.Random(99).shuffle(indexes)
      num_data = int(indexes.shape[0] * annotation_percent / 100.0)
      indexes = indexes[:num_data]

      _img_list, _img_label = copy.deepcopy(self.img_list), copy.deepcopy(self.img_label)
      self.img_list = []
      self.img_label = []

      for i in indexes:
        self.img_list.append(_img_list[i])
        self.img_label.append(_img_label[i])

  def __getitem__(self, index):
    imagePath = self.img_list[index]
    imageData = Image.open(imagePath).convert('RGB')
    imageLabel = torch.FloatTensor(self.img_label[index])

    if self.augment != None: 
      student_img, teacher_img = self.augment(imageData), self.augment(imageData)   
    else:
      imageData = (np.array(imageData)).astype('uint8')
      augmented = self.train_augment(image = imageData, mask = imageData)
      student_img = augmented['image']
      teacher_img = augmented['mask']
      student_img=np.array(student_img) / 255.
      teacher_img=np.array(teacher_img) / 255.
      
      mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
      student_img = (student_img-mean)/std
      teacher_img = (teacher_img-mean)/std
      student_img = student_img.transpose(2, 0, 1).astype('float32')
      teacher_img = teacher_img.transpose(2, 0, 1).astype('float32')
    
    # return student_img, teacher_img, imageLabel
    return student_img, imageLabel

  def __len__(self):
    return len(self.img_list)






class LinearProbeDataset(Dataset):
    def __init__(self, input_embedding_path, input_lbl_path, dataset):
        self.input_embedding = np.load(input_embedding_path)
        self.input_lbl = np.load(input_lbl_path)
        self.random_shuffle()
        self.dataset = dataset

    def __getitem__(self, index):
        imageData = self.input_embedding[index]


        if self.dataset == "chexpert":
            label = []
            for l in self.input_lbl[index]:
                if l == -1:
                    label.append(random.uniform(0.55, 0.85))
                else:
                    label.append(l)
            imageLabel = torch.FloatTensor(label)
        else:
            imageLabel = torch.FloatTensor(self.input_lbl[index])

        return imageData, imageLabel

    def random_shuffle(self):
        idx = np.random.permutation(range(len(self.input_embedding)))
        self.input_embedding = self.input_embedding[idx,:]
        self.input_lbl = self.input_lbl[idx,:]

    def __len__(self):
        return len(self.input_embedding)







### --- Return Dataloader ---- ##
def dataloader_return(args):
    from datasets import build_dataset
    import util.misc as utils
    from torch.utils.data import DataLoader, DistributedSampler

    if args.taskcomponent == 'segmentation' or args.taskcomponent == 'segmentation_cyclic' or args.taskcomponent == "segmentation_vindrcxr_organ_1h3ch":
        # print("CHECK!!!")
        if args.segmentation_dataset == 'jsrt_lung':
            if args.serverC == 'DFS':
                train_image_path_file = [("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/train.txt")]
                val_image_path_file = [("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/val.txt")]
                test_image_path_file = [("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/test.txt")]
            elif args.serverC == "SOL":
                train_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/train.txt")] 
                val_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/val.txt")]
                test_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/test.txt")]

            train_dataset = JSRTLungDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

            val_dataset = JSRTLungDataset(val_image_path_file,image_size=(args.imgsize,args.imgsize), mode="val")
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                     pin_memory=True, shuffle=True,drop_last=False )

            test_dataset = JSRTLungDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="val")
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      num_workers=args.num_workers, pin_memory=True,drop_last=False )
            return train_loader, val_loader, test_loader

        elif args.segmentation_dataset == 'jsrt_clavicle':
            if args.serverC == 'DFS':
                train_image_path_file = [("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/train.txt")]
                val_image_path_file = [("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/val.txt")]
                test_image_path_file = [("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/test.txt")]
            elif args.serverC == "SOL":
                train_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/train.txt")] 
                val_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/val.txt")]
                test_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/test.txt")]

            train_dataset = JSRTClavicleDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), mode="train", ann=args.segmentation_dataset_ann)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

            val_dataset = JSRTClavicleDataset(val_image_path_file,image_size=(args.imgsize,args.imgsize), mode="val")
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                     pin_memory=True, shuffle=True,drop_last=False )

            test_dataset = JSRTClavicleDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="val")

            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      num_workers=args.num_workers, pin_memory=True,drop_last=False )
            return train_loader, val_loader, test_loader

        elif args.segmentation_dataset == 'jsrt_heart': # jsrt_leftlung
            if args.serverC == 'DFS':
                train_image_path_file = [("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/train.txt")]
                val_image_path_file = [("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/val.txt")]
                test_image_path_file = [("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/test.txt")]
            elif args.serverC == "SOL":
                train_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/train.txt")] 
                val_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/val.txt")]
                test_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/test.txt")]

            train_dataset = JSRTHeartDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

            val_dataset = JSRTHeartDataset(val_image_path_file,image_size=(args.imgsize,args.imgsize), mode="val")
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                     pin_memory=True, shuffle=True,drop_last=False )

            test_dataset = JSRTHeartDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="val")

            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      num_workers=args.num_workers, pin_memory=True,drop_last=False )
            return train_loader, val_loader, test_loader

        elif args.segmentation_dataset == 'jsrt_leftlung':
            if args.serverC == 'DFS':
                train_image_path_file = [("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/train.txt")]
                val_image_path_file = [("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/val.txt")]
                test_image_path_file = [("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/test.txt")]
            elif args.serverC == "SOL":
                train_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/train.txt")] 
                val_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/val.txt")]
                test_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/test.txt")]

            train_dataset = JSRTLeftLungDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

            val_dataset = JSRTLeftLungDataset(val_image_path_file,image_size=(args.imgsize,args.imgsize), mode="val")
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                     pin_memory=True, shuffle=True,drop_last=False )

            test_dataset = JSRTLeftLungDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="val")

            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      num_workers=args.num_workers, pin_memory=True,drop_last=False )
            return train_loader, val_loader, test_loader

        elif args.segmentation_dataset == 'chestxdetdataset': # Disease Segmentation
            if args.serverC == 'DFS':
                train_image_path_file = [("/mnt/dfs/nuislam/Data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_train_data.txt")]
                # val_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "/scratch/jliang12/data/ChestX-Det/data_files/ChestX-Det_valid_data.txt")]
                test_image_path_file = [("/mnt/dfs/nuislam/Data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_test_data.txt")]
            elif args.serverC == "SOL":
                train_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_train_data.txt")]
                # val_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "/scratch/jliang12/data/ChestX-Det/data_files/ChestX-Det_valid_data.txt")]
                test_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_test_data.txt")]

            train_dataset = ChestXDetDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

            # val_dataset = ChestXDetDataset(val_image_path_file,image_size=(args.imgsize,args.imgsize), mode="val")
            # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            #                                          pin_memory=True, shuffle=True,drop_last=False )

            test_dataset = ChestXDetDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")

            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      num_workers=args.num_workers, pin_memory=True,drop_last=False )
            return train_loader, test_loader

        elif args.segmentation_dataset == 'jsrt_lung_heart_clavicle': # For cyclic
            if args.serverC == 'DFS':
                train_image_path_file = [("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/train.txt")]
                val_image_path_file = [("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/val.txt")]
                test_image_path_file = [("/mnt/dfs/jpang12/datasets/JSRT/All247images", "data/jsrt/test.txt")]
            elif args.serverC == "SOL":
                train_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/train.txt")] 
                val_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/val.txt")]
                test_image_path_file = [("/data/jliang12/jpang12/dataset/JSRT/All247images", "data/jsrt/test.txt")]

            train_dataset = JSRTLungDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize))
            train_loader_jsrtLung = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

            val_dataset = JSRTLungDataset(val_image_path_file,image_size=(args.imgsize,args.imgsize), mode="val")
            val_loader_jsrtLung = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                     pin_memory=True, shuffle=True,drop_last=False )

            test_dataset = JSRTLungDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="val")
            test_loader_jsrtLung = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      num_workers=args.num_workers, pin_memory=True,drop_last=False )


            train_dataset = JSRTClavicleDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize))
            train_loader_jsrtClavicle = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

            val_dataset = JSRTClavicleDataset(val_image_path_file,image_size=(args.imgsize,args.imgsize), mode="val")
            val_loader_jsrtClavicle = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                     pin_memory=True, shuffle=True,drop_last=False )

            test_dataset = JSRTClavicleDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="val")

            test_loader_jsrtClavicle = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      num_workers=args.num_workers, pin_memory=True,drop_last=False )


            train_dataset = JSRTHeartDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize))
            train_loader_jsrtHeart = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

            val_dataset = JSRTHeartDataset(val_image_path_file,image_size=(args.imgsize,args.imgsize), mode="val")
            val_loader_jsrtHeart = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                     pin_memory=True, shuffle=True,drop_last=False )

            test_dataset = JSRTHeartDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="val")

            test_loader_jsrtHeart = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                      num_workers=args.num_workers, pin_memory=True,drop_last=False )
            return train_loader_jsrtLung, val_loader_jsrtLung, test_loader_jsrtLung, train_loader_jsrtClavicle, val_loader_jsrtClavicle, test_loader_jsrtClavicle, train_loader_jsrtHeart, val_loader_jsrtHeart, test_loader_jsrtHeart

        elif args.segmentation_dataset == 'vindrcxr_lung':
            if args.serverC == 'DFS':
                train_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
                test_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
            elif args.serverC == "SOL":
                train_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
                test_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
            
            train_dataset = VindrCXRLungDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

            val_dataset = VindrCXRLungDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                     pin_memory=True, shuffle=True,drop_last=False )
            return train_loader, val_loader

        elif args.segmentation_dataset == 'vindrcxr_heart':
            if args.serverC == 'DFS':
                train_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
                test_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
            elif args.serverC == "SOL":
                train_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
                test_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
            
            train_dataset = VindrCXRHeartDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), ann=2500)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

            val_dataset = VindrCXRHeartDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                     pin_memory=True, shuffle=True,drop_last=False )
            return train_loader, val_loader

        elif args.segmentation_dataset == 'vindrcxr_leftlung':
            if args.serverC == 'DFS':
                train_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
                test_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
            elif args.serverC == "SOL":
                train_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
                test_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
            
            train_dataset = VindrCXRLeftLungDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), ann=2500)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

            val_dataset = VindrCXRLeftLungDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                     pin_memory=True, shuffle=True,drop_last=False )

            return train_loader, val_loader

        elif args.segmentation_dataset == 'vindrcxr_rightlung':
            if args.serverC == 'DFS':
                train_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
                test_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
            elif args.serverC == "SOL":
                train_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
                test_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
            
            train_dataset = VindrCXRRightLungDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), ann=2500)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

            val_dataset = VindrCXRRightLungDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                     pin_memory=True, shuffle=True,drop_last=False )
            return train_loader, val_loader


        elif args.segmentation_dataset == 'vindrcxr_lung_heart':
            if args.serverC == 'DFS':
                train_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
                test_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
            elif args.serverC == "SOL":
                train_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
                test_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]

            train_dataset = VindrCXRHeartDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize))
            train_loader_vindrcxrHeart = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

            val_dataset = VindrCXRHeartDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
            val_loader_vindrcxrtHeart = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                     pin_memory=True, shuffle=True,drop_last=False )


            train_dataset = VindrCXRLungDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize))
            train_loader_vindrcxrLung = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                       num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

            val_dataset = VindrCXRLungDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
            val_loader_vindrcxrtLung = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                     pin_memory=True, shuffle=True,drop_last=False )
            return train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLung, val_loader_vindrcxrtLung

        elif args.segmentation_dataset == 'segmentation_vindrcxr_organ_1h3ch': # segmentation_vindrcxr_organ_1h3ch
            print("[CHECK]", args.taskcomponent, args.segmentation_dataset)
            if args.serverC == 'DFS':
                train_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
                test_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
            elif args.serverC == "SOL":
                train_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
                test_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]

            train_dataset = VindrCXRHLLRL_3chDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), mode="train", ann=7500)
            train_loader_vindrcxr3ch = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

            val_dataset = VindrCXRHLLRL_3chDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
            val_loader_vindrcxrt3ch = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=False )

            return train_loader_vindrcxr3ch, val_loader_vindrcxrt3ch


    
    if args.taskcomponent == "detect_segmentation_cyclic":
        ### SEGMENTATION DATA LOADING
        if args.serverC == 'DFS':
            train_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
            test_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
        elif args.serverC == "SOL":
            train_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
            test_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
        
        train_dataset = VindrCXRHeartDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize))
        train_loader_vindrcxrHeart = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

        val_dataset = VindrCXRHeartDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
        val_loader_vindrcxrtHeart = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                 pin_memory=True, shuffle=True,drop_last=False )


        train_dataset = VindrCXRLungDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize))
        train_loader_vindrcxrLung = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

        val_dataset = VindrCXRLungDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
        val_loader_vindrcxrtLung = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                 pin_memory=True, shuffle=True,drop_last=False )


        ### Localization DATA LOADING
        # ## Localization of Heart
        # dataset_train = build_dataset(image_set='vindrcxrOrgan_trainHeart', args=args)
        # dataset_val = build_dataset(image_set='vindrcxrOrgan_testHeart', args=args)

        # if args.distributed:
        #     sampler_train = DistributedSampler(dataset_train)
        #     sampler_val = DistributedSampler(dataset_val, shuffle=False)
        # else:
        #     sampler_train = torch.utils.data.RandomSampler(dataset_train)
        #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        # batch_sampler_train = torch.utils.data.BatchSampler(
        #     sampler_train, args.batch_size, drop_last=True)

        # data_loader_train_Heart = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
        #                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
        # data_loader_val_Heart = DataLoader(dataset_val, 1, sampler=sampler_val,
        #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


        # ## Localization of Lung
        # dataset_train = build_dataset(image_set='vindrcxrOrgan_trainLung', args=args)
        # dataset_val = build_dataset(image_set='vindrcxrOrgan_testLung', args=args)

        # if args.distributed:
        #     sampler_train = DistributedSampler(dataset_train)
        #     sampler_val = DistributedSampler(dataset_val, shuffle=False)
        # else:
        #     sampler_train = torch.utils.data.RandomSampler(dataset_train)
        #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        # batch_sampler_train = torch.utils.data.BatchSampler( sampler_train, args.batch_size, drop_last=True )

        # data_loader_train_Lung = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
        #                                collate_fn=utils.collate_fn, num_workers=args.num_workers)
        # data_loader_val_Lung = DataLoader(dataset_val, 1, sampler=sampler_val,
        #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


        dataset_train = build_dataset(image_set='vindrcxrOrgan_train', args=args)
        dataset_val = build_dataset(image_set='vindrcxrOrgan_test', args=args)
        args.num_classes = 2+1
        args.dn_labelbook_size = 4

        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

        # return train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLung, val_loader_vindrcxrtLung, data_loader_train_Heart, data_loader_val_Heart, data_loader_train_Lung, data_loader_val_Lung, sampler_train, dataset_val
        return train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLung, val_loader_vindrcxrtLung, data_loader_train, data_loader_val, sampler_train, dataset_val


    if args.taskcomponent == "detect_segmentation_cyclic_v2" or args.taskcomponent == "detect_segmentation_cyclic_v3":  ## with EMA ## Training Strategy 1 OR Training Strategy 2

        ### SEGMENTATION DATA LOADING
        if args.serverC == 'DFS':
            train_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
            test_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
        elif args.serverC == "SOL":
            train_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
            test_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
        
        train_dataset = VindrCXRHeartDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), mode="train", ann=2500) # trainA/B 1250 | train 2500
        train_loader_vindrcxrHeart_A = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )
        # train_dataset = VindrCXRHeartDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), mode="train", ann=2500) # trainA/B 1250 | train 2500
        # train_loader_vindrcxrHeart_B = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
        #                                            num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

        val_dataset = VindrCXRHeartDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
        val_loader_vindrcxrtHeart = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                 pin_memory=True, shuffle=False,drop_last=False )


        train_dataset = VindrCXRLeftLungDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), mode="train", ann=2500) # trainA/B 1250 | train 2500
        train_loader_vindrcxrLeftLung_A = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )
        # train_dataset = VindrCXRLeftLungDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), mode="train", ann=2500) # trainA/B 1250 | train 2500
        # train_loader_vindrcxrLeftLung_B = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
        #                                            num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

        val_dataset = VindrCXRLeftLungDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
        val_loader_vindrcxrtLeftLung = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                 pin_memory=True, shuffle=False,drop_last=False )

        train_dataset = VindrCXRRightLungDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), mode="train", ann=2500) # trainA/B 1250 | train 2500
        train_loader_vindrcxrRightLung_A = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )
        # train_dataset = VindrCXRRightLungDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), mode="train", ann=2500) # trainA/B 1250 | train 2500
        # train_loader_vindrcxrRightLung_B = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
        #                                            num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

        val_dataset = VindrCXRRightLungDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
        val_loader_vindrcxrtRightLung = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                 pin_memory=True, shuffle=False,drop_last=False )

        ### 3ch for Testing
        val_dataset = VindrCXRHLLRL_3chDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
        val_loader_vindrcxrt3ch = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False,drop_last=False )


        ## Heart
        dataset_trainHeartA = build_dataset(image_set='vindrcxrOrgan_trainHeart', args=args) # vindrcxrOrgan_trainHeart | vindrcxrOrgan_trainHeartA1250
        # dataset_trainHeartB = build_dataset(image_set='vindrcxrOrgan_trainHeartB1250', args=args) # vindrcxrOrgan_trainHeart | vindrcxrOrgan_trainHeartB1250
        dataset_valHeart = build_dataset(image_set='vindrcxrOrgan_testHeart', args=args)
        if args.distributed:
            sampler_trainHeartA = DistributedSampler(dataset_trainHeartA)
            # sampler_trainHeartB = DistributedSampler(dataset_trainHeartB)
            sampler_valHeart = DistributedSampler(dataset_valHeart, shuffle=False)
        else:
            sampler_trainHeartA = torch.utils.data.RandomSampler(dataset_trainHeartA)
            # sampler_trainHeartB = torch.utils.data.RandomSampler(dataset_trainHeartB)
            sampler_valHeart = torch.utils.data.SequentialSampler(dataset_valHeart)
        batch_sampler_trainHeartA = torch.utils.data.BatchSampler(sampler_trainHeartA, args.batch_size, drop_last=True)
        # batch_sampler_trainHeartB = torch.utils.data.BatchSampler(sampler_trainHeartB, args.batch_size, drop_last=True)
        data_loader_trainHeartA = DataLoader(dataset_trainHeartA, batch_sampler=batch_sampler_trainHeartA, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        # data_loader_trainHeartB = DataLoader(dataset_trainHeartB, batch_sampler=batch_sampler_trainHeartB, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_valHeart = DataLoader(dataset_valHeart, 1, sampler=sampler_valHeart, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

        ## Left Lung
        dataset_trainLeftLungA = build_dataset(image_set='vindrcxrOrgan_trainLeftLung', args=args) # vindrcxrOrgan_trainLeftLung | vindrcxrOrgan_trainLeftLungA1250
        # dataset_trainLeftLungB = build_dataset(image_set='vindrcxrOrgan_trainLeftLungB1250', args=args) # vindrcxrOrgan_trainLeftLung | vindrcxrOrgan_trainLeftLungB1250
        dataset_valLeftLung = build_dataset(image_set='vindrcxrOrgan_testLeftLung', args=args)
        if args.distributed:
            sampler_trainLeftLungA = DistributedSampler(dataset_trainLeftLungA)
            # sampler_trainLeftLungB = DistributedSampler(dataset_trainLeftLungB)
            sampler_valLeftLung = DistributedSampler(dataset_valLeftLung, shuffle=False)
        else:
            sampler_trainLeftLungA = torch.utils.data.RandomSampler(dataset_trainLeftLungA)
            # sampler_trainLeftLungB = torch.utils.data.RandomSampler(dataset_trainLeftLungB)
            sampler_valLeftLung = torch.utils.data.SequentialSampler(dataset_valLeftLung)
        batch_sampler_trainLeftLungA = torch.utils.data.BatchSampler(sampler_trainLeftLungA, args.batch_size, drop_last=True)
        # batch_sampler_trainLeftLungB = torch.utils.data.BatchSampler(sampler_trainLeftLungB, args.batch_size, drop_last=True)
        data_loader_trainLeftLungA = DataLoader(dataset_trainLeftLungA, batch_sampler=batch_sampler_trainLeftLungA, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        # data_loader_trainLeftLungB = DataLoader(dataset_trainLeftLungB, batch_sampler=batch_sampler_trainLeftLungB, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_valLeftLung = DataLoader(dataset_valLeftLung, 1, sampler=sampler_valLeftLung, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

        ## Right Lung
        dataset_trainRightLungA = build_dataset(image_set='vindrcxrOrgan_trainRightLung', args=args) # vindrcxrOrgan_trainRightLung | vindrcxrOrgan_trainRightLungA1250
        # dataset_trainRightLungB = build_dataset(image_set='vindrcxrOrgan_trainRightLungB1250', args=args) # vindrcxrOrgan_trainRightLung | vindrcxrOrgan_trainRightLungB1250
        dataset_valRightLung = build_dataset(image_set='vindrcxrOrgan_testRightLung', args=args)
        if args.distributed:
            sampler_trainRightLungA = DistributedSampler(dataset_trainRightLungA)
            # sampler_trainRightLungB = DistributedSampler(dataset_trainRightLungB)
            sampler_valRightLung = DistributedSampler(dataset_valRightLung, shuffle=False)
        else:
            sampler_trainRightLungA = torch.utils.data.RandomSampler(dataset_trainRightLungA)
            # sampler_trainRightLungB = torch.utils.data.RandomSampler(dataset_trainRightLungB)
            sampler_valRightLung = torch.utils.data.SequentialSampler(dataset_valRightLung)
        batch_sampler_trainRightLungA = torch.utils.data.BatchSampler(sampler_trainRightLungA, args.batch_size, drop_last=True)
        # batch_sampler_trainRightLungB = torch.utils.data.BatchSampler(sampler_trainRightLungB, args.batch_size, drop_last=True)
        data_loader_trainRightLungA = DataLoader(dataset_trainRightLungA, batch_sampler=batch_sampler_trainRightLungA, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        # data_loader_trainRightLungB = DataLoader(dataset_trainRightLungB, batch_sampler=batch_sampler_trainRightLungB, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_valRightLung = DataLoader(dataset_valRightLung, 1, sampler=sampler_valRightLung, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

        # return train_loader_vindrcxrHeart_A, train_loader_vindrcxrHeart_B, val_loader_vindrcxrtHeart, train_loader_vindrcxrLeftLung_A, train_loader_vindrcxrLeftLung_B, val_loader_vindrcxrtLeftLung, train_loader_vindrcxrRightLung_A, train_loader_vindrcxrRightLung_B, val_loader_vindrcxrtRightLung, \
        #     data_loader_trainHeartA, sampler_trainHeartA, data_loader_trainHeartB, sampler_trainHeartB, data_loader_valHeart, dataset_valHeart, \
        #     data_loader_trainLeftLungA, sampler_trainLeftLungA, data_loader_trainLeftLungB, sampler_trainLeftLungB, data_loader_valLeftLung, dataset_valLeftLung, \
        #     data_loader_trainRightLungA, sampler_trainRightLungA, data_loader_trainRightLungB, sampler_trainRightLungB, data_loader_valRightLung, dataset_valRightLung
        return train_loader_vindrcxrHeart_A, val_loader_vindrcxrtHeart, train_loader_vindrcxrLeftLung_A, val_loader_vindrcxrtLeftLung, train_loader_vindrcxrRightLung_A, val_loader_vindrcxrtRightLung, val_loader_vindrcxrt3ch, \
            data_loader_trainHeartA, sampler_trainHeartA, data_loader_valHeart, dataset_valHeart, \
            data_loader_trainLeftLungA, sampler_trainLeftLungA, data_loader_valLeftLung, dataset_valLeftLung, \
            data_loader_trainRightLungA, sampler_trainRightLungA, data_loader_valRightLung, dataset_valRightLung


    # if args.taskcomponent == "detect_segmentation_cyclic_v3":  ## with EMA ## Training Strategy 2

    #     ### SEGMENTATION DATA LOADING
    #     if args.serverC == 'DFS':
    #         train_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
    #         test_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
    #     elif args.serverC == "SOL":
    #         train_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
    #         test_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
        
    #     train_dataset = VindrCXRHeartDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), ann=2500)
    #     train_loader_vindrcxrHeart = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
    #                                                num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

    #     val_dataset = VindrCXRHeartDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
    #     val_loader_vindrcxrtHeart = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    #                                              pin_memory=True, shuffle=True,drop_last=False )


    #     train_dataset = VindrCXRLeftLungDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), ann=2500)
    #     train_loader_vindrcxrLeftLung = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
    #                                                num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

    #     val_dataset = VindrCXRLeftLungDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
    #     val_loader_vindrcxrtLeftLung = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    #                                              pin_memory=True, shuffle=True,drop_last=False )

    #     train_dataset = VindrCXRRightLungDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), ann=2500)
    #     train_loader_vindrcxrRightLung = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
    #                                                num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

    #     val_dataset = VindrCXRRightLungDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
    #     val_loader_vindrcxrtRightLung = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
    #                                              pin_memory=True, shuffle=True,drop_last=False )


    #     ## Heart
    #     dataset_trainHeartA = build_dataset(image_set='vindrcxrOrgan_trainHeartPart1', args=args)
    #     dataset_trainHeartB = build_dataset(image_set='vindrcxrOrgan_trainHeartPart2', args=args)
    #     dataset_trainHeartC = build_dataset(image_set='vindrcxrOrgan_trainHeartPart3', args=args)
    #     dataset_valHeart = build_dataset(image_set='vindrcxrOrgan_testHeart', args=args)
    #     if args.distributed:
    #         sampler_trainHeartA = DistributedSampler(dataset_trainHeartA)
    #         sampler_trainHeartB = DistributedSampler(dataset_trainHeartB)
    #         sampler_trainHeartC = DistributedSampler(dataset_trainHeartC)
    #         sampler_valHeart = DistributedSampler(dataset_valHeart, shuffle=False)
    #     else:
    #         sampler_trainHeartA = torch.utils.data.RandomSampler(dataset_trainHeartA)
    #         sampler_trainHeartB = torch.utils.data.RandomSampler(dataset_trainHeartB)
    #         sampler_trainHeartC = torch.utils.data.RandomSampler(dataset_trainHeartC)
    #         sampler_valHeart = torch.utils.data.SequentialSampler(dataset_valHeart)
    #     batch_sampler_trainHeartA = torch.utils.data.BatchSampler(sampler_trainHeartA, args.batch_size, drop_last=True)
    #     batch_sampler_trainHeartB = torch.utils.data.BatchSampler(sampler_trainHeartB, args.batch_size, drop_last=True)
    #     batch_sampler_trainHeartC = torch.utils.data.BatchSampler(sampler_trainHeartC, args.batch_size, drop_last=True)
    #     data_loader_trainHeartA = DataLoader(dataset_trainHeartA, batch_sampler=batch_sampler_trainHeartA, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    #     data_loader_trainHeartB = DataLoader(dataset_trainHeartB, batch_sampler=batch_sampler_trainHeartB, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    #     data_loader_trainHeartC = DataLoader(dataset_trainHeartC, batch_sampler=batch_sampler_trainHeartC, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    #     data_loader_valHeart = DataLoader(dataset_valHeart, 1, sampler=sampler_valHeart, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    #     ## Left Lung
    #     dataset_trainLeftLungA = build_dataset(image_set='vindrcxrOrgan_trainLeftLungPart1', args=args)
    #     dataset_trainLeftLungB = build_dataset(image_set='vindrcxrOrgan_trainLeftLungPart2', args=args)
    #     dataset_trainLeftLungC = build_dataset(image_set='vindrcxrOrgan_trainLeftLungPart3', args=args)
    #     dataset_valLeftLung = build_dataset(image_set='vindrcxrOrgan_testLeftLung', args=args)
    #     if args.distributed:
    #         sampler_trainLeftLungA = DistributedSampler(dataset_trainLeftLungA)
    #         sampler_trainLeftLungB = DistributedSampler(dataset_trainLeftLungB)
    #         sampler_trainLeftLungC = DistributedSampler(dataset_trainLeftLungC)
    #         sampler_valLeftLung = DistributedSampler(dataset_valLeftLung, shuffle=False)
    #     else:
    #         sampler_trainLeftLungA = torch.utils.data.RandomSampler(dataset_trainLeftLungA)
    #         sampler_trainLeftLungB = torch.utils.data.RandomSampler(dataset_trainLeftLungB)
    #         sampler_trainLeftLungC = torch.utils.data.RandomSampler(dataset_trainLeftLungC)
    #         sampler_valLeftLung = torch.utils.data.SequentialSampler(dataset_valLeftLung)
    #     batch_sampler_trainLeftLungA = torch.utils.data.BatchSampler(sampler_trainLeftLungA, args.batch_size, drop_last=True)
    #     batch_sampler_trainLeftLungB = torch.utils.data.BatchSampler(sampler_trainLeftLungB, args.batch_size, drop_last=True)
    #     batch_sampler_trainLeftLungC = torch.utils.data.BatchSampler(sampler_trainLeftLungC, args.batch_size, drop_last=True)
    #     data_loader_trainLeftLungA = DataLoader(dataset_trainLeftLungA, batch_sampler=batch_sampler_trainLeftLungA, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    #     data_loader_trainLeftLungB = DataLoader(dataset_trainLeftLungB, batch_sampler=batch_sampler_trainLeftLungB, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    #     data_loader_trainLeftLungC = DataLoader(dataset_trainLeftLungC, batch_sampler=batch_sampler_trainLeftLungC, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    #     data_loader_valLeftLung = DataLoader(dataset_valLeftLung, 1, sampler=sampler_valLeftLung, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    #     ## Right Lung
    #     dataset_trainRightLungA = build_dataset(image_set='vindrcxrOrgan_trainRightLungPart1', args=args)
    #     dataset_trainRightLungB = build_dataset(image_set='vindrcxrOrgan_trainRightLungPart2', args=args)
    #     dataset_trainRightLungC = build_dataset(image_set='vindrcxrOrgan_trainRightLungPart3', args=args)
    #     dataset_valRightLung = build_dataset(image_set='vindrcxrOrgan_testRightLung', args=args)
    #     if args.distributed:
    #         sampler_trainRightLungA = DistributedSampler(dataset_trainRightLungA)
    #         sampler_trainRightLungB = DistributedSampler(dataset_trainRightLungB)
    #         sampler_trainRightLungC = DistributedSampler(dataset_trainRightLungC)
    #         sampler_valRightLung = DistributedSampler(dataset_valRightLung, shuffle=False)
    #     else:
    #         sampler_trainRightLungA = torch.utils.data.RandomSampler(dataset_trainRightLungA)
    #         sampler_trainRightLungB = torch.utils.data.RandomSampler(dataset_trainRightLungB)
    #         sampler_trainRightLungC = torch.utils.data.RandomSampler(dataset_trainRightLungC)
    #         sampler_valRightLung = torch.utils.data.SequentialSampler(dataset_valRightLung)
    #     batch_sampler_trainRightLungA = torch.utils.data.BatchSampler(sampler_trainRightLungA, args.batch_size, drop_last=True)
    #     batch_sampler_trainRightLungB = torch.utils.data.BatchSampler(sampler_trainRightLungB, args.batch_size, drop_last=True)
    #     batch_sampler_trainRightLungC = torch.utils.data.BatchSampler(sampler_trainRightLungC, args.batch_size, drop_last=True)
    #     data_loader_trainRightLungA = DataLoader(dataset_trainRightLungA, batch_sampler=batch_sampler_trainRightLungA, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    #     data_loader_trainRightLungB = DataLoader(dataset_trainRightLungB, batch_sampler=batch_sampler_trainRightLungB, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    #     data_loader_trainRightLungC = DataLoader(dataset_trainRightLungC, batch_sampler=batch_sampler_trainRightLungC, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    #     data_loader_valRightLung = DataLoader(dataset_valRightLung, 1, sampler=sampler_valRightLung, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    #     return train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLeftLung, val_loader_vindrcxrtLeftLung, train_loader_vindrcxrRightLung, val_loader_vindrcxrtRightLung, \
    #         data_loader_trainHeartA, sampler_trainHeartA, data_loader_trainHeartB, sampler_trainHeartB, data_loader_trainHeartC, sampler_trainHeartC, data_loader_valHeart, dataset_valHeart, \
    #         data_loader_trainLeftLungA, sampler_trainLeftLungA, data_loader_trainLeftLungB, sampler_trainLeftLungB, data_loader_trainLeftLungC, sampler_trainLeftLungC, data_loader_valLeftLung, dataset_valLeftLung, \
    #         data_loader_trainRightLungA, sampler_trainRightLungA, data_loader_trainRightLungB, sampler_trainRightLungB, data_loader_trainRightLungC, sampler_trainRightLungC, data_loader_valRightLung, dataset_valRightLung



    if args.taskcomponent == "detect_segmentation_cyclic_v4":  ## with EMA

        ### SEGMENTATION DATA LOADING
        if args.serverC == 'DFS':
            train_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
            test_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
        elif args.serverC == "SOL":
            train_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
            test_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
        
        train_dataset = VindrCXRHeartDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), ann=2500)
        train_loader_vindrcxrHeart = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

        val_dataset = VindrCXRHeartDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
        val_loader_vindrcxrtHeart = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                 pin_memory=True, shuffle=True,drop_last=False )


        train_dataset = VindrCXRLeftLungDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), ann=2500)
        train_loader_vindrcxrLeftLung = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

        val_dataset = VindrCXRLeftLungDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
        val_loader_vindrcxrtLeftLung = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                 pin_memory=True, shuffle=True,drop_last=False )

        train_dataset = VindrCXRRightLungDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), ann=2500)
        train_loader_vindrcxrRightLung = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

        val_dataset = VindrCXRRightLungDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
        val_loader_vindrcxrtRightLung = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                                 pin_memory=True, shuffle=True,drop_last=False )


        ### Localization Data Loading

        # dataset_train = build_dataset(image_set='vindrcxrOrgan_train', args=args)
        # dataset_val = build_dataset(image_set='vindrcxrOrgan_test', args=args)
        # if args.distributed:
        #     sampler_train = DistributedSampler(dataset_train)
        #     sampler_val = DistributedSampler(dataset_val, shuffle=False)
        # else:
        #     sampler_train = torch.utils.data.RandomSampler(dataset_train)
        #     sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        # batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

        # data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        # data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


        ## Heart
        dataset_trainHeart = build_dataset(image_set='vindrcxrOrgan_trainHeart', args=args)
        dataset_valHeart = build_dataset(image_set='vindrcxrOrgan_testHeart', args=args)
        if args.distributed:
            sampler_trainHeart = DistributedSampler(dataset_trainHeart)
            sampler_valHeart = DistributedSampler(dataset_valHeart, shuffle=False)
        else:
            sampler_trainHeart = torch.utils.data.RandomSampler(dataset_trainHeart)
            sampler_valHeart = torch.utils.data.SequentialSampler(dataset_valHeart)
        batch_sampler_trainHeart = torch.utils.data.BatchSampler(sampler_trainHeart, args.batch_size, drop_last=True)
        data_loader_trainHeart = DataLoader(dataset_trainHeart, batch_sampler=batch_sampler_trainHeart, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_valHeart = DataLoader(dataset_valHeart, 1, sampler=sampler_valHeart, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

        ## Left Lung
        dataset_trainLeftLung = build_dataset(image_set='vindrcxrOrgan_trainLeftLung', args=args)
        dataset_valLeftLung = build_dataset(image_set='vindrcxrOrgan_testLeftLung', args=args)
        if args.distributed:
            sampler_trainLeftLung = DistributedSampler(dataset_trainLeftLung)
            sampler_valLeftLung = DistributedSampler(dataset_valLeftLung, shuffle=False)
        else:
            sampler_trainLeftLung = torch.utils.data.RandomSampler(dataset_trainLeftLung)
            sampler_valLeftLung = torch.utils.data.SequentialSampler(dataset_valLeftLung)
        batch_sampler_trainLeftLung = torch.utils.data.BatchSampler(sampler_trainLeftLung, args.batch_size, drop_last=True)
        data_loader_trainLeftLung = DataLoader(dataset_trainLeftLung, batch_sampler=batch_sampler_trainLeftLung, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_valLeftLung = DataLoader(dataset_valLeftLung, 1, sampler=sampler_valLeftLung, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

        ## Right Lung
        dataset_trainRightLung = build_dataset(image_set='vindrcxrOrgan_trainRightLung', args=args)
        dataset_valRightLung = build_dataset(image_set='vindrcxrOrgan_testRightLung', args=args)
        if args.distributed:
            sampler_trainRightLung = DistributedSampler(dataset_trainRightLung)
            sampler_valRightLung = DistributedSampler(dataset_valRightLung, shuffle=False)
        else:
            sampler_trainRightLung = torch.utils.data.RandomSampler(dataset_trainRightLung)
            sampler_valRightLung = torch.utils.data.SequentialSampler(dataset_valRightLung)
        batch_sampler_trainRightLung = torch.utils.data.BatchSampler(sampler_trainRightLung, args.batch_size, drop_last=True)
        data_loader_trainRightLung = DataLoader(dataset_trainRightLung, batch_sampler=batch_sampler_trainRightLung, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_valRightLung = DataLoader(dataset_valRightLung, 1, sampler=sampler_valRightLung, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

        # del dataset_trainHeart, sampler_valHeart, batch_sampler_trainHeart, dataset_trainLeftLung, sampler_valLeftLung, batch_sampler_trainLeftLung, dataset_trainRightLung, sampler_valRightLung, batch_sampler_trainRightLung

        # return train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLeftLung, val_loader_vindrcxrtLeftLung, train_loader_vindrcxrRightLung, val_loader_vindrcxrtRightLung, data_loader_trainHeart, data_loader_valHeart, sampler_trainHeart, dataset_valHeart, data_loader_trainLeftLung, data_loader_valLeftLung, sampler_trainLeftLung, dataset_valLeftLung, data_loader_trainRightLung, data_loader_valRightLung, sampler_trainRightLung, dataset_valRightLung
        return train_loader_vindrcxrHeart, val_loader_vindrcxrtHeart, train_loader_vindrcxrLeftLung, val_loader_vindrcxrtLeftLung, train_loader_vindrcxrRightLung, val_loader_vindrcxrtRightLung, \
            data_loader_trainHeart, sampler_trainHeart, data_loader_valHeart, dataset_valHeart, \
            data_loader_trainLeftLung, sampler_trainLeftLung, data_loader_valLeftLung, dataset_valLeftLung, \
            data_loader_trainRightLung, sampler_trainRightLung, data_loader_valRightLung, dataset_valRightLung



    if args.taskcomponent in ["detect_vindrcxr_heart_segTest", "detect_vindrcxr_heart", "detect_vindrcxr_leftlung", "detect_vindrcxr_rightlung"]: 
        ### Localization Data Loading
        if args.taskcomponent == "detect_vindrcxr_heart":
            dataset_train = build_dataset(image_set='vindrcxrOrgan_trainHeart', args=args)
            dataset_val = build_dataset(image_set='vindrcxrOrgan_testHeart', args=args)
        elif args.taskcomponent == "detect_vindrcxr_leftlung":
            dataset_train = build_dataset(image_set='vindrcxrOrgan_trainLeftLung', args=args)
            dataset_val = build_dataset(image_set='vindrcxrOrgan_testLeftLung', args=args)
        elif args.taskcomponent == "detect_vindrcxr_rightlung":
            dataset_train = build_dataset(image_set='vindrcxrOrgan_trainRightLung', args=args)
            dataset_val = build_dataset(image_set='vindrcxrOrgan_testRightLung', args=args)


        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)


        # if args.serverC == 'DFS':
        #     train_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
        #     test_image_path_file = [("/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
        # elif args.serverC == "SOL":
        #     train_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_train.txt")]
        #     test_image_path_file = [("/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0", "data/vindrcxr_organ_segmentation/vindrcxr_organSegmentation_test.txt")]
        
        # train_dataset = VindrCXRHeartDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize), ann=2500)
        # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

        # val_dataset = VindrCXRHeartDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=False )

        return data_loader_train, data_loader_val, sampler_train, dataset_val



    if args.taskcomponent == 'classification':
        if args.classification_dataset == 'imagenet':
           normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
               train_loader = torch.utils.data.DataLoader(
                   train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                   num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)
           valdir = os.path.join("/scratch/jliang12/data/ImageNet", 'val') # /data/jliang12/rfeng12/Data/ImageNet  /scratch/jliang12/data/ImageNet
           val_loader = torch.utils.data.DataLoader(
               datasets.ImageFolder(valdir, transforms.Compose([
                   transforms.Resize(256),
                   transforms.CenterCrop(img_size),
                   transforms.ToTensor(),
                   normalize,
               ])),
               batch_size=args.batch_size, shuffle=False,
               num_workers=args.num_workers, pin_memory=True)
           return train_loader, val_loader


        if args.classification_dataset == "ChestXray14":
            diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
            train_list = 'data/xray14/official/train_official.txt'
            val_list = 'data/xray14/official/val_official.txt'
            test_list = 'data/xray14/official/test_official.txt'
            if args.serverC == 'DFS':
                data_dir = "/mnt/dfs/nuislam/Data/ChestXRay14_images/" #"/mnt/dfs/nuislam/Data/ChestXRay14_images/" ## "/data/jliang12/jpang12/dataset/nih_xray14/images/images/"
            elif args.serverC == "SOL":
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

            return train_loader, val_loader, test_loader

    if args.taskcomponent in ['detection']: #'detection_vindrcxr_disease'
        if args.dataset_file == "coco":
            dataset_train = build_dataset(image_set='train', args=args)
            dataset_val = build_dataset(image_set='val', args=args)
            # dataset_test = build_dataset(image_set='test', args=args) # added by Nahid
        elif args.dataset_file == "chestxdetdataset":
            dataset_train = build_dataset(image_set='chestxdet_train', args=args)
            dataset_val = build_dataset(image_set='chestxdet_test', args=args)
        elif args.dataset_file == "vindrcxr_detect":
            dataset_train = build_dataset(image_set='vindrcxr_train', args=args)
            dataset_val = build_dataset(image_set='vindrcxr_test', args=args)
            # args.num_classes = 23
            # args.dn_labelbook_size = 24
        elif args.dataset_file == "vindrcxr_OrganDetect":
            dataset_train = build_dataset(image_set='vindrcxrOrgan_train', args=args)
            dataset_val = build_dataset(image_set='vindrcxrOrgan_test', args=args)
            args.num_classes = 3
            args.dn_labelbook_size = 4

        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        # data_loader_test = DataLoader(dataset_test, 1, sampler=sampler_test,
        #                              drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

        return data_loader_train, data_loader_val, sampler_train, dataset_val
    
    if args.taskcomponent in ['detect_node21_nodule']:
        ## Dataloader for Classification
        train_list = 'data/node21_dataset/train.txt'
        test_list = 'data/node21_dataset/test.txt'

        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/NODE21_ann/"
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/NODE21/cxr_images/proccessed_data/"

        dataset_train = NODE21(images_path=data_dir, file_path=train_list,
                                            augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = NODE21(images_path=data_dir, file_path=test_list,
                                            augment=build_transform_classification(normalize="chestx-ray", mode="test2"))

        train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)


        ## Dataloader for Localization
        dataset_train = build_dataset(image_set='node21_noduleDataset_train', args=args)
        dataset_val = build_dataset(image_set='node21_noduleDataset_test', args=args)
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        
        return data_loader_train, data_loader_val, sampler_train, dataset_val, train_loader, test_loader
    

    if args.taskcomponent in ['ClsLoc_tbx11k_catagnostic']:
        ## Dataloader for Classification
        train_list = 'lists/TBX11K_train.txt'
        test_list = 'lists/TBX11K_val.txt'

        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/jpang12/datasets/tbx11k/TBX11K/"
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/tbx11k/tbx11k/TBX11K/"

        dataset_train = TBX11KDataset(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = TBX11KDataset(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))

        train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


        ## Dataloader for Localization
        dataset_train = build_dataset(image_set='tbx11k_catagnostic_train', args=args)
        dataset_val = build_dataset(image_set='tbx11k_catagnostic_test', args=args)
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        
        return data_loader_train, data_loader_val, sampler_train, dataset_val, train_loader, test_loader
    
        
    if args.taskcomponent in ['detection_vindrcxr_disease']:
        ## Dataloader for Classification
        # train_list = 'data/vindrcxr/image_labels_train.txt'
        # test_list = 'data/vindrcxr/image_labels_test.txt'

        train_list = 'data/vindrcxr/image_labels_train_consolidated.txt'
        test_list = 'data/vindrcxr/image_labels_test_consolidated.txt'

        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/jpang12/datasets/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/"
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/"

        dataset_train = VindrCXRClass(images_path=os.path.join(data_dir,"train_jpeg"), file_path=train_list,
                                            augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = VindrCXRClass(images_path=os.path.join(data_dir,"test_jpeg"), file_path=test_list,
                                            augment=build_transform_classification(normalize="chestx-ray", mode="test2"))

        train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)


        ## Dataloader for Localization
        dataset_train = build_dataset(image_set='vindrcxr_train', args=args)
        dataset_val = build_dataset(image_set='vindrcxr_test', args=args)
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        

        return data_loader_train, data_loader_val, sampler_train, dataset_val, train_loader, test_loader
        
    if args.taskcomponent in ['detection_vindrmammo_disease']:
        ## Dataloader for Classification
        train_list = 'data/vindrmammo/Vindr_mammo_train_result.txt'
        test_list = 'data/vindrmammo/Vindr_mammo_test_result.txt'

        if args.serverC == 'DFS':
            data_dir = "/scratch/jliang12/data/Vindr-Mammo/physionet.org/files/vindr-mammo/1.0.0/"
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/Vindr-Mammo/physionet.org/files/vindr-mammo/1.0.0/"

        dataset_train = VindrmammoClass(images_path=os.path.join(data_dir,"images_png"), file_path=train_list,
                                            augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = VindrmammoClass(images_path=os.path.join(data_dir,"images_png"), file_path=test_list,
                                            augment=build_transform_classification(normalize="chestx-ray", mode="test2"))

        train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True,
                                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)


        ## Dataloader for Localization
        dataset_train = build_dataset(image_set='vindrmammo_train', args=args)
        dataset_val = build_dataset(image_set='vindrmammo_test', args=args)
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        

        return data_loader_train, data_loader_val, sampler_train, dataset_val, train_loader, test_loader
    
    if args.taskcomponent in ['detect_chestxdet_dataset']:
        ## Segmentation
        if args.serverC == 'DFS':
            train_image_path_file = [("/mnt/dfs/nuislam/Data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_train_data.txt")]
            # val_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "/scratch/jliang12/data/ChestX-Det/data_files/ChestX-Det_valid_data.txt")]
            test_image_path_file = [("/mnt/dfs/nuislam/Data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_test_data.txt")]
        elif args.serverC == "SOL":
            train_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_train_data.txt")]
            # val_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "/scratch/jliang12/data/ChestX-Det/data_files/ChestX-Det_valid_data.txt")]
            test_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_test_data.txt")]

        train_dataset = ChestXDetDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize))
        train_loader_seg = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                    num_workers=args.num_workers, pin_memory=True, shuffle=True,drop_last=True )

        # val_dataset = ChestXDetDataset(val_image_path_file,image_size=(args.imgsize,args.imgsize), mode="val")
        # val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        #                                          pin_memory=True, shuffle=True,drop_last=False )

        test_dataset = ChestXDetDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")

        test_loader_seg = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                                    num_workers=args.num_workers, pin_memory=True,drop_last=False )

        ## Dataloader for Classification
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/ChestX-Det/"
            train_list = '/mnt/dfs/nuislam/Data/ChestX-Det/ChestX_det_train_NAD_v2.json'
            test_list = '/mnt/dfs/nuislam/Data/ChestX-Det/ChestX_det_test_NAD_v2.json'

        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/ChestX-Det/"
            train_list = '/scratch/jliang12/data/ChestX-Det/ChestX_det_train_NAD_v2.json'
            test_list = '/scratch/jliang12/data/ChestX-Det/ChestX_det_test_NAD_v2.json'

        dataset_train = ChestXDet_cls(images_path=data_dir, file_path="train", augment=build_transform_classification(normalize="chestx-ray", mode="train"), anno_percent=100)
        dataset_test = ChestXDet_cls(images_path=data_dir, file_path="test", augment=build_transform_classification(normalize="chestx-ray", mode="test2"))

        train_loader_cls = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


        ## Dataloader for Localization
        dataset_train = build_dataset(image_set='chestxdet_train', args=args)
        dataset_val = build_dataset(image_set='chestxdet_test', args=args)
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid

        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        # data_loader_test = DataLoader(dataset_test, 1, sampler=sampler_test, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

        return data_loader_train, data_loader_val, sampler_train, dataset_val, train_loader_seg, test_loader_seg, train_loader_cls, test_loader_cls
        # return data_loader_train, data_loader_val, sampler_train, train_loader_seg, test_loader_seg
    


    if args.taskcomponent in ['foundation_x_pretraining']:
        ## TaskHead_number  0 = TBX11k Classification
        ## TaskHead_number  1 = TBX11k Localization
        ## TaskHead_number  2 = NODE21 Classification
        ## TaskHead_number  3 = NODE21 Localization
        ## TaskHead_number  4 = CANDID-PTX Classification
        ## TaskHead_number  5 = CANDID-PTX Localization
        ## TaskHead_number  6 = CANDID-PTX Segmentation
        ## TaskHead_number  7 = RSNA_Pneumonia Classification
        ## TaskHead_number  8 = RSNA_Pneumonia Localization
        ## TaskHead_number  9 = ChestX-Det Classification
        ## TaskHead_number 10 = ChestX-Det Localization
        ## TaskHead_number 11 = ChestX-Det Segmentation
        ## TaskHead_number 12 = SIIM-ACR-Pneumothorax Classification
        ## TaskHead_number 13 = SIIM-ACR-Pneumothorax Localization
        ## TaskHead_number 14 = SIIM-ACR-Pneumothorax Segmentation
        
        ## Dataloader for Classification -------------------------------------------------------------
        ## -- TBX11K Dataset -- ##
        train_list = 'lists/TBX11K_train.txt'
        test_list = 'lists/TBX11K_val.txt'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/jpang12/datasets/tbx11k/TBX11K/"
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/tbx11k/tbx11k/TBX11K/"
        dataset_train = TBX11KDataset(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = TBX11KDataset(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_TBX11k = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_TBX11k = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ## -- NODE21 Dataset -- ##
        train_list = 'data/node21_dataset/train.txt'
        test_list = 'data/node21_dataset/test.txt'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/NODE21_ann/"
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/NODE21/cxr_images/proccessed_data/"
        dataset_train = NODE21(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = NODE21(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_NODE21 = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_NODE21 = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ## -- ChestX-Det Dataset -- ##
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/ChestX-Det/"
            train_list = '/mnt/dfs/nuislam/Data/ChestX-Det/ChestX_det_train_NAD_v2.json'
            test_list = '/mnt/dfs/nuislam/Data/ChestX-Det/ChestX_det_test_NAD_v2.json'
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/ChestX-Det/"
            train_list = '/scratch/jliang12/data/ChestX-Det/ChestX_det_train_NAD_v2.json'
            test_list = '/scratch/jliang12/data/ChestX-Det/ChestX_det_test_NAD_v2.json'
        dataset_train = ChestXDet_cls(images_path=data_dir, file_path="train", augment=build_transform_classification(normalize="chestx-ray", mode="train"), anno_percent=100)
        dataset_test = ChestXDet_cls(images_path=data_dir, file_path="test", augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_ChestXDet = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_ChestXDet = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ## -- RSNAPneumonia Dataset -- ##
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/rsna_pneumonia/RSNAPneumonia_train.txt'
        valid_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/rsna_pneumonia/RSNAPneumonia_val.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/rsna_pneumonia/RSNAPneumonia_test.txt'
        dataset_train = RSNAPneumonia(images_path="/scratch/jliang12/data/rsna-pneumonia-detection-challenge/stage_2_train_images_png/", file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"))
        dataset_test = RSNAPneumonia(images_path="/scratch/jliang12/data/rsna-pneumonia-detection-challenge/stage_2_test_images_png/", file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_RSNApneumonia = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_RSNApneumonia = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ## -- SIIMPTX Dataset -- ##
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_train.txt'
        valid_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_val.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_test.txt'
        dataset_train = SIIMPTX(images_path="/scratch/jliang12/data/siim_pneumothorax_segmentation/", file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"))
        dataset_test = SIIMPTX(images_path="/scratch/jliang12/data/siim_pneumothorax_segmentation/", file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_SIIMACRptx = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_SIIMACRptx = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ## -- CANDID-PTX Dataset -- ##
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/candid_ptx/CANDIDPTX_cls_train.txt'
        valid_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/candid_ptx/CANDIDPTX_cls_val.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/candid_ptx/CANDIDPTX_cls_test.txt'
        dataset_train = CANDIDPTX(images_path="/scratch/jliang12/data/CANDID-PTX/dataset/", file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"))
        dataset_test = CANDIDPTX(images_path="/scratch/jliang12/data/CANDID-PTX/dataset/", file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_CANDIDptx = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_CANDIDptx = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)



        ## Dataloader for Localization -------------------------------------------------------------
        ## -- TBX11K Dataset -- ##
        dataset_train_loc_TBX11k = build_dataset(image_set='tbx11k_catagnostic_train_A', args=args) ## TBX11K Dataset - Training A set # tbx11k_catagnostic_train | tbx11k_catagnostic_train_A | tbx11k_catagnostic_train_B
        dataset_val_loc_TBX11k = build_dataset(image_set='tbx11k_catagnostic_test', args=args)
        if args.distributed:
            sampler_train_TBX11k = DistributedSampler(dataset_train_loc_TBX11k)
            sampler_val = DistributedSampler(dataset_val_loc_TBX11k, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train_TBX11k = torch.utils.data.RandomSampler(dataset_train_loc_TBX11k)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_TBX11k)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_TBX11k, args.batch_size, drop_last=True)
        train_loader_loc_TBX11k = DataLoader(dataset_train_loc_TBX11k, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_TBX11k = DataLoader(dataset_val_loc_TBX11k, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        dataset_train_loc_TBX11k_B = build_dataset(image_set='tbx11k_catagnostic_train_B', args=args) ## TBX11K Dataset - Training B set
        if args.distributed:
            sampler_train_TBX11k_B = DistributedSampler(dataset_train_loc_TBX11k_B)
        else:
            sampler_train_TBX11k_B = torch.utils.data.RandomSampler(dataset_train_loc_TBX11k_B)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_TBX11k_B, args.batch_size, drop_last=True)
        train_loader_loc_TBX11k_B = DataLoader(dataset_train_loc_TBX11k_B, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        
        ## -- NODE21 Dataset -- ##
        dataset_train_loc_Node21 = build_dataset(image_set='node21_noduleDataset_train_A', args=args) ## NODE21 Dataset - Training A set # node21_noduleDataset_train | node21_noduleDataset_train_A | node21_noduleDataset_train_B
        dataset_val_loc_Node21 = build_dataset(image_set='node21_noduleDataset_test', args=args)
        if args.distributed:
            sampler_train_Node21 = DistributedSampler(dataset_train_loc_Node21)
            sampler_val = DistributedSampler(dataset_val_loc_Node21, shuffle=False)
        else:
            sampler_train_Node21 = torch.utils.data.RandomSampler(dataset_train_loc_Node21)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_Node21)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_Node21, args.batch_size, drop_last=True)
        train_loader_loc_Node21 = DataLoader(dataset_train_loc_Node21, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        test_loader_loc_Node21 = DataLoader(dataset_val_loc_Node21, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

        dataset_train_loc_Node21_B = build_dataset(image_set='node21_noduleDataset_train_B', args=args) ## NODE21 Dataset - Training B set
        if args.distributed:
            sampler_train_Node21_B = DistributedSampler(dataset_train_loc_Node21_B)
        else:
            sampler_train_Node21_B = torch.utils.data.RandomSampler(dataset_train_loc_Node21_B)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_Node21_B, args.batch_size, drop_last=True)
        train_loader_loc_Node21_B = DataLoader(dataset_train_loc_Node21_B, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        ## -- CANDID-PTX Dataset -- ##
        dataset_train_loc_CANDIDptx = build_dataset(image_set='candidptx_pneumothorax_train_A', args=args) ## CANDID-PTX Dataset - Training A set # candidptx_pneumothorax_train_full | candidptx_pneumothorax_train_A | candidptx_pneumothorax_train_B
        dataset_val_loc_CANDIDptx = build_dataset(image_set='candidptx_pneumothorax_val', args=args) # candidptx_pneumothorax_val | candidptx_pneumothorax_test
        if args.distributed:
            sampler_train_CANDIDptx = DistributedSampler(dataset_train_loc_CANDIDptx)
            sampler_val = DistributedSampler(dataset_val_loc_CANDIDptx, shuffle=False)
        else:
            sampler_train_CANDIDptx = torch.utils.data.RandomSampler(dataset_train_loc_CANDIDptx)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_CANDIDptx)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_CANDIDptx, args.batch_size, drop_last=True)
        train_loader_loc_CANDIDptx = DataLoader(dataset_train_loc_CANDIDptx, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_CANDIDptx = DataLoader(dataset_val_loc_CANDIDptx, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        dataset_train_loc_CANDIDptx_B = build_dataset(image_set='candidptx_pneumothorax_train_B', args=args) ## CANDID-PTX Dataset - Training B set
        if args.distributed:
            sampler_train_CANDIDptx_B = DistributedSampler(dataset_train_loc_CANDIDptx_B)
        else:
            sampler_train_CANDIDptx_B = torch.utils.data.RandomSampler(dataset_train_loc_CANDIDptx_B)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_CANDIDptx_B, args.batch_size, drop_last=True)
        train_loader_loc_CANDIDptx_B = DataLoader(dataset_train_loc_CANDIDptx_B, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        ## -- ChestX-Det Dataset -- ##
        dataset_train_loc_ChestXDet = build_dataset(image_set='chestxdet_train_A', args=args) ## ChestX-Det Dataset - Training A set # chestxdet_train | chestxdet_train_A | chestxdet_train_B
        dataset_val_loc_ChestXDet = build_dataset(image_set='chestxdet_test', args=args)
        if args.distributed:
            sampler_train_ChestXDet = DistributedSampler(dataset_train_loc_ChestXDet)
            sampler_val = DistributedSampler(dataset_val_loc_ChestXDet, shuffle=False)
        else:
            sampler_train_ChestXDet = torch.utils.data.RandomSampler(dataset_train_loc_ChestXDet)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_ChestXDet)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_ChestXDet, args.batch_size, drop_last=True)
        train_loader_loc_ChestXDet = DataLoader(dataset_train_loc_ChestXDet, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_ChestXDet = DataLoader(dataset_val_loc_ChestXDet, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        dataset_train_loc_ChestXDet_B = build_dataset(image_set='chestxdet_train_B', args=args) ## ChestX-Det Dataset - Training B set
        if args.distributed:
            sampler_train_ChestXDet_B = DistributedSampler(dataset_train_loc_ChestXDet_B)
        else:
            sampler_train_ChestXDet_B = torch.utils.data.RandomSampler(dataset_train_loc_ChestXDet_B)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_ChestXDet_B, args.batch_size, drop_last=True)
        train_loader_loc_ChestXDet_B = DataLoader(dataset_train_loc_ChestXDet_B, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        ## -- RSNA Pneumonia Challenge Dataset -- ##
        dataset_train_loc_RSNApneumonia = build_dataset(image_set='rsnaPneumoniaDetection_Train_A', args=args) ## RSNApneumonia Dataset - Training A set # rsnaPneumoniaDetection_Train | rsnaPneumoniaDetection_Train_A | rsnaPneumoniaDetection_Train_B
        dataset_val_loc_RSNApneumonia = build_dataset(image_set='rsnaPneumoniaDetection_Valid', args=args)
        if args.distributed:
            sampler_train_RSNApneumonia = DistributedSampler(dataset_train_loc_RSNApneumonia)
            sampler_val = DistributedSampler(dataset_val_loc_RSNApneumonia, shuffle=False)
        else:
            sampler_train_RSNApneumonia = torch.utils.data.RandomSampler(dataset_train_loc_RSNApneumonia)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_RSNApneumonia)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_RSNApneumonia, args.batch_size, drop_last=True)
        train_loader_loc_RSNApneumonia = DataLoader(dataset_train_loc_RSNApneumonia, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_RSNApneumonia = DataLoader(dataset_val_loc_RSNApneumonia, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        dataset_train_loc_RSNApneumonia_B = build_dataset(image_set='rsnaPneumoniaDetection_Train_B', args=args) ## RSNApneumonia Dataset - Training B set
        if args.distributed:
            sampler_train_RSNApneumonia_B = DistributedSampler(dataset_train_loc_RSNApneumonia_B)
        else:
            sampler_train_RSNApneumonia_B = torch.utils.data.RandomSampler(dataset_train_loc_RSNApneumonia_B)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_RSNApneumonia_B, args.batch_size, drop_last=True)
        train_loader_loc_RSNApneumonia_B = DataLoader(dataset_train_loc_RSNApneumonia_B, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        ## -- SIIM-ACR Dataset -- ##
        dataset_train_loc_SiimACR = build_dataset(image_set='siimacr_train_A', args=args) ## RSNApneumonia Dataset - Training A set # rsnaPneumoniaDetection_Train | rsnaPneumoniaDetection_Train_A | rsnaPneumoniaDetection_Train_B
        dataset_val_loc_SiimACR = build_dataset(image_set='siimacr_val', args=args)
        if args.distributed:
            sampler_train_SiimACR = DistributedSampler(dataset_train_loc_SiimACR)
            sampler_val = DistributedSampler(dataset_val_loc_SiimACR, shuffle=False)
        else:
            sampler_train_SiimACR = torch.utils.data.RandomSampler(dataset_train_loc_SiimACR)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_SiimACR)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_SiimACR, args.batch_size, drop_last=True)
        train_loader_loc_SiimACR = DataLoader(dataset_train_loc_SiimACR, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_SiimACR = DataLoader(dataset_val_loc_SiimACR, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        dataset_train_loc_SiimACR_B = build_dataset(image_set='siimacr_train_B', args=args) ## RSNApneumonia Dataset - Training B set
        if args.distributed:
            sampler_train_SiimACR_B = DistributedSampler(dataset_train_loc_SiimACR_B)
        else:
            sampler_train_SiimACR_B = torch.utils.data.RandomSampler(dataset_train_loc_SiimACR_B)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_SiimACR_B, args.batch_size, drop_last=True)
        train_loader_loc_SiimACR_B = DataLoader(dataset_train_loc_SiimACR_B, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        ## -- CheXMask-VinDrCXR Organ (H LL RL) Localization Dataset -- ##
        # dataset_train_loc_chexmaskVinDrCXRorgan = build_dataset(image_set='vindrcxrOrgan_train_A', args=args) ## CheXMask-VinDrCXR Organ - Training A set # vindrcxrOrgan_train | vindrcxrOrgan_train_A | vindrcxrOrgan_train_B
        # dataset_val_loc_chexmaskVinDrCXRorgan = build_dataset(image_set='vindrcxrOrgan_test', args=args)
        # if args.distributed:
        #     sampler_train_chexmaskVinDrCXRorgan = DistributedSampler(dataset_train_loc_chexmaskVinDrCXRorgan)
        #     sampler_val = DistributedSampler(dataset_val_loc_chexmaskVinDrCXRorgan, shuffle=False)
        # else:
        #     sampler_train_chexmaskVinDrCXRorgan = torch.utils.data.RandomSampler(dataset_train_loc_chexmaskVinDrCXRorgan)
        #     sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_chexmaskVinDrCXRorgan)
        # batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_chexmaskVinDrCXRorgan, args.batch_size, drop_last=True)
        # train_loader_loc_chexmaskVinDrCXRorgan = DataLoader(dataset_train_loc_chexmaskVinDrCXRorgan, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        # test_loader_loc_chexmaskVinDrCXRorgan = DataLoader(dataset_val_loc_chexmaskVinDrCXRorgan, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        # dataset_train_loc_chexmaskVinDrCXRorgan_B = build_dataset(image_set='vindrcxrOrgan_train_B', args=args) ## CheXMask-VinDrCXR Organ - Training B set
        # if args.distributed:
        #     sampler_train_chexmaskVinDrCXRorgan_B = DistributedSampler(dataset_train_loc_chexmaskVinDrCXRorgan_B)
        # else:
        #     sampler_train_RSNApneumonia_B = torch.utils.data.RandomSampler(dataset_train_loc_chexmaskVinDrCXRorgan_B)
        # batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_RSNApneumonia_B, args.batch_size, drop_last=True)
        # train_loader_loc_chexmaskVinDrCXRorgan_B = DataLoader(dataset_train_loc_chexmaskVinDrCXRorgan_B, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN


        ## Segmentation -------------------------------------------------------------
        ## -- ChestX-Det Dataset -- ##
        if args.serverC == 'DFS':
            train_image_path_file = [("/mnt/dfs/nuislam/Data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_train_data.txt")]
            # val_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "/scratch/jliang12/data/ChestX-Det/data_files/ChestX-Det_valid_data.txt")]
            test_image_path_file = [("/mnt/dfs/nuislam/Data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_test_data.txt")]
        elif args.serverC == "SOL":
            train_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_train_data.txt")]
            # val_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "/scratch/jliang12/data/ChestX-Det/data_files/ChestX-Det_valid_data.txt")]
            test_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_test_data.txt")]
        # train_dataset = ChestXDetDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize))
        # train_loader_seg_ChestXDet = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=True )
        # test_dataset = ChestXDetDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
        # test_loader_seg_ChestXDet = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False )

        train_dataset = chestxdet_dataset(image_path="/scratch/jliang12/data/ChestX-Det/train/", masks_path="/scratch/jliang12/data/ChestX-Det/train_binary_mask/", image_size=(args.imgsize,args.imgsize), mode='train') ## ChestXDetDataset chestxdet_dataset
        train_loader_seg_ChestXDet = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=True )

        test_dataset = chestxdet_dataset(image_path="/scratch/jliang12/data/ChestX-Det/test/",masks_path="/scratch/jliang12/data/ChestX-Det/test_binary_mask/", image_size=(args.imgsize,args.imgsize), mode='test')
        test_loader_seg_ChestXDet = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False )


        train_dataset = SIIM_PXSDataset([("/scratch/jliang12/data/siim_pneumothorax_segmentation/train_jpeg","data/pxs/train.txt")], image_size=(args.imgsize,args.imgsize), mode="train")
        sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader_seg_SIIM = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler, drop_last=True)
        # valid_dataset = SIIM_PXSDataset([("/scratch/jliang12/data/siim_pneumothorax_segmentation/val_jpeg","data/pxs/val.txt")], image_size=(args.imgsize,args.imgsize), mode="valid")
        # sampler = torch.utils.data.RandomSampler(valid_dataset)
        # val_loader_seg_SIIM = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=True, sampler=sampler)
        test_dataset = SIIM_PXSDataset([("/scratch/jliang12/data/siim_pneumothorax_segmentation/test_jpeg","data/pxs/test.txt")], image_size=(args.imgsize,args.imgsize), mode="test")
        sampler = torch.utils.data.RandomSampler(test_dataset)
        test_loader_seg_SIIM = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler)


        train_dataset = Candid_PTX_PXSDataset([("/scratch/jliang12/data/CANDID-PTX/dataset","data/candid_ptx/train.txt")], image_size=(args.imgsize,args.imgsize), mode="train")
        sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader_seg_CANDIDptx = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler)
        # valid_dataset = Candid_PTX_PXSDataset([("/scratch/jliang12/data/CANDID-PTX/dataset","data/candid_ptx/valid.txt")], image_size=(args.imgsize,args.imgsize), mode="valid")
        # sampler = torch.utils.data.RandomSampler(valid_dataset)
        # val_loader_seg_CANDIDptx = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=True, sampler=sampler)
        test_dataset = Candid_PTX_PXSDataset([("/scratch/jliang12/data/CANDID-PTX/dataset","data/candid_ptx/test.txt")], image_size=(args.imgsize,args.imgsize), mode="test")
        sampler = torch.utils.data.RandomSampler(test_dataset)
        test_loader_seg_CANDIDptx = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler)


        return train_loader_cls_TBX11k, test_loader_cls_TBX11k, train_loader_cls_NODE21, test_loader_cls_NODE21, train_loader_cls_ChestXDet, test_loader_cls_ChestXDet, \
            train_loader_cls_RSNApneumonia, test_loader_cls_RSNApneumonia, train_loader_cls_SIIMACRptx, test_loader_cls_SIIMACRptx, train_loader_cls_CANDIDptx, test_loader_cls_CANDIDptx, \
            train_loader_loc_TBX11k, test_loader_loc_TBX11k, dataset_val_loc_TBX11k, sampler_train_TBX11k, train_loader_loc_TBX11k_B, \
            train_loader_loc_Node21, test_loader_loc_Node21, dataset_val_loc_Node21, sampler_train_Node21, train_loader_loc_Node21_B, \
            train_loader_loc_CANDIDptx, test_loader_loc_CANDIDptx, dataset_val_loc_CANDIDptx, sampler_train_CANDIDptx, train_loader_loc_CANDIDptx_B, \
            train_loader_loc_ChestXDet, test_loader_loc_ChestXDet, dataset_val_loc_ChestXDet, sampler_train_ChestXDet, train_loader_loc_ChestXDet_B, \
            train_loader_loc_RSNApneumonia, test_loader_loc_RSNApneumonia, dataset_val_loc_RSNApneumonia, sampler_train_RSNApneumonia, train_loader_loc_RSNApneumonia_B, \
            train_loader_loc_SiimACR, test_loader_loc_SiimACR, dataset_val_loc_SiimACR, sampler_train_SiimACR, train_loader_loc_SiimACR_B, \
            train_loader_seg_ChestXDet, test_loader_seg_ChestXDet, train_loader_seg_SIIM, test_loader_seg_SIIM, train_loader_seg_CANDIDptx, test_loader_seg_CANDIDptx

        # return train_loader_cls_TBX11k, test_loader_cls_TBX11k, train_loader_cls_NODE21, test_loader_cls_NODE21, train_loader_cls_ChestXDet, test_loader_cls_ChestXDet, \
        #     train_loader_cls_RSNApneumonia, test_loader_cls_RSNApneumonia, train_loader_cls_SIIMACRptx, test_loader_cls_SIIMACRptx, train_loader_cls_CANDIDptx, test_loader_cls_CANDIDptx, \
        #     train_loader_loc_TBX11k, test_loader_loc_TBX11k, dataset_val_loc_TBX11k, sampler_train_TBX11k, train_loader_loc_TBX11k_B, \
        #     train_loader_loc_Node21, test_loader_loc_Node21, dataset_val_loc_Node21, sampler_train_Node21, train_loader_loc_Node21_B, \
        #     train_loader_loc_CANDIDptx, test_loader_loc_CANDIDptx, dataset_val_loc_CANDIDptx, sampler_train_CANDIDptx, train_loader_loc_CANDIDptx_B, \
        #     train_loader_loc_ChestXDet, test_loader_loc_ChestXDet, dataset_val_loc_ChestXDet, sampler_train_ChestXDet, train_loader_loc_ChestXDet_B, \
        #     train_loader_loc_RSNApneumonia, test_loader_loc_RSNApneumonia, dataset_val_loc_RSNApneumonia, sampler_train_RSNApneumonia, train_loader_loc_RSNApneumonia_B, \
        #     # train_loader_loc_chexmaskVinDrCXRorgan, test_loader_loc_chexmaskVinDrCXRorgan, dataset_val_loc_chexmaskVinDrCXRorgan, sampler_train_chexmaskVinDrCXRorgan, train_loader_loc_chexmaskVinDrCXRorgan_B, \
        #     train_loader_seg_ChestXDet, test_loader_seg_ChestXDet, train_loader_seg_SIIM, test_loader_seg_SIIM, train_loader_seg_CANDIDptx, test_loader_seg_CANDIDptx
    



    if args.taskcomponent in ['foundation_x2_pretraining']:
        ## TaskHead_number  0 = TBX11k Classification
        ## TaskHead_number  1 = TBX11k Localization
        ## TaskHead_number  2 = NODE21 Classification
        ## TaskHead_number  3 = NODE21 Localization
        ## TaskHead_number  4 = CANDID-PTX Classification
        ## TaskHead_number  5 = CANDID-PTX Localization
        ## TaskHead_number  6 = CANDID-PTX Segmentation
        ## TaskHead_number  7 = RSNA_Pneumonia Classification
        ## TaskHead_number  8 = RSNA_Pneumonia Localization
        ## TaskHead_number  9 = ChestX-Det Classification
        ## TaskHead_number 10 = ChestX-Det Localization
        ## TaskHead_number 11 = ChestX-Det Segmentation
        ## TaskHead_number 12 = SIIM-ACR-Pneumothorax Classification
        ## TaskHead_number 13 = SIIM-ACR-Pneumothorax Localization
        ## TaskHead_number 14 = SIIM-ACR-Pneumothorax Segmentation
        
        ## Dataloader for Classification -------------------------------------------------------------
        ## -- TBX11K Dataset -- ##
        train_list = 'lists/TBX11K_train.txt'
        test_list = 'lists/TBX11K_val.txt'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/jpang12/datasets/tbx11k/TBX11K/"
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/tbx11k/tbx11k/TBX11K/"
        dataset_train = TBX11KDataset(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = TBX11KDataset(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_TBX11k = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_TBX11k = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ## -- NODE21 Dataset -- ##
        train_list = 'data/node21_dataset/train.txt'
        test_list = 'data/node21_dataset/test.txt'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/NODE21_ann/"
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/NODE21/cxr_images/proccessed_data/"
        dataset_train = NODE21(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = NODE21(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_NODE21 = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_NODE21 = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ## -- ChestX-Det Dataset -- ##
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/ChestX-Det/"
            train_list = '/mnt/dfs/nuislam/Data/ChestX-Det/ChestX_det_train_NAD_v2.json'
            test_list = '/mnt/dfs/nuislam/Data/ChestX-Det/ChestX_det_test_NAD_v2.json'
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/ChestX-Det/"
            train_list = '/scratch/jliang12/data/ChestX-Det/ChestX_det_train_NAD_v2.json'
            test_list = '/scratch/jliang12/data/ChestX-Det/ChestX_det_test_NAD_v2.json'
        dataset_train = ChestXDet_cls(images_path=data_dir, file_path="train", augment=build_transform_classification(normalize="chestx-ray", mode="train"), anno_percent=100)
        dataset_test = ChestXDet_cls(images_path=data_dir, file_path="test", augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_ChestXDet = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_ChestXDet = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ## -- RSNAPneumonia Dataset -- ##
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/rsna_pneumonia/RSNAPneumonia_train.txt'
        valid_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/rsna_pneumonia/RSNAPneumonia_val.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/rsna_pneumonia/RSNAPneumonia_test.txt'
        dataset_train = RSNAPneumonia(images_path="/scratch/jliang12/data/rsna-pneumonia-detection-challenge/stage_2_train_images_png/", file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"))
        dataset_test = RSNAPneumonia(images_path="/scratch/jliang12/data/rsna-pneumonia-detection-challenge/stage_2_train_images_png/", file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_RSNApneumonia = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_RSNApneumonia = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ## -- SIIMPTX Dataset -- ##
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_train.txt'
        valid_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_val.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_test.txt'
        dataset_train = SIIMPTX(images_path="/scratch/jliang12/data/siim_pneumothorax_segmentation/", file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"))
        dataset_test = SIIMPTX(images_path="/scratch/jliang12/data/siim_pneumothorax_segmentation/", file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_SIIMACRptx = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_SIIMACRptx = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ## -- CANDID-PTX Dataset -- ##
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/candid_ptx/CANDIDPTX_cls_train.txt'
        valid_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/candid_ptx/CANDIDPTX_cls_val.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/candid_ptx/CANDIDPTX_cls_test.txt'
        dataset_train = CANDIDPTX(images_path="/scratch/jliang12/data/CANDID-PTX/dataset/", file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"))
        dataset_test = CANDIDPTX(images_path="/scratch/jliang12/data/CANDID-PTX/dataset/", file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_CANDIDptx = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_CANDIDptx = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)



        ## Dataloader for Localization -------------------------------------------------------------
        ## -- TBX11K Dataset -- ##
        dataset_train_loc_TBX11k = build_dataset(image_set='tbx11k_catagnostic_train', args=args) ## TBX11K Dataset - Training A set # tbx11k_catagnostic_train | tbx11k_catagnostic_train_A | tbx11k_catagnostic_train_B
        dataset_val_loc_TBX11k = build_dataset(image_set='tbx11k_catagnostic_test', args=args)
        if args.distributed:
            sampler_train_TBX11k = DistributedSampler(dataset_train_loc_TBX11k, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_TBX11k, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train_TBX11k = torch.utils.data.RandomSampler(dataset_train_loc_TBX11k)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_TBX11k)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_TBX11k, args.batch_size, drop_last=True)
        train_loader_loc_TBX11k = DataLoader(dataset_train_loc_TBX11k, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_TBX11k = DataLoader(dataset_val_loc_TBX11k, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        
        ## -- NODE21 Dataset -- ##
        dataset_train_loc_Node21 = build_dataset(image_set='node21_noduleDataset_train', args=args) ## NODE21 Dataset - Training A set # node21_noduleDataset_train | node21_noduleDataset_train_A | node21_noduleDataset_train_B
        dataset_val_loc_Node21 = build_dataset(image_set='node21_noduleDataset_test', args=args)
        if args.distributed:
            sampler_train_Node21 = DistributedSampler(dataset_train_loc_Node21, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_Node21, shuffle=False)
        else:
            sampler_train_Node21 = torch.utils.data.RandomSampler(dataset_train_loc_Node21)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_Node21)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_Node21, args.batch_size, drop_last=True)
        train_loader_loc_Node21 = DataLoader(dataset_train_loc_Node21, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        test_loader_loc_Node21 = DataLoader(dataset_val_loc_Node21, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

        ## -- CANDID-PTX Dataset -- ##
        dataset_train_loc_CANDIDptx = build_dataset(image_set='candidptx_pneumothorax_train_full', args=args) ## CANDID-PTX Dataset - Training A set # candidptx_pneumothorax_train_full | candidptx_pneumothorax_train_A | candidptx_pneumothorax_train_B
        dataset_val_loc_CANDIDptx = build_dataset(image_set='candidptx_pneumothorax_val', args=args) # candidptx_pneumothorax_val | candidptx_pneumothorax_test
        if args.distributed:
            sampler_train_CANDIDptx = DistributedSampler(dataset_train_loc_CANDIDptx, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_CANDIDptx, shuffle=False)
        else:
            sampler_train_CANDIDptx = torch.utils.data.RandomSampler(dataset_train_loc_CANDIDptx)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_CANDIDptx)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_CANDIDptx, args.batch_size, drop_last=True)
        train_loader_loc_CANDIDptx = DataLoader(dataset_train_loc_CANDIDptx, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_CANDIDptx = DataLoader(dataset_val_loc_CANDIDptx, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        ## -- ChestX-Det Dataset -- ##
        dataset_train_loc_ChestXDet = build_dataset(image_set='chestxdet_train', args=args) ## ChestX-Det Dataset - Training A set # chestxdet_train | chestxdet_train_A | chestxdet_train_B
        dataset_val_loc_ChestXDet = build_dataset(image_set='chestxdet_test', args=args)
        if args.distributed:
            sampler_train_ChestXDet = DistributedSampler(dataset_train_loc_ChestXDet, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_ChestXDet, shuffle=False)
        else:
            sampler_train_ChestXDet = torch.utils.data.RandomSampler(dataset_train_loc_ChestXDet)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_ChestXDet)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_ChestXDet, args.batch_size, drop_last=True)
        train_loader_loc_ChestXDet = DataLoader(dataset_train_loc_ChestXDet, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_ChestXDet = DataLoader(dataset_val_loc_ChestXDet, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        ## -- RSNA Pneumonia Challenge Dataset -- ##
        dataset_train_loc_RSNApneumonia = build_dataset(image_set='rsnaPneumoniaDetection_Train', args=args) ## RSNApneumonia Dataset - Training A set # rsnaPneumoniaDetection_Train | rsnaPneumoniaDetection_Train_A | rsnaPneumoniaDetection_Train_B
        dataset_val_loc_RSNApneumonia = build_dataset(image_set='rsnaPneumoniaDetection_Valid', args=args)
        if args.distributed:
            sampler_train_RSNApneumonia = DistributedSampler(dataset_train_loc_RSNApneumonia, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_RSNApneumonia, shuffle=False)
        else:
            sampler_train_RSNApneumonia = torch.utils.data.RandomSampler(dataset_train_loc_RSNApneumonia)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_RSNApneumonia)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_RSNApneumonia, args.batch_size, drop_last=True)
        train_loader_loc_RSNApneumonia = DataLoader(dataset_train_loc_RSNApneumonia, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_RSNApneumonia = DataLoader(dataset_val_loc_RSNApneumonia, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        ## -- SIIM-ACR Dataset -- ##
        dataset_train_loc_SiimACR = build_dataset(image_set='siimacr_train', args=args) ## RSNApneumonia Dataset - Training A set # siimacr_train | siimacr_train_A | siimacr_train_B
        dataset_val_loc_SiimACR = build_dataset(image_set='siimacr_val', args=args)
        if args.distributed:
            sampler_train_SiimACR = DistributedSampler(dataset_train_loc_SiimACR, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_SiimACR, shuffle=False)
        else:
            sampler_train_SiimACR = torch.utils.data.RandomSampler(dataset_train_loc_SiimACR)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_SiimACR)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_SiimACR, args.batch_size, drop_last=True)
        train_loader_loc_SiimACR = DataLoader(dataset_train_loc_SiimACR, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_SiimACR = DataLoader(dataset_val_loc_SiimACR, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN


        ## Segmentation -------------------------------------------------------------
        ## -- ChestX-Det Dataset -- ##
        if args.serverC == 'DFS':
            train_image_path_file = [("/mnt/dfs/nuislam/Data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_train_data.txt")]
            # val_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "/scratch/jliang12/data/ChestX-Det/data_files/ChestX-Det_valid_data.txt")]
            test_image_path_file = [("/mnt/dfs/nuislam/Data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_test_data.txt")]
        elif args.serverC == "SOL":
            train_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_train_data.txt")]
            # val_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "/scratch/jliang12/data/ChestX-Det/data_files/ChestX-Det_valid_data.txt")]
            test_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_test_data.txt")]
        # train_dataset = ChestXDetDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize))
        # train_loader_seg_ChestXDet = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=True )
        # test_dataset = ChestXDetDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
        # test_loader_seg_ChestXDet = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False )

        train_dataset = Candid_PTX_PXSDataset([("/scratch/jliang12/data/CANDID-PTX/dataset","data/candid_ptx/train.txt")], image_size=(args.imgsize,args.imgsize), mode="train")
        sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader_seg_CANDIDptx = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler)
        # valid_dataset = Candid_PTX_PXSDataset([("/scratch/jliang12/data/CANDID-PTX/dataset","data/candid_ptx/valid.txt")], image_size=(args.imgsize,args.imgsize), mode="valid")
        # sampler = torch.utils.data.RandomSampler(valid_dataset)
        # val_loader_seg_CANDIDptx = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=True, sampler=sampler)
        test_dataset = Candid_PTX_PXSDataset([("/scratch/jliang12/data/CANDID-PTX/dataset","data/candid_ptx/test.txt")], image_size=(args.imgsize,args.imgsize), mode="test")
        sampler = torch.utils.data.RandomSampler(test_dataset)
        test_loader_seg_CANDIDptx = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)


        train_dataset = chestxdet_dataset(image_path="/scratch/jliang12/data/ChestX-Det/train/", masks_path="/scratch/jliang12/data/ChestX-Det/train_binary_mask/", image_size=(args.imgsize,args.imgsize), mode='train') ## ChestXDetDataset chestxdet_dataset
        train_loader_seg_ChestXDet = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=True )

        test_dataset = chestxdet_dataset(image_path="/scratch/jliang12/data/ChestX-Det/test/",masks_path="/scratch/jliang12/data/ChestX-Det/test_binary_mask/", image_size=(args.imgsize,args.imgsize), mode='test')
        test_loader_seg_ChestXDet = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False )


        train_dataset = SIIM_PXSDataset([("/scratch/jliang12/data/siim_pneumothorax_segmentation/train_jpeg","data/pxs/train.txt")], image_size=(args.imgsize,args.imgsize), mode="train")
        sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader_seg_SIIM = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler, drop_last=True)
        # valid_dataset = SIIM_PXSDataset([("/scratch/jliang12/data/siim_pneumothorax_segmentation/val_jpeg","data/pxs/val.txt")], image_size=(args.imgsize,args.imgsize), mode="valid")
        # sampler = torch.utils.data.RandomSampler(valid_dataset)
        # val_loader_seg_SIIM = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=True, sampler=sampler)
        test_dataset = SIIM_PXSDataset([("/scratch/jliang12/data/siim_pneumothorax_segmentation/test_jpeg","data/pxs/test.txt")], image_size=(args.imgsize,args.imgsize), mode="test")
        sampler = torch.utils.data.RandomSampler(test_dataset)
        test_loader_seg_SIIM = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)





        return train_loader_cls_TBX11k, test_loader_cls_TBX11k, train_loader_cls_NODE21, test_loader_cls_NODE21, train_loader_cls_ChestXDet, test_loader_cls_ChestXDet, \
            train_loader_cls_RSNApneumonia, test_loader_cls_RSNApneumonia, train_loader_cls_SIIMACRptx, test_loader_cls_SIIMACRptx, train_loader_cls_CANDIDptx, test_loader_cls_CANDIDptx, \
            train_loader_loc_TBX11k, test_loader_loc_TBX11k, dataset_val_loc_TBX11k, sampler_train_TBX11k, \
            train_loader_loc_Node21, test_loader_loc_Node21, dataset_val_loc_Node21, sampler_train_Node21, \
            train_loader_loc_CANDIDptx, test_loader_loc_CANDIDptx, dataset_val_loc_CANDIDptx, sampler_train_CANDIDptx, \
            train_loader_loc_ChestXDet, test_loader_loc_ChestXDet, dataset_val_loc_ChestXDet, sampler_train_ChestXDet, \
            train_loader_loc_RSNApneumonia, test_loader_loc_RSNApneumonia, dataset_val_loc_RSNApneumonia, sampler_train_RSNApneumonia, \
            train_loader_loc_SiimACR, test_loader_loc_SiimACR, dataset_val_loc_SiimACR, sampler_train_SiimACR, \
            train_loader_seg_ChestXDet, test_loader_seg_ChestXDet, train_loader_seg_SIIM, test_loader_seg_SIIM, train_loader_seg_CANDIDptx, test_loader_seg_CANDIDptx
    

    if args.taskcomponent in ['foundation_x3_pretraining']:       
        ## Dataloader for Classification -------------------------------------------------------------
        train_list = 'data/xray14/official/train_official.txt'
        val_list = 'data/xray14/official/val_official.txt'
        test_list = 'data/xray14/official/test_official.txt'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/ChestXRay14_images/" #"/mnt/dfs/nuislam/Data/ChestXRay14_images/" ## "/data/jliang12/jpang12/dataset/nih_xray14/images/images/"
        elif args.serverC == "SOL":
            # data_dir = "/data/jliang12/jpang12/dataset/nih_xray14/images/images/" 
            data_dir = "/scratch/jliang12/data/nih_xray14/images/images/"
        dataset_train = ChestXray14Dataset(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        # dataset_val = ChestXray14Dataset(images_path=data_dir, file_path=val_list, augment=build_transform_classification(normalize="chestx-ray", mode="valid"))
        dataset_test = ChestXray14Dataset(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_NIHChestXray14 = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        # val_loader_cls_NIHChestXray14 = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_loader_cls_NIHChestXray14 = DataLoader(dataset=dataset_test, batch_size=args.batch_size//2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/CheXpert-v1.0_train.csv'
        val_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/CheXpert-v1.0_valid.csv'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/CheXpert-v1.0_valid.csv'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/ChestXRay14_images/" #"/mnt/dfs/nuislam/Data/ChestXRay14_images/" ## "/data/jliang12/jpang12/dataset/nih_xray14/images/images/"
        elif args.serverC == "SOL":
            # data_dir = "/data/jliang12/mhossei2/Dataset/" ## CheXpert-v1.0
            data_dir = "/scratch/jliang12/data/"
        dataset_train = CheXpert(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = CheXpert(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_CheXpert = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_CheXpert = DataLoader(dataset=dataset_test, batch_size=args.batch_size//2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/VinDrCXR_train_pe_global_one.txt'
        val_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/VinDrCXR_test_pe_global_one.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/VinDrCXR_test_pe_global_one.txt'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/ChestXRay14_images/" #"/mnt/dfs/nuislam/Data/ChestXRay14_images/" ## "/data/jliang12/jpang12/dataset/nih_xray14/images/images/"
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/"
        dataset_train = VinDrCXR(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = VinDrCXR(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_VinDRCXR = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_VinDRCXR = DataLoader(dataset=dataset_test, batch_size=args.batch_size//2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/ShenzenCXR_train_data.txt'
        val_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/ShenzenCXR_valid_data.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/ShenzenCXR_test_data.txt'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/ChestXRay14_images/" #"/mnt/dfs/nuislam/Data/ChestXRay14_images/" ## "/data/jliang12/jpang12/dataset/nih_xray14/images/images/"
        elif args.serverC == "SOL":
            # data_dir = "/data/jliang12/mhossei2/Dataset/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png/"
            data_dir = "/scratch/jliang12/data/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png/"
        dataset_train = ShenzhenCXR(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = ShenzhenCXR(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_NIHShenzhen = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_NIHShenzhen = DataLoader(dataset=dataset_test, batch_size=args.batch_size//2, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/mimic-cxr-2.0.0-train.csv'
        val_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/mimic-cxr-2.0.0-validate.csv'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/mimic-cxr-2.0.0-test.csv'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/ChestXRay14_images/" #"/mnt/dfs/nuislam/Data/ChestXRay14_images/" ## "/data/jliang12/jpang12/dataset/nih_xray14/images/images/"
        elif args.serverC == "SOL":
            # data_dir = "/data/jliang12/jpang12/dataset/MIMIC_jpeg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
            data_dir = "/scratch/jliang12/data/MIMIC_jpeg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
        dataset_train = MIMIC(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = MIMIC(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_MIMICII = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_MIMICII = DataLoader(dataset=dataset_test, batch_size=args.batch_size//2, shuffle=False, num_workers=args.num_workers, pin_memory=True)


        ## -- TBX11K Dataset -- ##
        train_list = 'lists/TBX11K_train.txt'
        test_list = 'lists/TBX11K_val.txt'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/jpang12/datasets/tbx11k/TBX11K/"
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/tbx11k/tbx11k/TBX11K/"
        dataset_train = TBX11KDataset(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = TBX11KDataset(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_TBX11k = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_TBX11k = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ## -- NODE21 Dataset -- ##
        train_list = 'data/node21_dataset/train.txt'
        test_list = 'data/node21_dataset/test.txt'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/NODE21_ann/"
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/NODE21/cxr_images/proccessed_data/"
        dataset_train = NODE21(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = NODE21(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_NODE21 = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_NODE21 = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ## -- ChestX-Det Dataset -- ##
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/ChestX-Det/"
            train_list = '/mnt/dfs/nuislam/Data/ChestX-Det/ChestX_det_train_NAD_v2.json'
            test_list = '/mnt/dfs/nuislam/Data/ChestX-Det/ChestX_det_test_NAD_v2.json'
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/ChestX-Det/"
            train_list = '/scratch/jliang12/data/ChestX-Det/ChestX_det_train_NAD_v2.json'
            test_list = '/scratch/jliang12/data/ChestX-Det/ChestX_det_test_NAD_v2.json'
        dataset_train = ChestXDet_cls(images_path=data_dir, file_path="train", augment=build_transform_classification(normalize="chestx-ray", mode="train"), anno_percent=100)
        dataset_test = ChestXDet_cls(images_path=data_dir, file_path="test", augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_ChestXDet = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_ChestXDet = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ## -- RSNAPneumonia Dataset -- ##
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/rsna_pneumonia/RSNAPneumonia_train.txt'
        valid_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/rsna_pneumonia/RSNAPneumonia_val.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/rsna_pneumonia/RSNAPneumonia_test.txt'
        dataset_train = RSNAPneumonia(images_path="/scratch/jliang12/data/rsna-pneumonia-detection-challenge/stage_2_train_images_png/", file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"))
        dataset_test = RSNAPneumonia(images_path="/scratch/jliang12/data/rsna-pneumonia-detection-challenge/stage_2_train_images_png/", file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_RSNApneumonia = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_RSNApneumonia = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ## -- SIIMPTX Dataset -- ##
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_train.txt'
        valid_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_val.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_test.txt'
        dataset_train = SIIMPTX(images_path="/scratch/jliang12/data/siim_pneumothorax_segmentation/", file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"))
        dataset_test = SIIMPTX(images_path="/scratch/jliang12/data/siim_pneumothorax_segmentation/", file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_SIIMACRptx = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_SIIMACRptx = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

        ## -- CANDID-PTX Dataset -- ##
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/candid_ptx/CANDIDPTX_cls_train.txt'
        valid_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/candid_ptx/CANDIDPTX_cls_val.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/candid_ptx/CANDIDPTX_cls_test.txt'
        dataset_train = CANDIDPTX(images_path="/scratch/jliang12/data/CANDID-PTX/dataset/", file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"))
        dataset_test = CANDIDPTX(images_path="/scratch/jliang12/data/CANDID-PTX/dataset/", file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_CANDIDptx = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_CANDIDptx = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)



        ## Dataloader for Localization -------------------------------------------------------------
        ## -- TBX11K Dataset -- ##
        dataset_train_loc_TBX11k = build_dataset(image_set='tbx11k_catagnostic_train', args=args) ## TBX11K Dataset - Training A set # tbx11k_catagnostic_train | tbx11k_catagnostic_train_A | tbx11k_catagnostic_train_B
        dataset_val_loc_TBX11k = build_dataset(image_set='tbx11k_catagnostic_test', args=args)
        if args.distributed:
            sampler_train_TBX11k = DistributedSampler(dataset_train_loc_TBX11k, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_TBX11k, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train_TBX11k = torch.utils.data.RandomSampler(dataset_train_loc_TBX11k)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_TBX11k)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_TBX11k, args.batch_size, drop_last=True)
        train_loader_loc_TBX11k = DataLoader(dataset_train_loc_TBX11k, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_TBX11k = DataLoader(dataset_val_loc_TBX11k, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        
        ## -- NODE21 Dataset -- ##
        dataset_train_loc_Node21 = build_dataset(image_set='node21_noduleDataset_train', args=args) ## NODE21 Dataset - Training A set # node21_noduleDataset_train | node21_noduleDataset_train_A | node21_noduleDataset_train_B
        dataset_val_loc_Node21 = build_dataset(image_set='node21_noduleDataset_test', args=args)
        if args.distributed:
            sampler_train_Node21 = DistributedSampler(dataset_train_loc_Node21, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_Node21, shuffle=False)
        else:
            sampler_train_Node21 = torch.utils.data.RandomSampler(dataset_train_loc_Node21)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_Node21)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_Node21, args.batch_size, drop_last=True)
        train_loader_loc_Node21 = DataLoader(dataset_train_loc_Node21, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        test_loader_loc_Node21 = DataLoader(dataset_val_loc_Node21, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

        ## -- CANDID-PTX Dataset -- ##
        dataset_train_loc_CANDIDptx = build_dataset(image_set='candidptx_pneumothorax_train_full', args=args) ## CANDID-PTX Dataset - Training A set # candidptx_pneumothorax_train_full | candidptx_pneumothorax_train_A | candidptx_pneumothorax_train_B
        dataset_val_loc_CANDIDptx = build_dataset(image_set='candidptx_pneumothorax_val', args=args) # candidptx_pneumothorax_val | candidptx_pneumothorax_test
        if args.distributed:
            sampler_train_CANDIDptx = DistributedSampler(dataset_train_loc_CANDIDptx, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_CANDIDptx, shuffle=False)
        else:
            sampler_train_CANDIDptx = torch.utils.data.RandomSampler(dataset_train_loc_CANDIDptx)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_CANDIDptx)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_CANDIDptx, args.batch_size, drop_last=True)
        train_loader_loc_CANDIDptx = DataLoader(dataset_train_loc_CANDIDptx, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_CANDIDptx = DataLoader(dataset_val_loc_CANDIDptx, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        ## -- ChestX-Det Dataset -- ##
        dataset_train_loc_ChestXDet = build_dataset(image_set='chestxdet_train', args=args) ## ChestX-Det Dataset - Training A set # chestxdet_train | chestxdet_train_A | chestxdet_train_B
        dataset_val_loc_ChestXDet = build_dataset(image_set='chestxdet_test', args=args)
        if args.distributed:
            sampler_train_ChestXDet = DistributedSampler(dataset_train_loc_ChestXDet, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_ChestXDet, shuffle=False)
        else:
            sampler_train_ChestXDet = torch.utils.data.RandomSampler(dataset_train_loc_ChestXDet)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_ChestXDet)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_ChestXDet, args.batch_size, drop_last=True)
        train_loader_loc_ChestXDet = DataLoader(dataset_train_loc_ChestXDet, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_ChestXDet = DataLoader(dataset_val_loc_ChestXDet, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        ## -- RSNA Pneumonia Challenge Dataset -- ##
        dataset_train_loc_RSNApneumonia = build_dataset(image_set='rsnaPneumoniaDetection_Train', args=args) ## RSNApneumonia Dataset - Training A set # rsnaPneumoniaDetection_Train | rsnaPneumoniaDetection_Train_A | rsnaPneumoniaDetection_Train_B
        dataset_val_loc_RSNApneumonia = build_dataset(image_set='rsnaPneumoniaDetection_Valid', args=args)
        if args.distributed:
            sampler_train_RSNApneumonia = DistributedSampler(dataset_train_loc_RSNApneumonia, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_RSNApneumonia, shuffle=False)
        else:
            sampler_train_RSNApneumonia = torch.utils.data.RandomSampler(dataset_train_loc_RSNApneumonia)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_RSNApneumonia)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_RSNApneumonia, args.batch_size, drop_last=True)
        train_loader_loc_RSNApneumonia = DataLoader(dataset_train_loc_RSNApneumonia, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_RSNApneumonia = DataLoader(dataset_val_loc_RSNApneumonia, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN

        ## -- SIIM-ACR Dataset -- ##
        dataset_train_loc_SiimACR = build_dataset(image_set='siimacr_train', args=args) ## RSNApneumonia Dataset - Training A set # siimacr_train | siimacr_train_A | siimacr_train_B
        dataset_val_loc_SiimACR = build_dataset(image_set='siimacr_val', args=args)
        if args.distributed:
            sampler_train_SiimACR = DistributedSampler(dataset_train_loc_SiimACR, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_SiimACR, shuffle=False)
        else:
            sampler_train_SiimACR = torch.utils.data.RandomSampler(dataset_train_loc_SiimACR)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_SiimACR)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_SiimACR, args.batch_size, drop_last=True)
        train_loader_loc_SiimACR = DataLoader(dataset_train_loc_SiimACR, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_SiimACR = DataLoader(dataset_val_loc_SiimACR, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN


        ## Segmentation -------------------------------------------------------------
        ## -- ChestX-Det Dataset -- ##
        if args.serverC == 'DFS':
            train_image_path_file = [("/mnt/dfs/nuislam/Data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_train_data.txt")]
            # val_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "/scratch/jliang12/data/ChestX-Det/data_files/ChestX-Det_valid_data.txt")]
            test_image_path_file = [("/mnt/dfs/nuislam/Data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_test_data.txt")]
        elif args.serverC == "SOL":
            train_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_train_data.txt")]
            # val_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "/scratch/jliang12/data/ChestX-Det/data_files/ChestX-Det_valid_data.txt")]
            test_image_path_file = [("/scratch/jliang12/data/ChestX-Det", "data/chestxdetdataset/ChestX-Det_test_data.txt")]
        # train_dataset = ChestXDetDataset(train_image_path_file, image_size=(args.imgsize,args.imgsize))
        # train_loader_seg_ChestXDet = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=True )
        # test_dataset = ChestXDetDataset(test_image_path_file,image_size=(args.imgsize,args.imgsize), mode="test")
        # test_loader_seg_ChestXDet = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False )

        train_dataset = Candid_PTX_PXSDataset([("/scratch/jliang12/data/CANDID-PTX/dataset","data/candid_ptx/train.txt")], image_size=(args.imgsize,args.imgsize), mode="train")
        sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader_seg_CANDIDptx = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler)
        # valid_dataset = Candid_PTX_PXSDataset([("/scratch/jliang12/data/CANDID-PTX/dataset","data/candid_ptx/valid.txt")], image_size=(args.imgsize,args.imgsize), mode="valid")
        # sampler = torch.utils.data.RandomSampler(valid_dataset)
        # val_loader_seg_CANDIDptx = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=True, sampler=sampler)
        test_dataset = Candid_PTX_PXSDataset([("/scratch/jliang12/data/CANDID-PTX/dataset","data/candid_ptx/test.txt")], image_size=(args.imgsize,args.imgsize), mode="test")
        sampler = torch.utils.data.RandomSampler(test_dataset)
        test_loader_seg_CANDIDptx = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)


        train_dataset = chestxdet_dataset(image_path="/scratch/jliang12/data/ChestX-Det/train/", masks_path="/scratch/jliang12/data/ChestX-Det/train_binary_mask/", image_size=(args.imgsize,args.imgsize), mode='train') ## ChestXDetDataset chestxdet_dataset
        train_loader_seg_ChestXDet = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=True )

        test_dataset = chestxdet_dataset(image_path="/scratch/jliang12/data/ChestX-Det/test/",masks_path="/scratch/jliang12/data/ChestX-Det/test_binary_mask/", image_size=(args.imgsize,args.imgsize), mode='test')
        test_loader_seg_ChestXDet = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False )


        train_dataset = SIIM_PXSDataset([("/scratch/jliang12/data/siim_pneumothorax_segmentation/train_jpeg","data/pxs/train.txt")], image_size=(args.imgsize,args.imgsize), mode="train")
        sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader_seg_SIIM = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler, drop_last=True)
        # valid_dataset = SIIM_PXSDataset([("/scratch/jliang12/data/siim_pneumothorax_segmentation/val_jpeg","data/pxs/val.txt")], image_size=(args.imgsize,args.imgsize), mode="valid")
        # sampler = torch.utils.data.RandomSampler(valid_dataset)
        # val_loader_seg_SIIM = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=True, sampler=sampler)
        test_dataset = SIIM_PXSDataset([("/scratch/jliang12/data/siim_pneumothorax_segmentation/test_jpeg","data/pxs/test.txt")], image_size=(args.imgsize,args.imgsize), mode="test")
        sampler = torch.utils.data.RandomSampler(test_dataset)
        test_loader_seg_SIIM = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)





        return train_loader_cls_CheXpert, test_loader_cls_CheXpert, train_loader_cls_NIHChestXray14, test_loader_cls_NIHChestXray14, train_loader_cls_VinDRCXR, test_loader_cls_VinDRCXR, train_loader_cls_NIHShenzhen, test_loader_cls_NIHShenzhen, train_loader_cls_MIMICII, test_loader_cls_MIMICII, \
            train_loader_cls_TBX11k, test_loader_cls_TBX11k, train_loader_cls_NODE21, test_loader_cls_NODE21, train_loader_cls_ChestXDet, test_loader_cls_ChestXDet, \
            train_loader_cls_RSNApneumonia, test_loader_cls_RSNApneumonia, train_loader_cls_SIIMACRptx, test_loader_cls_SIIMACRptx, train_loader_cls_CANDIDptx, test_loader_cls_CANDIDptx, \
            train_loader_loc_TBX11k, test_loader_loc_TBX11k, dataset_val_loc_TBX11k, sampler_train_TBX11k, \
            train_loader_loc_Node21, test_loader_loc_Node21, dataset_val_loc_Node21, sampler_train_Node21, \
            train_loader_loc_CANDIDptx, test_loader_loc_CANDIDptx, dataset_val_loc_CANDIDptx, sampler_train_CANDIDptx, \
            train_loader_loc_ChestXDet, test_loader_loc_ChestXDet, dataset_val_loc_ChestXDet, sampler_train_ChestXDet, \
            train_loader_loc_RSNApneumonia, test_loader_loc_RSNApneumonia, dataset_val_loc_RSNApneumonia, sampler_train_RSNApneumonia, \
            train_loader_loc_SiimACR, test_loader_loc_SiimACR, dataset_val_loc_SiimACR, sampler_train_SiimACR, \
            train_loader_seg_ChestXDet, test_loader_seg_ChestXDet, train_loader_seg_SIIM, test_loader_seg_SIIM, train_loader_seg_CANDIDptx, test_loader_seg_CANDIDptx



############## ===========  ClsLoc DataLoader ============ ################################## -- Anni's code for Dataloader
    if args.taskcomponent in ['ClsLoc_tbx11k_catagnostic']:
        ## Dataloader for Classification
        train_list = 'lists/TBX11K_train.txt'
        test_list = 'lists/TBX11K_val.txt'

        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/jpang12/datasets/tbx11k/TBX11K/"
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/tbx11k/tbx11k/TBX11K/"

        dataset_train = TBX11KDataset(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = TBX11KDataset(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))

        train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


        ## Dataloader for Localization
        dataset_train = build_dataset(image_set='tbx11k_catagnostic_train', args=args)
        dataset_val = build_dataset(image_set='tbx11k_catagnostic_test', args=args)
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        
        return data_loader_train, data_loader_val, sampler_train, dataset_val, train_loader, test_loader
    
    
    if args.taskcomponent in ['ClsLoc_rsna_pneumonia']:
        ## Dataloader for Classification
        train_list = 'data/rsna_pneumonia/train.txt'
        test_list = 'data/rsna_pneumonia/val.txt'

        if args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/rsna-pneumonia-detection-challenge/"

        dataset_train = RSNAPneumonia(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = RSNAPneumonia(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))

        train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


        ## Dataloader for Localization
        dataset_train = build_dataset(image_set='rsnaPneumoniaDetection_Train', args=args)
        dataset_val = build_dataset(image_set='rsnaPneumoniaDetection_Valid', args=args)
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        
        return data_loader_train, data_loader_val, sampler_train, dataset_val, train_loader, test_loader
    
    if args.taskcomponent in ['ClsLoc_node21_nodule']:
        ## Dataloader for Classification
        train_list = 'data/node21_dataset/train.txt'
        test_list = 'data/node21_dataset/val.txt'

        if args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/NODE21/cxr_images/proccessed_data/"

        dataset_train = NODE21(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = NODE21(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))

        train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


        ## Dataloader for Localization
        dataset_train = build_dataset(image_set='node21_noduleDataset_train', args=args)
        dataset_val = build_dataset(image_set='node21_noduleDataset_test', args=args)
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        
        return data_loader_train, data_loader_val, sampler_train, dataset_val, train_loader, test_loader
    
    if args.taskcomponent in ['ClsLoc_candid_ptx']:
        ## Dataloader for Classification
        train_list = 'data/candid_ptx/train.txt'
        test_list = 'data/candid_ptx/val.txt'

        if args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/CANDID-PTX/dataset/"

        dataset_train = CANDIDPTX(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = CANDIDPTX(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))

        train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


        ## Dataloader for Localization
        dataset_train = build_dataset(image_set='candidptx_pneumothorax_train_A', args=args)
        dataset_val = build_dataset(image_set='candidptx_pneumothorax_val', args=args)
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        
        return data_loader_train, data_loader_val, sampler_train, dataset_val, train_loader, test_loader

    if args.taskcomponent in ['ClsLoc_SIIM_ACR']:
        ## Dataloader for Classification
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_train.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_val.txt'

        if args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/siim_pneumothorax_segmentation/"

        dataset_train = SIIMPTX(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = SIIMPTX(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))

        train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


        ## Dataloader for Localization
        dataset_train = build_dataset(image_set='siimacr_train', args=args)
        dataset_val = build_dataset(image_set='siimacr_val', args=args)
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        
        return data_loader_train, data_loader_val, sampler_train, dataset_val, train_loader, test_loader
    
    if args.taskcomponent in ['ClsLoc_chestxdet_dataset']:
        ## Dataloader for Classification
        train_list = 'data/chestxdetdataset/ChestX-Det_train_data.txt'
        test_list = 'data/chestxdetdataset/ChestX-Det_test_data.txt'

        if args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/ChestX-Det/"

        dataset_train = ChestXDet_cls(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = ChestXDet_cls(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))

        train_loader = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)


        ## Dataloader for Localization
        dataset_train = build_dataset(image_set='chestxdet_train', args=args)
        dataset_val = build_dataset(image_set='chestxdet_test', args=args)
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=True)

        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                       collate_fn=utils.collate_fn, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, 1, sampler=sampler_val,
                                     drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        
        return data_loader_train, data_loader_val, sampler_train, dataset_val, train_loader, test_loader
    






############## ===========  Cls - Loc - Seg DataLoader ============ ################################## -- NAD's code for Dataloader
    ## SEGMENTATION DATASETS
    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'segFT_TESTseg_candidptxSEG':  
        train_dataset = Candid_PTX_PXSDataset([("/scratch/jliang12/data/CANDID-PTX/dataset","data/candid_ptx/train.txt")], image_size=(args.imgsize,args.imgsize), mode="train")
        sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader_seg_CANDIDptx = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler)
        # valid_dataset = Candid_PTX_PXSDataset([("/scratch/jliang12/data/CANDID-PTX/dataset","data/candid_ptx/valid.txt")], image_size=(args.imgsize,args.imgsize), mode="valid")
        # sampler = torch.utils.data.RandomSampler(valid_dataset)
        # val_loader_seg_CANDIDptx = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=True, sampler=sampler)
        test_dataset = Candid_PTX_PXSDataset([("/scratch/jliang12/data/CANDID-PTX/dataset","data/candid_ptx/test.txt")], image_size=(args.imgsize,args.imgsize), mode="test")
        sampler = torch.utils.data.RandomSampler(test_dataset)
        test_loader_seg_CANDIDptx = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        return train_loader_seg_CANDIDptx, test_loader_seg_CANDIDptx, 0

    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'segFT_TESTseg_chestxdetSEG':   ## ChestXDet_13Diseases [DongAo] ## chestxdet_dataset [Anni]
        train_dataset = chestxdet_dataset(image_path="/scratch/jliang12/data/ChestX-Det/train/", masks_path="/scratch/jliang12/data/ChestX-Det/train_binary_mask/", image_size=(args.imgsize,args.imgsize), mode='train') ## ChestXDetDataset chestxdet_dataset
        train_loader_seg_ChestXDet = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=True )

        test_dataset = chestxdet_dataset(image_path="/scratch/jliang12/data/ChestX-Det/test/",masks_path="/scratch/jliang12/data/ChestX-Det/test_binary_mask/", image_size=(args.imgsize,args.imgsize), mode='test')
        test_loader_seg_ChestXDet = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False )
        return train_loader_seg_ChestXDet, test_loader_seg_ChestXDet, 0

    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'segFT_TESTseg_siimacrSEG':  
        train_dataset = SIIM_PXSDataset([("/scratch/jliang12/data/siim_pneumothorax_segmentation/train_jpeg","data/pxs/train.txt")], image_size=(args.imgsize,args.imgsize), mode="train")
        sampler = torch.utils.data.RandomSampler(train_dataset)
        train_loader_seg_SIIM = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, sampler=sampler, drop_last=True)
        # valid_dataset = SIIM_PXSDataset([("/scratch/jliang12/data/siim_pneumothorax_segmentation/val_jpeg","data/pxs/val.txt")], image_size=(args.imgsize,args.imgsize), mode="valid")
        # sampler = torch.utils.data.RandomSampler(valid_dataset)
        # val_loader_seg_SIIM = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers,pin_memory=True, sampler=sampler)
        test_dataset = SIIM_PXSDataset([("/scratch/jliang12/data/siim_pneumothorax_segmentation/test_jpeg","data/pxs/test.txt")], image_size=(args.imgsize,args.imgsize), mode="test")
        sampler = torch.utils.data.RandomSampler(test_dataset)
        test_loader_seg_SIIM = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, drop_last=False)
        return train_loader_seg_SIIM, test_loader_seg_SIIM, 0
    


    ## Localization DATASETS
    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'locFT_TESTloc_objects365LOC':
        dataset_train_loc_objects365 = build_dataset(image_set='objects365_NAD_train', args=args) ## TBX11K Dataset - Training A set # tbx11k_catagnostic_train | tbx11k_catagnostic_train_A | tbx11k_catagnostic_train_B
        dataset_val_loc_objects365 = build_dataset(image_set='objects365_NAD_test', args=args)
        if args.distributed:
            sampler_train_objects365 = DistributedSampler(dataset_train_loc_objects365, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_objects365, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train_objects365 = torch.utils.data.RandomSampler(dataset_train_loc_objects365)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_objects365)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_objects365, args.batch_size, drop_last=True)
        train_loader_loc_objects365 = DataLoader(dataset_train_loc_objects365, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_objects365 = DataLoader(dataset_val_loc_objects365, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        return train_loader_loc_objects365, test_loader_loc_objects365, dataset_val_loc_objects365
    



    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'locFT_TESTloc_tbx11kLOC':
        dataset_train_loc_TBX11k = build_dataset(image_set='tbx11k_catagnostic_train', args=args) ## TBX11K Dataset - Training A set # tbx11k_catagnostic_train | tbx11k_catagnostic_train_A | tbx11k_catagnostic_train_B
        dataset_val_loc_TBX11k = build_dataset(image_set='tbx11k_catagnostic_test', args=args)
        if args.distributed:
            sampler_train_TBX11k = DistributedSampler(dataset_train_loc_TBX11k, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_TBX11k, shuffle=False)
            # sampler_test = DistributedSampler(dataset_test, shuffle=False) # added by Nahid
        else:
            sampler_train_TBX11k = torch.utils.data.RandomSampler(dataset_train_loc_TBX11k)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_TBX11k)
            # sampler_test = torch.utils.data.SequentialSampler(dataset_test) # added by Nahid
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_TBX11k, args.batch_size, drop_last=True)
        train_loader_loc_TBX11k = DataLoader(dataset_train_loc_TBX11k, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_TBX11k = DataLoader(dataset_val_loc_TBX11k, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        return train_loader_loc_TBX11k, test_loader_loc_TBX11k, dataset_val_loc_TBX11k
    
    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'locFT_TESTloc_node21LOC':
        dataset_train_loc_Node21 = build_dataset(image_set='node21_noduleDataset_train', args=args) ## NODE21 Dataset - Training A set # node21_noduleDataset_train | node21_noduleDataset_train_A | node21_noduleDataset_train_B
        dataset_val_loc_Node21 = build_dataset(image_set='node21_noduleDataset_test', args=args)
        if args.distributed:
            sampler_train_Node21 = DistributedSampler(dataset_train_loc_Node21, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_Node21, shuffle=False)
        else:
            sampler_train_Node21 = torch.utils.data.RandomSampler(dataset_train_loc_Node21)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_Node21)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_Node21, args.batch_size, drop_last=True)
        train_loader_loc_Node21 = DataLoader(dataset_train_loc_Node21, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        test_loader_loc_Node21 = DataLoader(dataset_val_loc_Node21, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
        return train_loader_loc_Node21, test_loader_loc_Node21, dataset_val_loc_Node21
    
    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'locFT_TESTloc_candidptxLOC':
        dataset_train_loc_CANDIDptx = build_dataset(image_set='candidptx_pneumothorax_train_full', args=args) ## CANDID-PTX Dataset - Training A set # candidptx_pneumothorax_train_full | candidptx_pneumothorax_train_A | candidptx_pneumothorax_train_B
        dataset_val_loc_CANDIDptx = build_dataset(image_set='candidptx_pneumothorax_val', args=args) # candidptx_pneumothorax_val | candidptx_pneumothorax_test
        if args.distributed:
            sampler_train_CANDIDptx = DistributedSampler(dataset_train_loc_CANDIDptx, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_CANDIDptx, shuffle=False)
        else:
            sampler_train_CANDIDptx = torch.utils.data.RandomSampler(dataset_train_loc_CANDIDptx)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_CANDIDptx)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_CANDIDptx, args.batch_size, drop_last=True)
        train_loader_loc_CANDIDptx = DataLoader(dataset_train_loc_CANDIDptx, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_CANDIDptx = DataLoader(dataset_val_loc_CANDIDptx, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        return train_loader_loc_CANDIDptx, test_loader_loc_CANDIDptx, dataset_val_loc_CANDIDptx

    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'locFT_TESTloc_rsnapneumoniaLOC':
        dataset_train_loc_RSNApneumonia = build_dataset(image_set='rsnaPneumoniaDetection_Train', args=args) ## RSNApneumonia Dataset - Training A set # rsnaPneumoniaDetection_Train | rsnaPneumoniaDetection_Train_A | rsnaPneumoniaDetection_Train_B
        dataset_val_loc_RSNApneumonia = build_dataset(image_set='rsnaPneumoniaDetection_Valid', args=args)
        if args.distributed:
            sampler_train_RSNApneumonia = DistributedSampler(dataset_train_loc_RSNApneumonia, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_RSNApneumonia, shuffle=False)
        else:
            sampler_train_RSNApneumonia = torch.utils.data.RandomSampler(dataset_train_loc_RSNApneumonia)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_RSNApneumonia)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_RSNApneumonia, args.batch_size, drop_last=True)
        train_loader_loc_RSNApneumonia = DataLoader(dataset_train_loc_RSNApneumonia, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_RSNApneumonia = DataLoader(dataset_val_loc_RSNApneumonia, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        return train_loader_loc_RSNApneumonia, test_loader_loc_RSNApneumonia, dataset_val_loc_RSNApneumonia
    
    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'locFT_TESTloc_chestxdetLOC':
        ## -- ChestX-Det Dataset -- ##
        dataset_train_loc_ChestXDet = build_dataset(image_set='chestxdet_train', args=args) ## ChestX-Det Dataset - Training A set # chestxdet_train | chestxdet_train_A | chestxdet_train_B
        dataset_val_loc_ChestXDet = build_dataset(image_set='chestxdet_test', args=args)
        if args.distributed:
            sampler_train_ChestXDet = DistributedSampler(dataset_train_loc_ChestXDet, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_ChestXDet, shuffle=False)
        else:
            sampler_train_ChestXDet = torch.utils.data.RandomSampler(dataset_train_loc_ChestXDet)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_ChestXDet)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_ChestXDet, args.batch_size, drop_last=True)
        train_loader_loc_ChestXDet = DataLoader(dataset_train_loc_ChestXDet, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_ChestXDet = DataLoader(dataset_val_loc_ChestXDet, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        return train_loader_loc_ChestXDet, test_loader_loc_ChestXDet, dataset_val_loc_ChestXDet
    
    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'locFT_TESTloc_siimacrLOC':
        dataset_train_loc_SiimACR = build_dataset(image_set='siimacr_train', args=args) ## RSNApneumonia Dataset - Training A set # siimacr_train | siimacr_train_A | siimacr_train_B
        dataset_val_loc_SiimACR = build_dataset(image_set='siimacr_val', args=args)
        if args.distributed:
            sampler_train_SiimACR = DistributedSampler(dataset_train_loc_SiimACR, shuffle=True)
            sampler_val = DistributedSampler(dataset_val_loc_SiimACR, shuffle=False)
        else:
            sampler_train_SiimACR = torch.utils.data.RandomSampler(dataset_train_loc_SiimACR)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val_loc_SiimACR)
        batch_sampler_train = torch.utils.data.BatchSampler(sampler_train_SiimACR, args.batch_size, drop_last=True)
        train_loader_loc_SiimACR = DataLoader(dataset_train_loc_SiimACR, batch_sampler=batch_sampler_train, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        test_loader_loc_SiimACR = DataLoader(dataset_val_loc_SiimACR, 1, sampler=sampler_val, drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers) # RETURN
        return train_loader_loc_SiimACR, test_loader_loc_SiimACR, dataset_val_loc_SiimACR
    
    ## CLASSIFICATION DATASETS
    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'clsFT_TESTcls_nihchestxray14CLS':
        train_list = 'data/xray14/official/train_official.txt'
        val_list = 'data/xray14/official/val_official.txt'
        test_list = 'data/xray14/official/test_official.txt'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/ChestXRay14_images/" #"/mnt/dfs/nuislam/Data/ChestXRay14_images/" ## "/data/jliang12/jpang12/dataset/nih_xray14/images/images/"
        elif args.serverC == "SOL":
            # data_dir = "/data/jliang12/jpang12/dataset/nih_xray14/images/images/" 
            data_dir = "/scratch/jliang12/data/nih_xray14/images/images/"
        dataset_train = ChestXray14Dataset(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        # dataset_val = ChestXray14Dataset(images_path=data_dir, file_path=val_list, augment=build_transform_classification(normalize="chestx-ray", mode="valid"))
        dataset_test = ChestXray14Dataset(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_NIHChestXray14 = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        # val_loader_cls_NIHChestXray14 = DataLoader(dataset=dataset_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        test_loader_cls_NIHChestXray14 = DataLoader(dataset=dataset_test, batch_size=args.batch_size//2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        return train_loader_cls_NIHChestXray14, test_loader_cls_NIHChestXray14, 0

    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'clsFT_TESTcls_chexpertCLS':
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/CheXpert-v1.0_train.csv'
        val_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/CheXpert-v1.0_valid.csv'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/CheXpert-v1.0_valid.csv'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/ChestXRay14_images/" #"/mnt/dfs/nuislam/Data/ChestXRay14_images/" ## "/data/jliang12/jpang12/dataset/nih_xray14/images/images/"
        elif args.serverC == "SOL":
            # data_dir = "/data/jliang12/mhossei2/Dataset/" ## CheXpert-v1.0
            data_dir = "/scratch/jliang12/data/"
        dataset_train = CheXpert(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = CheXpert(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_CheXpert = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_CheXpert = DataLoader(dataset=dataset_test, batch_size=args.batch_size//2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        return train_loader_cls_CheXpert, test_loader_cls_CheXpert, 0

    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'clsFT_TESTcls_vindrcxrCLS':
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/VinDrCXR_train_pe_global_one.txt'
        val_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/VinDrCXR_test_pe_global_one.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/VinDrCXR_test_pe_global_one.txt'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/ChestXRay14_images/" #"/mnt/dfs/nuislam/Data/ChestXRay14_images/" ## "/data/jliang12/jpang12/dataset/nih_xray14/images/images/"
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/"
        dataset_train = VinDrCXR(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = VinDrCXR(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_VinDRCXR = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_VinDRCXR = DataLoader(dataset=dataset_test, batch_size=args.batch_size//2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        return train_loader_cls_VinDRCXR, test_loader_cls_VinDRCXR, 0

    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'clsFT_TESTcls_nihshenzenCLS':
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/ShenzenCXR_train_data.txt'
        val_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/ShenzenCXR_valid_data.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/ShenzenCXR_test_data.txt'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/ChestXRay14_images/" #"/mnt/dfs/nuislam/Data/ChestXRay14_images/" ## "/data/jliang12/jpang12/dataset/nih_xray14/images/images/"
        elif args.serverC == "SOL":
            # data_dir = "/data/jliang12/mhossei2/Dataset/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png/"
            data_dir = "/scratch/jliang12/data/ShenzhenHospitalXray/ChinaSet_AllFiles/CXR_png/"
        dataset_train = ShenzhenCXR(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = ShenzhenCXR(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_NIHShenzhen = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_NIHShenzhen = DataLoader(dataset=dataset_test, batch_size=args.batch_size//2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        return train_loader_cls_NIHShenzhen, test_loader_cls_NIHShenzhen, 0

    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'clsFT_TESTcls_mimic2CLS':
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/mimic-cxr-2.0.0-train.csv'
        val_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/mimic-cxr-2.0.0-validate.csv'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/ark6_classificationTask/mimic-cxr-2.0.0-test.csv'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/ChestXRay14_images/" #"/mnt/dfs/nuislam/Data/ChestXRay14_images/" ## "/data/jliang12/jpang12/dataset/nih_xray14/images/images/"
        elif args.serverC == "SOL":
            # data_dir = "/data/jliang12/jpang12/dataset/MIMIC_jpeg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
            data_dir = "/scratch/jliang12/data/MIMIC_jpeg/physionet.org/files/mimic-cxr-jpg/2.0.0/"
        dataset_train = MIMIC(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = MIMIC(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_MIMICII = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_MIMICII = DataLoader(dataset=dataset_test, batch_size=args.batch_size//2, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        return train_loader_cls_MIMICII, test_loader_cls_MIMICII, 0

    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'clsFT_TESTcls_tbx11kCLS':
        ## -- TBX11K Dataset -- ##
        train_list = 'lists/TBX11K_train.txt'
        test_list = 'lists/TBX11K_val.txt'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/jpang12/datasets/tbx11k/TBX11K/"
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/tbx11k/tbx11k/TBX11K/"
        dataset_train = TBX11KDataset(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = TBX11KDataset(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_TBX11k = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_TBX11k = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        return train_loader_cls_TBX11k, test_loader_cls_TBX11k, 0

    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'clsFT_TESTcls_node21CLS':
        ## -- NODE21 Dataset -- ##
        train_list = 'data/node21_dataset/train.txt'
        test_list = 'data/node21_dataset/test.txt'
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/NODE21_ann/"
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/NODE21/cxr_images/proccessed_data/"
        dataset_train = NODE21(images_path=data_dir, file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"), annotation_percent=100)
        dataset_test = NODE21(images_path=data_dir, file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_NODE21 = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_NODE21 = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        return train_loader_cls_NODE21, test_loader_cls_NODE21, 0

    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'clsFT_TESTcls_chestxdetCLS':
        ## -- ChestX-Det Dataset -- ##
        if args.serverC == 'DFS':
            data_dir = "/mnt/dfs/nuislam/Data/ChestX-Det/"
            train_list = '/mnt/dfs/nuislam/Data/ChestX-Det/ChestX_det_train_NAD_v2.json'
            test_list = '/mnt/dfs/nuislam/Data/ChestX-Det/ChestX_det_test_NAD_v2.json'
        elif args.serverC == "SOL":
            data_dir = "/scratch/jliang12/data/ChestX-Det/"
            train_list = '/scratch/jliang12/data/ChestX-Det/ChestX_det_train_NAD_v2.json'
            test_list = '/scratch/jliang12/data/ChestX-Det/ChestX_det_test_NAD_v2.json'
        dataset_train = ChestXDet_cls(images_path=data_dir, file_path="train", augment=build_transform_classification(normalize="chestx-ray", mode="train"), anno_percent=100)
        dataset_test = ChestXDet_cls(images_path=data_dir, file_path="test", augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_ChestXDet = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_ChestXDet = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        return train_loader_cls_ChestXDet, test_loader_cls_ChestXDet, 0

    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'clsFT_TESTcls_rsnapneumoniaCLS':
        ## -- RSNAPneumonia Dataset -- ##
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/rsna_pneumonia/RSNAPneumonia_train.txt'
        valid_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/rsna_pneumonia/RSNAPneumonia_val.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/rsna_pneumonia/RSNAPneumonia_test.txt'
        dataset_train = RSNAPneumonia(images_path="/scratch/jliang12/data/rsna-pneumonia-detection-challenge/stage_2_train_images_png/", file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"))
        dataset_test = RSNAPneumonia(images_path="/scratch/jliang12/data/rsna-pneumonia-detection-challenge/stage_2_train_images_png/", file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_RSNApneumonia = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_RSNApneumonia = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        return train_loader_cls_RSNApneumonia, test_loader_cls_RSNApneumonia, 0

    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'clsFT_TESTcls_siimacrCLS':
        ## -- SIIMPTX Dataset -- ##
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_train.txt'
        valid_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_val.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/siimacr_ptx/SIIMPTX_cls_test.txt'
        dataset_train = SIIMPTX(images_path="/scratch/jliang12/data/siim_pneumothorax_segmentation/", file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"))
        dataset_test = SIIMPTX(images_path="/scratch/jliang12/data/siim_pneumothorax_segmentation/", file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_SIIMACRptx = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_SIIMACRptx = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        return train_loader_cls_SIIMACRptx, test_loader_cls_SIIMACRptx, 0

    if args.taskcomponent in ['foundation_x3_FineTuning'] and args.cyclictask == 'clsFT_TESTcls_candidptxCLS':
        ## -- CANDID-PTX Dataset -- ##
        train_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/candid_ptx/CANDIDPTX_cls_train.txt'
        valid_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/candid_ptx/CANDIDPTX_cls_val.txt'
        test_list = '/scratch/nuislam/Model_Checkpoints/data_files_splits/candid_ptx/CANDIDPTX_cls_test.txt'
        dataset_train = CANDIDPTX(images_path="/scratch/jliang12/data/CANDID-PTX/dataset/", file_path=train_list, augment=build_transform_classification(normalize="chestx-ray", mode="train"))
        dataset_test = CANDIDPTX(images_path="/scratch/jliang12/data/CANDID-PTX/dataset/", file_path=test_list, augment=build_transform_classification(normalize="chestx-ray", mode="test2"))
        train_loader_cls_CANDIDptx = DataLoader(dataset=dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        test_loader_cls_CANDIDptx = DataLoader(dataset=dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        return train_loader_cls_CANDIDptx, test_loader_cls_CANDIDptx, 0
