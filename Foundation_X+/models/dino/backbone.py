# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from collections import OrderedDict
import os

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List


from util.misc import NestedTensor, clean_state_dict, is_main_process

from .position_encoding import build_position_encoding
from .convnext import build_convnext

from .swin_transformer_CyclicSegmentation import build_swin_transformer

import copy



class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                      for ft_size in feature_channels[1:]])
        self.smooth_conv = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)]
                                         * (len(feature_channels) - 1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]  ##
        P = [up_and_add(features[i], features[i - 1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])  # P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x

def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y





class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_indices: list):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        return_layers = {}
        for idx, layer_index in enumerate(return_interm_indices):
            return_layers.update({"layer{}".format(5 - len(return_interm_indices) + idx): "{}".format(layer_index)})

        # if len:
        #     if use_stage1_feature:
        #         return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        #     else:
        #         return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
        # else:
        #     return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)

        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 dilation: bool,
                 return_interm_indices:list,
                 batch_norm=FrozenBatchNorm2d,
                 ):
        if name in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
            backbone = getattr(torchvision.models, name)(
                replace_stride_with_dilation=[False, False, dilation],
                pretrained=is_main_process(), norm_layer=batch_norm)
        else:
            raise NotImplementedError("Why you can get here with name {}".format(name))
        # num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        assert name not in ('resnet18', 'resnet34'), "Only resnet50 and resnet101 are available."
        assert return_interm_indices in [[0,1,2,3], [1,2,3], [3]]
        num_channels_all = [256, 512, 1024, 2048] # was active
        # num_channels_all = [128, 256, 512, 1024, 2048] # for InternImage
        num_channels = num_channels_all[4-len(return_interm_indices):]
        super().__init__(backbone, train_backbone, num_channels, return_interm_indices)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, input_size, output_size, feature_channels, segmentationClass):
        super().__init__(backbone, position_embedding)

        ## Issue faced adding Classification & Segmentation -- Unused parameters were not contributing to loss -- needed to make find_unused_parameters=True for DDP initialization
        # Classification
        # self.classification_input_size = input_size # input from the backbone # For Swin-L
        # self.classification_output_size = output_size # number of class # For ImageNet classification task
        # self.classification_norm = nn.LayerNorm(self.classification_input_size) # newly added
        # self.classification_avgpool = nn.AdaptiveAvgPool1d(1)
        # # self.classification_head = nn.Linear(self.classification_input_size, self.classification_output_size, bias=True)
        # self.classification_head = nn.Linear(self.classification_input_size, self.classification_output_size)

        # # Segmentation -- Jiaxuan's Version
        # segmentation_num_classes = segmentationClass
        # # segmentation_feature_channels = [192, 384, 768, 768] # for Swin-T
        # # segmentation_feature_channels = [256, 512, 1024, 1024] # for Swin-B
        # # segmentation_feature_channels = [192, 384, 768, 1536] # for Swin-L --- Is this correct??
        # # segmentation_feature_channels = [384, 768, 1536, 1536] # for Swin-L --- Is this correct??
        # segmentation_feature_channels = feature_channels
        # self.segmentation_PPN = PSPModule(segmentation_feature_channels[-1])
        # self.segmentation_FPN = FPN_fuse(segmentation_feature_channels, fpn_out=segmentation_feature_channels[0])
        # self.segmentation_head = nn.Conv2d(segmentation_feature_channels[0], segmentation_num_classes, kernel_size=3, padding=1)


        # from mmseg.models.decode_heads import UPerHead
        # upernet_seg_head = UPerHead(in_channels=[192, 384, 768, 1536],
        #                     in_index=[0, 1, 2, 3],
        #                     pool_scales=(1, 2, 3, 6),
        #                     channels=512,
        #                     dropout_ratio=0.1,
        #                     num_classes=2,
        #                     out_channels=1,
        #                     threshold = 0.5,
        #                     align_corners=False,
        #                     loss_decode=dict(
        #                         type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))


    def forward(self, tensor_list: NestedTensor):
        
        xs = self[0](tensor_list) # Backbone
        out: List[NestedTensor] = []
        pos = []

        try: # Original
            for name, x in xs.items():
                # print("[Inside Joiner - Regular] Features", x.shape)
                out.append(x)
                # position encoding
                pos.append(self[1](x).to(x.tensors.dtype)) # DINO decoder
            return out, pos

        except: # to only extract the ouput features from the Backbone.

            # ## Classification
            # ### out_x = torch.tensor(out[-1])
            # # out_x = out[-1].clone().detach().requires_grad_(True)
            # # out_x = out_x.view(out_x.shape[0], -1, self.classification_input_size)
            # # out_x = self.classification_norm(out_x) # newly added
            # # out_x = self.classification_avgpool(out_x.transpose(1,2))
            # # out_x = self.classification_head(torch.flatten(out_x, 1))
            # out_x = self[0].extra_features(tensor_list)



            # ## Segmentation
            # out = []
            # for x in xs:
            #     out.append(x)
            # input_size = (tensor_list.shape[2], tensor_list.shape[3]) # B C H W
            # # input_size = (x.size()[2], x.size()[3])
            # out_segX = out
            # # out_segX = out.clone().detach().requires_grad_(True)

            # out_segX[-1] = self.segmentation_PPN(out_segX[-1])
            # # print("[Inside Segmentation] out_segX[-1]", out_segX[-1].shape)
            # out_segX = self.segmentation_head(self.segmentation_FPN(out_segX))
            # out_segX = F.interpolate(out_segX, size=input_size, mode='bilinear') # Needs to be Original Image Size
            # # out_segX = F.interpolate(out_segX, size=input_size, mode='bilinear', align_corners=True) # Needs to be Original Image Size

            # return out, out_x, out_segX
            print("ERROR !!!")
            exit(0)
            return 0


def build_backbone(args):
    """
    Useful args:
        - backbone: backbone name
        - lr_backbone: 
        - dilation
        - return_interm_indices: available: [0,1,2,3], [1,2,3], [3]
        - backbone_freeze_keywords: 
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    if not train_backbone:
        raise ValueError("Please set lr_backbone > 0")
    return_interm_indices = args.return_interm_indices
    assert return_interm_indices in [[0,1,2,3], [1,2,3], [3]]
    backbone_freeze_keywords = args.backbone_freeze_keywords
    use_checkpoint = getattr(args, 'use_checkpoint', False)

    if args.backbone in ['resnet50', 'resnet101']:
        backbone = Backbone(args.backbone, train_backbone, args.dilation,   
                                return_interm_indices,   
                                batch_norm=FrozenBatchNorm2d)
        bb_num_channels = backbone.num_channels
    elif args.backbone in ['swin_T_224_1k', 'swin_B_224_22k', 'swin_B_384_22k', 'swin_L_224_22k', 'swin_L_384_22k']:
        pretrain_img_size = int(args.backbone.split('_')[-2])
        backbone = build_swin_transformer(args.backbone, \
                    pretrain_img_size=pretrain_img_size, num_classes=args.numClasses, \
                    out_indices=tuple(return_interm_indices), \
                dilation=args.dilation, use_checkpoint=use_checkpoint)

        # freeze some layers
        if backbone_freeze_keywords is not None:
            for name, parameter in backbone.named_parameters():
                for keyword in backbone_freeze_keywords:
                    if keyword in name:
                        parameter.requires_grad_(False)
                        break
        # if "backbone_dir" in args:
        if args.taskcomponent == 'detection' and args.backbone_dir is not None:
            pretrained_dir = args.backbone_dir
            PTDICT = {
                'swin_T_224_1k': 'swin_tiny_patch4_window7_224.pth',
                'swin_B_224_22k': 'swin_base_patch4_window7_224_22k.pth', # swin_base_patch4_window7_224.pth # swin_base_patch4_window7_224_22k.pth
                'swin_B_384_22k': 'swin_base_patch4_window12_384.pth',
                'swin_L_384_22k': 'swin_large_patch4_window12_384_22k.pth',
            }
            # pretrainedpath = os.path.join(pretrained_dir, PTDICT[args.backbone]) # was active

            if args.init == 'imagenet':
                pretrainedpath = args.backbone_dir
                checkpoint = torch.load(pretrainedpath, map_location='cpu')['model']
                from collections import OrderedDict
                def key_select_function(keyname):
                    if 'head' in keyname:
                        return False
                    if args.dilation and 'layers.3' in keyname:
                        return False
                    return True
                _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if key_select_function(k)})
                _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)

                # _tmp_st_output = backbone.load_state_dict(checkpoint, strict=False)
                print(str(_tmp_st_output))
                print("[Model Info.] Backbone Pre-trained Weights Loaded:", args.backbone_dir)
            elif args.init == 'ark':
                checkpoint = torch.load(args.backbone_dir, map_location='cpu')
                state_dict = checkpoint['teacher']
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = key.replace('module.', '')  # Remove the module. prefix
                    new_state_dict[new_key] = value
                state_dict = new_state_dict

                new_state_dict = {}
                prefix = "module.backbone.0."
                for key, value in state_dict.items():
                    if "head.weight" in key or "head.bias" in key:
                        continue
                    new_key = prefix + key
                    new_state_dict[new_key] = value
                status_w = backbone.load_state_dict(new_state_dict, strict=False)
            
                renaming_dict = {
                    'norm.weight': 'module.backbone.classification_norm.weight',
                    'norm.bias': 'module.backbone.classification_norm.bias',
                    # 'head.weight': 'backbone.classification_head.weight',
                    # 'head.bias': 'backbone.classification_head.bias'
                }
                new_state_dict = {}
                for old_key, new_key in renaming_dict.items():
                    new_state_dict[new_key] = state_dict[old_key]

                status_w = backbone.load_state_dict(new_state_dict, strict=False)
                print("[Model Info.] Backbone Pre-trained Weights Loaded:", args.backbone_dir)
        
        bb_num_channels = backbone.num_features[4 - len(return_interm_indices):]


    elif args.backbone in ['convnext_xlarge_22k']:
        backbone = build_convnext(modelname=args.backbone, pretrained=True, out_indices=tuple(return_interm_indices),backbone_dir=args.backbone_dir)
        bb_num_channels = backbone.dims[4 - len(return_interm_indices):]

    elif args.backbone in ['internimage_T_224_1k']: # Added by Nahid -- InternImage-T should work!
        pretrain_img_size = int(args.backbone.split('_')[-2])
        backbone = build_internimage_transformer(args.backbone, pretrain_img_size=pretrain_img_size, out_indices=tuple(return_interm_indices), dilation=args.dilation, use_checkpoint=use_checkpoint)

        # Loading pre-trained weights only for the InternImage component.
        if "backbone_dir" in args:
            pretrained_dir = args.backbone_dir
            # PTDICT = {
            #     'internimage_T_224_1k': 'internimage_t_1k_224.pth', 
            #     'internimage_B_384_22k': '',
            #     'internimage_L_384_22k': '',
            # }
            # pretrainedpath = os.path.join(pretrained_dir, PTDICT[args.backbone])


            # checkpoint = torch.load(pretrained_dir, map_location='cpu')['model']
            # from collections import OrderedDict
            # def key_select_function(keyname):
            #     if 'head' in keyname:
            #         return False
            #     if args.dilation and 'layers.3' in keyname:
            #         return False
            #     return True
            # _tmp_st = OrderedDict({k:v for k, v in clean_state_dict(checkpoint).items() if key_select_function(k)})
            # _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)

            # df = open("model_InternImageT_w.txt",'w')
            # checkpoint_w = backbone.state_dict()
            # for name, _ in checkpoint_w.items():
            #     df.write(name + "\n")
            # df.close()

            checkpoint = torch.load(pretrained_dir, map_location='cpu')['model']
            # df = open("checkpoint_InternImageT_w.txt",'w')
            # for name, _ in checkpoint.items():
            #     df.write(name + "\n")
            # df.close()

            model_dict = backbone.state_dict()
            new_dict = {}
            for k, v in checkpoint.items(): # was state_dict for RSNASlice
                temp_k = k.split("backbone.0.")[-1]
                new_dict[temp_k] = v
            ignore_keys = ['conv_head.0.weight', 'conv_head.1.0.weight', 'conv_head.1.0.bias', 'conv_head.1.0.running_mean', 'conv_head.1.0.running_var', 'conv_head.1.0.num_batches_tracked', 'head.weight', 'head.bias']
            new_dict = {k: v for k, v in new_dict.items() if k in model_dict and k not in ignore_keys}
            _tmp_st_output = backbone.load_state_dict(new_dict, strict=False)
            print()
            print("[Model Info]", str(_tmp_st_output))
            print("----")
            # exit()


        bb_num_channels = backbone.num_features[4 - len(return_interm_indices):]
        # print("[CHECK - inside Backbone.py] len(return_interm_indices)", len(return_interm_indices))
        # print("[CHECK - inside Backbone.py] bb_num_channels", bb_num_channels)
        # exit()
    else:
        raise NotImplementedError("Unknown backbone {}".format(args.backbone))
    

    assert len(bb_num_channels) == len(return_interm_indices), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"


    # model = Joiner(backbone, position_embedding)

    ## For Segmentation Purpose
    if args.backbonemodel == "Swin-L":
        BACKBONE_LAST_FEATURE_SIZE = 1536
        BACKBONE_FEATURE_CHANNELS = [192, 384, 768, 1536]
    elif args.backbonemodel == "Swin-B":
        BACKBONE_LAST_FEATURE_SIZE = 1024
        BACKBONE_FEATURE_CHANNELS = [128, 256, 512, 1024]
    elif args.backbonemodel == "Swin-T":
        BACKBONE_LAST_FEATURE_SIZE = 768
        BACKBONE_FEATURE_CHANNELS = [192, 384, 768, 768] # Need to check 
    else:
        print("  [ERROR] Please verify the backbone...  ")
        exit(0)
    
    CLASSIFICATION_OUTPUT_CLASS = args.numClasses # For ImageNet1k 1000
    SEGMENTATION_CLASS = 1

    model = Joiner(backbone, position_embedding, BACKBONE_LAST_FEATURE_SIZE, CLASSIFICATION_OUTPUT_CLASS, BACKBONE_FEATURE_CHANNELS, SEGMENTATION_CLASS)

    model.num_channels = bb_num_channels 
    assert isinstance(bb_num_channels, List), "bb_num_channels is expected to be a List but {}".format(type(bb_num_channels))

    return model
