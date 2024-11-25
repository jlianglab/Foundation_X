import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        
import torch, json
import numpy as np

from main import build_model_main
from util.slconfig import SLConfig
from datasets import build_dataset
from util.visualizer import COCOVisualizer
from util import box_ops
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    x_end_intersection = min(x1 + w1, x2 + w2)
    y_end_intersection = min(y1 + h1, y2 + h2)
    intersection_area = max(0, x_end_intersection - x_intersection) * max(0, y_end_intersection - y_intersection)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - intersection_area
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def calculate_miou(target_bboxes, predicted_bboxes):
    total_iou = 0.0
    num_matches = 0
    for target_bbox in target_bboxes:
        best_iou = 0.0
        for predicted_bbox in predicted_bboxes:
            iou = calculate_iou(target_bbox, predicted_bbox)
#             print("Individual IoU:", iou)
            best_iou = max(best_iou, iou)
        print("Best IoU:", best_iou)
        total_iou += best_iou
        num_matches += 1
    mIoU = total_iou / num_matches
    return mIoU


model_config_path = "config/DINO/DINO_4scale_swinBASE.py"
model_checkpoint_path = "/mnt/dfs/nuislam/Projects/IntegratedModel_GitHubV/Model_Checkpoints/IntegratedModel_DINOpipeline/Swin-B_224_Ark6_UperNet_FineTune_Node21_Nodule/run103_DFS_LocCls/ckpt_E136_TH2.pth"
# model_checkpoint_path = "/mnt/dfs/nuislam/Projects/IntegratedModel_GitHubV/Model_Checkpoints/IntegratedModel_DINOpipeline/Swin-B_224_Ark6_UperNet_FineTune_Node21_Nodule/fromSOL/ckpt_E136_TH2.pth"

args = SLConfig.fromfile(model_config_path) 
args.device = 'cuda' 
args.backbonemodel = "Swin-B"
args.taskcomponent = "detection"
# args.dataset_file = "coco"
# args.dataset_file = "chestxdetdataset"
# args.dataset_file = "vindrcxr_detect"
args.dataset_file = "vindrcxr_OrganDetect"
# args.init = None
args.init = "ark"
# args.backbone_dir = "/mnt/dfs/nuislam/Projects/DINO_Detection_old/checkpoints/checkpoint0029_4scale_swin.pth"
args.backbone_dir = "/mnt/dfs/nuislam/pretrainedModels/Ark6/TSconsist_NoOD_MIMIC_CheXpert_ChestXray14_RSNAPneumonia_VinDrCXR_Shenzhen_ep200.pth.tar"
# args.backbone_dir = None
args.num_classes = 2 # was 91 COCO
args.numClasses = 1
args.dn_labelbook_size = 3

args.lr_backbone = 0.001
model, criterion, postprocessors = build_model_main(args)

checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
# checkpoint = torch.load(args.backbone_dir , map_location='cpu')
# for k,v in checkpoint['model'].items():
#     print(k, v.shape)

# State_CHECK=model.load_state_dict(checkpoint['model'], strict=False)

from models.load_weights_model import load_weights
model = load_weights(model, args)
State_CHECK = model.load_state_dict(checkpoint['teacher_model'], strict=True)
print(State_CHECK)
_ = model.eval()


with open('util/node21Nodule_id2name.json') as f: # node21Nodule_id2name
    id2name = json.load(f)
    id2name = {int(k):v for k,v in id2name.items()}


args.dataset_file = 'node21_noduleDataset'
args.coco_path = "/mnt/dfs/nuislam/Data/NODE21_ann/" # the path of coco
args.fix_size = False
dataset_val = build_dataset(image_set='node21_noduleDataset_test', args=args) # node21_noduleDataset_train node21_noduleDataset_test
# len(dataset_val)



import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(), 
])

from util.utils import slprint, to_device
device = torch.device("cuda")

print()


for INDEX in range(0, len(dataset_val)):
# image, targets = dataset_val[750]
    image, targets = dataset_val[INDEX]    


    # print(image.shape, image.min(), image.max())
    image_id = targets['image_id'][0].item()
    print(str(INDEX)+'/'+str(len(dataset_val)),"image_id:", image_id)
    # print("size:",targets['size'])
    # print("boxes:",targets['boxes'])
    # print()


    # build gt_dict for vis
    box_label = [id2name[int(item)] for item in targets['labels']]
    # print(box_label)
    gt_dict = {
        'boxes': targets['boxes'],
        'image_id': targets['image_id'],
        'size': targets['size'],
        'box_label': box_label,
    }
    vslzr = COCOVisualizer()
    vslzr.visualize(image, gt_dict, savedir=None)

    # print(targets)
    # print()


    # print(image[None].shape)
    model.task_DetHead = 0
    targets__ = [{k: to_device(v, device) for k, v in targets.items()}]
    output, _, _ = model.cuda()(image[None].cuda(), targets__)
    output_backup = output

    # print()

    ### print(output)
    ### output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
    # print()
    # print()
    # for k,v in output.items():
    #     print(k)
    # print()
    # print("OUTPUT - Pred_Logits:", output['pred_logits'].shape)
    # print("OUTPUT:", output['pred_logits'])
    # print("OUTPUT MAX:", output['pred_logits'].max())
    # print("OUTPUT pred_bbox:", output['pred_boxes'].shape)
    # print()
    # print()
    output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
    # print()

    # print("Output Scores:")
    # print(output['scores'])
    # print("Done...")


    thershold = 0.3 # set a thershold

    scores = output['scores']
    labels = output['labels']
    # boxes = output['boxes']
    boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
    select_mask = scores > thershold


    # print(select_mask)


    # Ground Truth
    vslzr = COCOVisualizer()
    plt1 = vslzr.visualize(image, gt_dict, dpi=200, savedir=None)
    plt1.tight_layout()
    plt1.savefig("viz_/NODE21/gt.png")

    # Predicted
    box_label = [id2name[int(item)] for item in labels[select_mask]]
    pred_dict = {
        'boxes': boxes[select_mask],
        'size': targets['size'],
        'box_label': box_label
    }
    plt2 = vslzr.visualize(image, pred_dict, dpi=200, savedir=None)
    plt2.tight_layout()
    plt2.savefig("viz_/NODE21/pred.png")

    # print("GroundTruth:", targets)
    # print()
    # print("Predicted:", pred_dict)
    # print()

    # print("mIoU:", calculate_miou(targets['boxes'], pred_dict['boxes'].cpu()) )


    image_path1 = 'viz_/NODE21/gt.png'
    image_path2 = 'viz_/NODE21/pred.png'
    # img1 = mpimg.imread(image_path1)
    # img2 = mpimg.imread(image_path2)
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1)
    axes[0].set_title('GroundTruth')
    axes[0].axis('off') 

    axes[1].imshow(img2)
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('viz_/NODE21/Node21_Test_Image_ID'+str(image_id)+'.png')

print("Done...")