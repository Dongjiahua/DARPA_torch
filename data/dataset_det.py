import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import PIL
import os 
from detectron2.structures import Boxes, ImageList, Instances
from utils.heatmap import generate_channel_heatmap
from data.base import BaseData

training_path = "/media/jiahua/FILE/uiuc/NCSA/processed/training"
validation_path = "/media/jiahua/FILE/uiuc/NCSA/processed/validation"

def crop_RGBA(image):
    """
    Crop a RGBA image's transparent border.
    
    Parameters:
        image (PIL.Image.Image): Input RGBA image.
    
    Returns:
        np.ndarray: Cropped image.
    """
    image_array = np.array(image)
    mask = image_array[:,:,3] != 0
    
    # Getting the bounding box
    x_nonzero, y_nonzero = np.nonzero(mask)
    x_min, x_max = np.min(x_nonzero), np.max(x_nonzero)
    y_min, y_max = np.min(y_nonzero), np.max(y_nonzero)
    
    return Image.fromarray(image_array[x_min:x_max+1, y_min:y_max+1, :]).resize(mask.shape[:2])

# optimize the crop_RGBA code:

class metric:
    def __init__(self) -> None:
        self.total = 0
        self.count = 0
    
    def update(self,num):
        self.total += num
        self.count += 1
    
    def compute(self):
        return self.total/self.count
    
    def reset(self):
        self.total = 0
        self.count = 0
        
class DetData(BaseData):
    '''
    return:
        map_img: map image (3,224,224)
        legend_img: legend image (3,224,224)
        seg_img: segmentation image (3,224,224)
    '''
        
    def __getitem__(self, index):
        map_img = Image.open(self.map_path[index])
        legend_img = self.get_front_legend(self.legend_path[index])
        seg_img = Image.open(self.seg_path[index])
        
        front_array = np.array(legend_img)
        front_array[front_array[:,:,3]==0,:] = 255
        legend_img = Image.fromarray(front_array).convert("RGB")
        

        img_size = np.array(map_img).shape[:2]
        seg_img = np.array(seg_img)
        # origin_seg = np.array(seg_img)
        assert self.type=="point"
        point_annotation, keypoints = self.get_bbox(seg_img)
        seg_img = self.get_seg_from_bbox(point_annotation,seg_img)
        point_annotation[:,[0,2]] = point_annotation[:,[0,2]]/seg_img.shape[1]*self.image_size[0]
        point_annotation[:,[1,3]] = point_annotation[:,[1,3]]/seg_img.shape[0]*self.image_size[1]
        boxes = torch.tensor(point_annotation)
        instance = Instances(self.image_size)
        instance.gt_boxes = Boxes(boxes)
        instance.gt_classes = torch.zeros((len(boxes),),dtype=torch.int64)


        map_img = self.data_transforms(map_img)
        legend_img = self.data_transforms(legend_img)
        seg_img = torch.tensor(seg_img).float()

        

        keypoints = torch.tensor(keypoints)
        seg_img = generate_channel_heatmap(seg_img.shape[-2:],keypoints,3,device="cpu")

        # print(seg_img.shape)
        return_dict = {
            "map_img": map_img,
            "legend_img": legend_img,
            "seg_img": seg_img,
            # "instance": instance,
            "keypoints": keypoints,
            "metadata":{
                "img_size": img_size,
            }
        }
        return return_dict

    
    
    def __len__(self):
        return len(self.map_path)

def collect_fn_det(batch):
    # print(batch)
    dics = batch
    return_dict = {}
    for k in dics[0].keys():
        if k=="instance" or k =="keypoints":
            return_dict[k] = [dic[k] for dic in dics]
        elif k=="metadata":
            return_dict[k] = dics[0][k]
        else:
            return_dict[k] = torch.stack([dic[k] for dic in dics],dim=0)
    return return_dict