import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import PIL
import os 
training_path = "/media/jiahua/FILE/uiuc/NCSA/processed/training"
validation_path = "/media/jiahua/FILE/uiuc/NCSA/processed/validation"

class MAPData(data.Dataset):
    '''
    return:
        map_img: map image (3,224,224)
        legend_img: legend image (3,224,224)
        seg_img: segmentation image (3,224,224)
    '''
    def __init__(self, data_path=training_path,type="poly",range=None):
        self.data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.root = data_path
        self.type = type 
        map_path = os.listdir(os.path.join(self.root,self.type,"map_patches"))
        legend_path = ['_'.join(x.split('_')[0:-2])+'.png' for x in map_path]
        
        if range is not None:
            map_path = map_path[range[0]:range[1]]
            legend_path = legend_path[range[0]:range[1]]
        self.map_path = [os.path.join(self.root,self.type,"map_patches",x) for x in map_path]
        self.legend_path = [os.path.join(self.root,self.type,"legend",x) for x in legend_path]
        self.seg_path = [os.path.join(self.root,self.type,"seg_patches",x) for x in map_path]
        
    def __getitem__(self, index):
        map_img = Image.open(self.map_path[index])
        legend_img = Image.open(self.legend_path[index])
        seg_img = Image.open(self.seg_path[index])
        
        map_img = self.data_transforms(map_img)
        legend_img = self.data_transforms(legend_img)
        # print(seg_img.max())
        # print(np.asarray(seg_img).max())
        seg_img = torch.tensor(np.asarray(seg_img)).float().unsqueeze(0)

        
        return {
            "map_img": map_img,
            "legend_img": legend_img,
            "seg_img": seg_img
        }
    
    def __len__(self):
        return len(self.map_path)
