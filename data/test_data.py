import cv2 
import os 
import numpy as np 
import imageio
import patchify
import json 
from itertools import chain
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
from data.data_paint import *
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

class TestData(data.Dataset):
    '''
    return:
        map_img: map image (3,224,224)
        legend_img: legend image (3,224,224)
        seg_img: segmentation image (3,224,224)
    '''
    def __init__(self, data_paths, legendPath,args):
        if args.patches>1:
            self.image_size = (256,256)
        else:
            self.image_size = (args.input_size,args.input_size)
        self.data_transforms = transforms.Compose([
        transforms.Resize(self.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.root = data_paths
        self.type = type 
        self.legendPath = legendPath
        
    def get_front_legend(self,legend_path):
        
        sharpend_legend = thresholding(legend_path)
        # convert color from cv2 -> Image
        sharpend_legend = cv2.cvtColor(sharpend_legend, cv2.COLOR_BGR2RGB)
        sharpend_legend = Image.fromarray(sharpend_legend)
        bgrm_legend = remove_bg(sharpend_legend)
        bgrm_legend = crop_RGBA(bgrm_legend)
        
        # plt.imshow(bgrm_legend)
        # plt.show()
        # assert False
        return bgrm_legend    
    
    def __getitem__(self, index):
        map_img = Image.open(self.root[index])
        legend_img = self.get_front_legend(self.legendPath)
        front_array = np.array(legend_img)
        front_array[front_array[:,:,3]==0,:] = 255
        legend_img = Image.fromarray(front_array).convert("RGB")
        
        map_img = self.data_transforms(map_img)
        legend_img = self.data_transforms(legend_img)
        img_size = np.array(map_img).shape[-2:]
        return_dict = {
            "map_img": map_img,
            "legend_img": legend_img,
            "metadata":{
                "img_size": img_size,
            }
            
        }
        return return_dict
    
    def __len__(self):
        return len(self.root)