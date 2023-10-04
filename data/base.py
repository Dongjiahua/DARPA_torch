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
from data.data_paint import *

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
        
class BaseData(data.Dataset):
    '''
    return:
        map_img: map image (3,224,224)
        legend_img: legend image (3,224,224)
        seg_img: segmentation image (3,224,224)
    '''
    def __init__(self, data_path="",type="poly",args=None, data_range = None):
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
        self.root = data_path
        self.type = type 
        map_path = os.listdir(os.path.join(self.root,self.type,"map_patches"))
        legend_path = ['_'.join(x.split('_')[0:-2])+'.png' for x in map_path]
        
        if data_range is not None:
            map_path = map_path[data_range[0]:data_range[1]]
            legend_path = legend_path[data_range[0]:data_range[1]]
        self.map_path = [os.path.join(self.root,self.type,"map_patches",x) for x in map_path]
        self.legend_path = [os.path.join(self.root,self.type,"legend",x) for x in legend_path]
        self.seg_path = [os.path.join(self.root,self.type,"seg_patches",x) for x in map_path]

    def get_front_legend(self,legend_path):
        sharpend_legend = thresholding(legend_path)
        # convert color from cv2 -> Image
        sharpend_legend = cv2.cvtColor(sharpend_legend, cv2.COLOR_BGR2RGB)
        sharpend_legend = Image.fromarray(sharpend_legend)
        bgrm_legend = remove_bg(sharpend_legend)
        bgrm_legend = crop_RGBA(bgrm_legend)
        # import matplotlib.pyplot as plt
        # plt.imshow(bgrm_legend)
        # plt.show()
        # assert False
        return bgrm_legend 
    
    def RGBA_to_RGB(self,img):
        front_array = np.array(img)
        front_array[front_array[:,:,3]==0,:] = 255
        img = Image.fromarray(front_array).convert("RGB")
        return img
    
    def __getitem__(self, index):
        raise NotImplementedError

    def get_seg_from_bbox(self,point_annotation,seg_img):
        for bbox in point_annotation:
            seg_img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])] = 1
        return seg_img
    
    def get_bbox(self,seg_img,frac = 0.05):
        '''
        return:
            point_annotation: (N,2)
        '''

        points = np.array(np.where(seg_img==1)).T
        # if len(points.shape)==1:
        #     points = points.reshape(1,-1)
        # print(points.shape)
        
        # use boxes around the points as annotations
        point_annotation = np.zeros((len(points),4))
        for i,point in enumerate(points):
            y, x = point[0],point[1]
            range = int(seg_img.shape[0]*frac)
            point_annotation[i,:] = [x-range,y-range,x+range,y+range]
        point_annotation[point_annotation>=seg_img.shape[0]] = seg_img.shape[0]
        point_annotation[point_annotation<0] = 0
        return point_annotation, points[:,[1,0]]
    
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