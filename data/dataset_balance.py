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
import copy
import random
training_path = "/media/jiahua/FILE/uiuc/NCSA/processed/training"
validation_path = "/media/jiahua/FILE/uiuc/NCSA/processed/validation"

def crop_and_paste_random(rgb_image, seg_map, dest_rgb, dest_seg, point, size):
    """
    Crop an area from the input RGB image and segmentation map and paste it into
    a random position within the destination images.

    :param rgb_image: numpy array, source RGB image.
    :param seg_map: numpy array, source segmentation map.
    :param dest_rgb: numpy array, destination RGB image.
    :param dest_seg: numpy array, destination segmentation map.
    :param point: tuple of ints (x, y), the center point around which to crop.
    :param size: int, the size of the sides of the square to crop.
    :return: tuple of numpy arrays representing the modified destination RGB image and segmentation map.
    """
    # Check the consistency between RGB images and segmentation maps
    if (rgb_image.shape[0:2] != seg_map.shape[0:2]) or (dest_rgb.shape[0:2] != dest_seg.shape[0:2]):
        raise ValueError("The shape of the RGB image and segmentation map do not match.")

    # Calculate half the size
    half_size = size // 2

    # Define the boundaries of the cropping area
    left = max(0, point[0] - half_size)
    upper = max(0, point[1] - half_size)
    right = min(rgb_image.shape[1], point[0] + half_size)
    lower = min(rgb_image.shape[0], point[1] + half_size)

    # Crop the area from the source images
    cropped_rgb = rgb_image[upper:lower, left:right]
    cropped_seg = seg_map[upper:lower, left:right]

    # Check if cropping was successful (i.e., cropped area is not empty)
    if cropped_rgb.size == 0 or cropped_seg.size == 0:
        raise ValueError("The cropped section is out of the source image boundaries.")

    # Generate random coordinates in the destination images, ensuring the entire cropped section fits
    max_x = dest_rgb.shape[1] - (right - left)
    max_y = dest_rgb.shape[0] - (lower - upper)

    if max_x <= 0 or max_y <= 0:
        raise ValueError("The destination image is too small to fit the cropped section.")

    paste_x = random.randint(0, max_x)
    paste_y = random.randint(0, max_y)

    # Paste the cropped sections into the destination images
    dest_rgb[paste_y:paste_y + (lower - upper), paste_x:paste_x + (right - left)] = cropped_rgb
    dest_seg[paste_y:paste_y + (lower - upper), paste_x:paste_x + (right - left)] = cropped_seg

    return dest_rgb, dest_seg


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
        
class BalanceData(BaseData):
    '''
    return:
        map_img: map image (3,224,224)
        legend_img: legend image (3,224,224)
        seg_img: segmentation image (3,224,224)
    '''
    def __init__(self, data_path="",type="poly",args=None,  data_range=None, size=None, phase=None, end=None,zero_ratio=0.5, paste_ratio=0.5):
        super().__init__(data_path,type,args,data_range)
        self.zero_ratio = zero_ratio
        self.paste_ratio = paste_ratio
        filtered_index = []
        self.size = size
        self.phase=  phase
        self.origin_map_path = copy.deepcopy(self.map_path)
        
        if end is not None:
            for e in end:
                for i in range(len(self.map_path)):
                    if self.legend_path[i].endswith(e):
                        filtered_index.append(i)
        
            print("filtered index: ",len(filtered_index))
            self.map_path = [self.map_path[i] for i in filtered_index]
            self.legend_path = [self.legend_path[i] for i in filtered_index]
            self.seg_path = [self.seg_path[i] for i in filtered_index]
        self.get_balanced_dataset()
        self.get_fraction(end)
        assert phase is not None
        if phase=="train":
            self.data_transforms = transforms.Compose([
                transforms.Resize(self.image_size),
                # transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
            ])      
    
    def get_zero_seg(self,index):
        legend_path = self.legend_path[index]
        point_legend = legend_path.split(".")[-2]
        point_legend = point_legend.split("_")[-2]+"_"+point_legend.split("_")[-1]
        while True:
            map_path = np.random.choice(self.origin_map_path )
            map_legend = map_path.split(".")[-2]
            map_legend = "_".join(map_legend.split("_")[0:-2])
            map_legend = map_legend.split("_")[-2]+"_"+map_legend.split("_")[-1]
            if map_legend != point_legend:
                break
        
        map_path = os.path.join(map_path)
        return map_path
    
    def get_balanced_dataset(self):
        print("get balanced dataset")
        self.legend2map = {}
        for i, path in enumerate(self.map_path):
            map_legend = path.split(".")[-2]
            map_legend = "_".join(map_legend.split("_")[0:-2])
            map_legend = map_legend.split("_")[-2]+"_"+map_legend.split("_")[-1]
            if map_legend not in self.legend2map.keys():
                self.legend2map[map_legend] = []
            self.legend2map[map_legend].append(i)
        self.legendnames = list(self.legend2map.keys())
        # for k in self.legend2map.keys():
        #     print(k,len(self.legend2map[k]))

    def get_fraction(self, end):
        self.max_len = max([len(self.legend2map[k]) for k in self.legend2map.keys()])
        if end is None:
            return
        self.legend2map = {}
        for i, path in enumerate(self.origin_map_path):
            map_legend = path.split(".")[-2]
            map_legend = "_".join(map_legend.split("_")[0:-2])
            map_legend = map_legend.split("_")[-2]+"_"+map_legend.split("_")[-1]
            if map_legend not in self.legend2map.keys():
                self.legend2map[map_legend] = []
            self.legend2map[map_legend].append(i)
        self.legendnames = list(self.legend2map.keys())
        for k in self.legend2map.keys():
            if k in end:
                cur_k = k 
                break
        cur_len = len(self.legend2map[cur_k])
        print(f"cur_len: {cur_len}/max_len: {self.max_len}, fraction: {cur_len/self.max_len}")
        
                
    def __getitem__(self, index):
        index = index%len(self.legendnames)
        legend_name = self.legendnames[index]
        index = np.random.choice(self.legend2map[legend_name])
        fraction = len(self.legend2map[legend_name])/self.max_len
        if 0.1<fraction<0.3:
            zero_ratio = 0.5
        elif fraction<0.1:
            zero_ratio = 0.8
        else:
            zero_ratio = 0
        if np.random.rand()<zero_ratio:
            map_img = Image.open(self.get_zero_seg(index))
            img_size = np.array(map_img).shape[:2]
            seg_img = np.zeros(img_size)
            point_annotation, keypoints = self.get_bbox(seg_img)
        else:
            map_img = np.array(Image.open(self.map_path[index]))
            seg_img = np.array(Image.open(self.seg_path[index]))
            img_size = np.array(map_img).shape[:2]
            point_annotation, keypoints = self.get_bbox(seg_img)
            if np.random.rand()<self.paste_ratio:
                map_img2 = np.array(Image.open(self.get_zero_seg(index)))
                seg_img2 = np.zeros(img_size)
                # crop around one keypoint from map and paste to map2
                kpt = keypoints[np.random.randint(len(keypoints)),:]
                kpt = kpt.astype(int)
                st = np.random.randint(30,100)
                # get the crop box and avoid outside the image 
                
                map_img, seg_img = crop_and_paste_random(map_img, seg_img, map_img2, seg_img2, kpt, st)
                point_annotation, keypoints = self.get_bbox(seg_img)
            map_img = Image.fromarray(map_img)
            
                
        # origin_seg = np.array(seg_img)
        if self.type=="point":
            legend_img = self.get_front_legend(self.legend_path[index])
            legend_img = self.RGBA_to_RGB(legend_img)
            
            seg_img = self.get_seg_from_bbox(point_annotation,seg_img)
            point_annotation[:,[0,2]] = point_annotation[:,[0,2]]/seg_img.shape[1]*self.image_size[0]
            point_annotation[:,[1,3]] = point_annotation[:,[1,3]]/seg_img.shape[0]*self.image_size[1]
            boxes = torch.tensor(point_annotation)
            instance = Instances(self.image_size)
            instance.gt_boxes = Boxes(boxes)
            instance.gt_classes = torch.zeros((len(boxes),),dtype=torch.int64)
            keypoints = torch.tensor(keypoints)
            seg_img = generate_channel_heatmap(seg_img.shape[-2:],keypoints,3,device="cpu")
            
        else:
            legend_img = Image.open(self.legend_path[index])
            seg_img = torch.tensor(seg_img).float()
            keypoints =  torch.tensor([[0,0]])
        map_img = self.data_transforms(map_img)
        legend_img = self.data_transforms(legend_img)
        

        if len(seg_img.shape)==2:
            seg_img = seg_img.unsqueeze(0)
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
        if self.phase=="train" and self.size is not None and self.size>len(self.legendnames):
            repeat = self.size//len(self.legendnames)
            return len(self.legendnames)*repeat
        return len(self.legendnames)

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