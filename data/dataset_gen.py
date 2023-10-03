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
import pandas as pd
from data.data_paint import *
training_path = "/media/jiahua/FILE/uiuc/NCSA/processed/training"
validation_path = "/media/jiahua/FILE/uiuc/NCSA/processed/validation"
import matplotlib.pyplot as plt


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
        
class GENData(data.Dataset):
    '''
    return:
        map_img: map image (3,224,224)
        legend_img: legend image (3,224,224)
        seg_img: segmentation image (3,224,224)
    '''
    def __init__(self, data_path=training_path,type="poly",args=None, size=10000):
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
        self.legend_trans = transforms.Compose([
            transforms.RandomAffine(0, translate=(0.1,0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.2, hue=0.3),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
        ])
        self.root = data_path
        self.type = type 
        self.map_path = os.listdir(os.path.join(self.root,self.type,"map_patches"))
        legend_path = os.listdir(os.path.join(self.root,self.type,"legend"))
        self.size = size
        self.score_csv = pd.read_csv("data/20230923_140953_pt.csv")
        self.legend_path = []
        used_legend = []
        all_scores = []
        # for i in range(len(self.score_csv)):
        #     all_scores.append(eval(self.score_csv.iloc[i,1])[2])
        #     if eval(self.score_csv.iloc[i,1])[2]<0.2:
        #         point_legend = self.score_csv.iloc[i,0].split(".")[0]
        #         point_legend = point_legend.split("_")[-3]+"_"+point_legend.split("_")[-2]+"_"+point_legend.split("_")[-1]
        #         for k in legend_path:
        #             if point_legend in k:
        #                 used_legend.append(k)
        #                 break
        used_legend = legend_path
        print(f"median score: {np.median(np.array(all_scores))}")
        frequency = size//len(used_legend)
        for legend in used_legend:
            legend = os.path.join(self.root,self.type,"legend",legend)
            self.legend_path += [legend]*frequency
        # if range is not None:
        #     map_path = map_path[range[0]:range[1]]
        #     legend_path = legend_path[range[0]:range[1]]
        # self.map_path = [os.path.join(self.root,self.type,"map_patches",x) for x in map_path]
        # self.legend_path = [os.path.join(self.root,self.type,"legend",x) for x in legend_path]
        # self.seg_path = [os.path.join(self.root,self.type,"seg_patches",x) for x in map_path]
    
    def get_pairs(self, idx):
        legend_path = self.legend_path[idx]
        point_legend = legend_path.split(".")[0]
        point_legend = point_legend.split("_")[-3]+"_"+point_legend.split("_")[-2]+"_"+point_legend.split("_")[-1]
        while True:
            map_path = np.random.choice(self.map_path)
            map_legend = map_path.split(".")[0]
            map_legend = "_".join(map_legend.split("_")[0:-2])
            map_legend = map_legend.split("_")[-3]+"_"+map_legend.split("_")[-2]+"_"+map_legend.split("_")[-1]
            if map_legend != point_legend:
                break
        
        map_path = os.path.join(self.root,self.type,"map_patches",map_path)
        return map_path, legend_path
    
    def get_front_legend(self,legend_path):
        sharpend_legend = thresholding(legend_path)
        # convert color from cv2 -> Image
        sharpend_legend = cv2.cvtColor(sharpend_legend, cv2.COLOR_BGR2RGB)
        sharpend_legend = Image.fromarray(sharpend_legend)
        bgrm_legend = remove_bg(sharpend_legend)
        bgrm_legend = crop_RGBA(bgrm_legend)
        import matplotlib.pyplot as plt
        # plt.imshow(bgrm_legend)
        # plt.show()
        # assert False
        return bgrm_legend 
    
    def paint_legend(self,map, legend, img_size):
        

        
        # randomly sample coordinates on the map with threshold distance
        coords = []
        num = np.random.randint(3,10)
        while True:
            y = np.random.randint(0,img_size[0])
            x = np.random.randint(0,img_size[1])
            if len(coords)==0:
                coords.append([x,y])
            else:
                dist = np.sqrt((np.array(coords)-np.array([x,y]))**2)
                if np.min(dist)>10:
                    coords.append([x,y])
            if len(coords)==num:
                break
        # map.save("testx.png")
        scale0 = np.random.choice([15]*5+[20]*3+[30])
        scale1 = int(scale0*(1+(np.random.rand()-0.5)*2*0.4))
        bright = np.random.randint(0,150)
        for i,coord in enumerate(coords):
            angle = np.random.randint(0,360)
            rotated_legend = legend.rotate(angle)
            
            scaled_legend = rotated_legend.resize((scale0,scale1))
            np_img = np.array(scaled_legend)
            mask = np_img[:,:,3]==0
            jittered_img = self.legend_trans(scaled_legend)
            jittered_img = np.array(jittered_img)
            jittered_img[mask,:] = np.array([225,225,225,0])
            jittered_img[...,:3] = np.clip(jittered_img[...,:3]+bright,0,255)
            scaled_legend = Image.fromarray(jittered_img)
            # print(f"map max:{np.max(np.array(map))}, legend max:{np.max(np.array(scaled_legend))}")
            # print
            map = merge_bg(scaled_legend,map,location=(coord[0]-scale0//2,coord[1]-scale1//2))
        #     map.save(f"test_{i}.png")
        # scaled_legend.save("test1.png")
        # assert False
        return map, np.array(coords)
            
        
        
    def __getitem__(self, index):
        map_path, legend_path = self.get_pairs(index)
        map_img = Image.open(map_path)
        map_img = transforms.RandomResizedCrop(map_img.size,scale=(0.4,1.0))(map_img)
        legend_img = self.get_front_legend(legend_path)
        
        img_size = np.array(map_img).shape[:2]
        seg_img = np.zeros(img_size)
        # origin_seg = np.array(seg_img)
        assert self.type=="point"
        map_img, keypoints = self.paint_legend(map_img, legend_img, img_size)
        front_array = np.array(legend_img)
        front_array[front_array[:,:,3]==0,:] = 255
        legend_img = Image.fromarray(front_array).convert("RGB")
        # print(np.array(map_img).shape)
        map_img = self.data_transforms(map_img)
        legend_img = self.data_transforms(legend_img)
        seg_img = torch.tensor(seg_img).float()

        

        keypoints = torch.tensor(keypoints)
        seg_img = generate_channel_heatmap(seg_img.shape[-2:],keypoints,3,device="cpu")
        # print(f"map max:{torch.max(map_img)}, legend max:{torch.max(legend_img)}")
        # print(seg_img.shape)
        return_dict = {
            "map_img": map_img,
            "legend_img": legend_img,
            "seg_img": seg_img,
            "keypoints": keypoints,
            "metadata":{
                "img_size": img_size,
            }
        }
        return return_dict

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
        return len(self.legend_path)

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