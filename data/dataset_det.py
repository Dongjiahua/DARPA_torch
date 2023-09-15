import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import PIL
import os 
from detectron2.structures import Boxes, ImageList, Instances
training_path = "/media/jiahua/FILE/uiuc/NCSA/processed/training"
validation_path = "/media/jiahua/FILE/uiuc/NCSA/processed/validation"

class DetData(data.Dataset):
    '''
    return:
        map_img: map image (3,224,224)
        legend_img: legend image (3,224,224)
        seg_img: segmentation image (3,224,224)
    '''
    def __init__(self, data_path=training_path,type="poly",range=None):
        self.image_size = (224,224)
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

        seg_img = np.array(seg_img)
        # origin_seg = np.array(seg_img)
        assert self.type=="point"
        point_annotation = self.get_bbox(seg_img)
        seg_img = self.get_seg_from_bbox(point_annotation,seg_img)
        point_annotation[:,[0,2]] = point_annotation[:,[0,2]]/seg_img.shape[1]*self.image_size[0]
        point_annotation[:,[1,3]] = point_annotation[:,[1,3]]/seg_img.shape[0]*self.image_size[1]
        boxes = torch.tensor(point_annotation)
        instance = Instances(self.image_size)
        instance.gt_boxes = Boxes(boxes)
        instance.gt_classes = torch.zeros((len(boxes),),dtype=torch.int64)


        map_img = self.data_transforms(map_img)
        legend_img = self.data_transforms(legend_img)
        seg_img = torch.tensor(seg_img).float().unsqueeze(0)

        return_dict = {
            "map_img": map_img,
            "legend_img": legend_img,
            "seg_img": seg_img,
            "instance": instance
        }
        return return_dict

    def get_seg_from_bbox(self,point_annotation,seg_img):
        for bbox in point_annotation:
            seg_img[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])] = 1
        return seg_img
    
    def get_bbox(self,seg_img,frac = 0.08):
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
        return point_annotation
    
    def __len__(self):
        return len(self.map_path)

def collect_fn_det(batch):
    # print(batch)
    dics = batch
    return_dict = {}
    for k in dics[0].keys():
        if k!="instance":
            return_dict[k] = torch.stack([dic[k] for dic in dics],dim=0)
        else:
            return_dict[k] = [dic[k] for dic in dics]
    return return_dict