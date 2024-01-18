import torch 
from PIL import Image
from torchvision import transforms
import numpy as np 
from ultralytics import YOLO 
import gc
import cv2
#import nvtx
from time import time
from patchify import patchify, unpatchify

class OneshotYOLO(torch.nn.Module):
    def __init__(self, ) -> None:
        super().__init__()
        self.model = None 
    
    def eval(self):
        return 
        
    def load(self, path):
        self.model = YOLO(path)
        
    def process_image(self,image):  
        if type(image)==np.ndarray:
            image = Image.fromarray(image)
        # zero pad the image to make it square, the length of the longer side, pad on the right and bottom
        if image.width != image.height:
            if image.width > image.height:
                pad = (0, 0, 0, image.width - image.height)
            elif image.width < image.height:
                pad = (0, 0, image.height - image.width,0)
                
            image = transforms.functional.pad(image, pad, 0, 'constant')         
        return image
        
    
    def forward(self, map_patch,legend_patch):
        if self.model is None: 
            raise NotImplementedError("The model hasn't been build yet")
        if type(map_patch)==np.ndarray and len(map_patch.shape==4):
            map_patch = [map_patch[i] for i in range(map_patch.shape[0])]
            img_size = map_patch[0].shape[:2]
        if type(map_patch)!=list:
            if type(map_patch)==np.ndarray:
                img_size = map_patch.shape[:2]
            elif type(map_patch)==Image.Image:
                img_size = np.array(map_patch).shape[:2]
            legend_patch = self.process_image(legend_patch)
            map_patch = self.process_image(map_patch)

        patched_predicted = np.zeros(img_size)
        input = {
            "img": map_patch,
            "legend": legend_patch
        }
        output = self.model(input, verbose=False)[0].boxes

        xywh = output.xywh
        cls = output.cls.cpu().numpy()
        kpts = xywh.cpu().numpy()[:,:2]
        kpts = np.round(kpts).astype(int)
        for i, kpt in enumerate(kpts):
            if int(cls[i]) == 0 and kpt[0] >= 0 and kpt[0] < img_size[1] and kpt[1] >= 0 and kpt[1] < img_size[0]:
                patched_predicted[int(kpt[1]), int(kpt[0])] = 1
        
        return patched_predicted
    




class pipeline_model(object):
    def __init__(self):
        self.name = 'base pipeline model'
        self.model = None

    def load_model(self):
        raise NotImplementedError
    
    def inference(self, image, legend_images, batch_size=16, patch_size=256, patch_overlap=0):
        raise NotImplementedError

class pipeline_pytorch_model(pipeline_model):
    def __init__(self):
        super().__init__()
        self.name = 'base pipeline pytorch model'

        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            logging.error('PyTorch could not load cuda, failing')
            exit(1)

    def transform():
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])])

    # @override
    def inference(self, image, legend_images, batch_size=16, patch_size=256, patch_overlap=0):
        # Make sure model is set to inference mode
        #self.model.cuda()
        self.model.eval() 

        # Get the size of the map
        map_width, map_height, map_channels = image.shape

        # Reshape maps with 1 channel images (greyscale) to 3 channels for inference
        if map_channels == 1: # This is tmp fix!
            image = np.concatenate([image,image,image], axis=2)        

        # Generate patches
        # Pad image so we get a size that can be evenly divided into patches.
        right_pad = patch_size - (map_width % patch_size)
        bottom_pad = patch_size - (map_height % patch_size)
        image = np.pad(image, ((0, right_pad), (0, bottom_pad), (0,0)), mode='constant', constant_values=0)
        patches = patchify(image, (patch_size, patch_size, 3), step=patch_size-patch_overlap)

        rows = patches.shape[0]
        cols = patches.shape[1]

        # Flatten row col dims and normalize map patches to [0,1]
        norm_patches = patches.reshape(-1, patch_size, patch_size, 3)
