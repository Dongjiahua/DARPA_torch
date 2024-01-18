import gc
import cv2
#import nvtx
import logging
import numpy as np

import torch
from torchvision import transforms

from time import time
from patchify import patchify, unpatchify
from submodules.models.flat_iceberg.inference_model import OneshotYOLO
log = logging.getLogger('DARPA_CMAAS_PIPELINE')



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
        patches = patches.reshape(-1, patch_size, patch_size, 3) 
        
        rows = patches.shape[0]
        cols = patches.shape[1]
        #NOTICE: Leave the transformation to the model since it may differ for different method

        log.debug(f"\tMap size: {map_width}, {map_height} patched into : {rows} x {cols} = {rows*cols} patches")
        predictions = {}
        for label, legend_img in legend_images.items():
            log.debug(f'\t\tInferencing legend: {label}')
            lgd_stime = time()

            norm_legend_img = cv2.resize(legend_img, (patch_size, patch_size))

            # Reshape maps with 1 channel legends (greyscale) to 3 channels for inference
            if map_channels == 1: # This is tmp fix!
                norm_legend_img = np.stack([norm_legend_img,norm_legend_img,norm_legend_img], axis=2)

            # Create legend array to merge with patches
            norm_legend_patches = np.array([norm_legend_img for i in range(rows*cols)])

            # Perform Inference in batches
            prediction_patches = []
            with torch.no_grad():
                for i in range(0, len(norm_legend_patches), batch_size):
                    prediction = self.model(patches[i:i+batch_size], norm_legend_patches[i:i+batch_size])
                
                    if prediction_patches is None:
                        prediction_patches = prediction
                    else:
                        prediction_patches+=(prediction)

            # Merge patches back into single image and remove padding
            prediction_patches = np.array(prediction_patches) # I have no idea why but sometimes model predict outputs a np array and sometimes a tensor array???
            prediction_patches = prediction_patches.reshape([rows, cols, 1, patch_size, patch_size, 1])
            prediction_image = unpatchify(prediction_patches, [image.shape[0], image.shape[1], 1])
            prediction_image = prediction_image[:map_width,:map_height,:]

            # Convert prediction result to a binary format using a threshold
            prediction_mask = (prediction_image > 0.5).astype(np.uint8)
            predictions[label] = prediction_mask
            gc.collect() # This is needed otherwise gpu memory is not freed up on each loop

            lgd_time = time() - lgd_stime
            log.debug("\t\tExecution time for {} legend: {:.2f} seconds. {:.2f} patches per second".format(label, lgd_time, (rows*cols)/lgd_time))
            
        return predictions
    
class flat_iceberg_model(pipeline_pytorch_model):
    def __init__(self):
        super().__init__()
        self.name = 'flat iceberg model'
    
    def load_model(self):
        self.model = OneshotYOLO()
        self.model.load('/projects/bbym/shared/models/flat-iceberg/best.pt')
        self.model.eval()