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
            prediction_patches = np.stack(prediction_patches, axis=0) # I have no idea why but sometimes model predict outputs a np array and sometimes a tensor array???
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
        self.name = 'flat iceberg'
        self.checkpoint = '/projects/bbym/shared/models/flat-iceberg/best.pt'
    
        # Modifiable parameters
        self.device = torch.device("cuda")
        self.batch_size = 256
        self.patch_size = 256
        self.patch_overlap = 32
        self.unpatch_mode = 'discard'

    # @override
    def load_model(self):
        self.model = OneshotYOLO()
        self.model.load(self.checkpoint)
        self.model.eval()

    # @override
    def inference(self, image, legend_images):
        """Image data is in CHW format. legend_images is a dictionary of label to map_unit label images in CHW format."""

        # Get the size of the map
        map_width, map_height, map_channels = image.shape

        # Reshape maps with 1 channel images (greyscale) to 3 channels for inference
        if map_channels == 1: # This is tmp fix!
            image = np.concatenate([image,image,image], axis=2)        

        # Generate patches
        # Pad image so we get a size that can be evenly divided into patches.
        right_pad = self.patch_size - (map_width % self.patch_size)
        bottom_pad = self.patch_size - (map_height % self.patch_size)
        padded_image = np.pad(image, ((0,0), (0, bottom_pad), (0, right_pad)), mode='constant', constant_values=0)
        map_patches = patchify(padded_image, (3, self.patch_size, self.patch_size), step=self.patch_size-self.patch_overlap)

        cols = map_patches.shape[1]
        rows = map_patches.shape[2]

        # Flatten row col dims
        map_patches = map_patches.reshape(-1, 3, self.patch_size, self.patch_size)
        
        log.debug(f"\tMap size: {map_width}, {map_height} patched into : {rows} x {cols} = {rows*cols} patches")
        map_prediction = np.zeros((1, map_height, map_width), dtype=np.float32)
        map_confidence = np.zeros((1, map_height, map_width), dtype=np.float32)
        legend_index = 1
        for label, legend_img in legend_images.items():
            log.debug(f'\t\tInferencing legend: {label}')
            lgd_stime = time()

            # Resize the legend patch
            legend_img = cv2.resize(legend_img, (self.patch_size, self.patch_size))

            # Reshape maps with 1 channel legends (greyscale) to 3 channels for inference
            if legend_img.shape[0] == 1:
                legend_img = np.concatenate([legend_img,legend_img,legend_img], axis=0)

            # Create legend array to merge with patches
            legend_patches = np.array([legend_img for i in range(rows*cols)])

            # Perform Inference in batches
            prediction_patches = []
            with torch.no_grad():
                for i in range(0, len(map_patches), self.batch_size):
                    prediction = self.model(map_patches[i:i+self.batch_size], legend_patches[:len(map_patches[i:i+self.batch_size])])
                    prediction_patches += prediction
            
            # Merge patches back into single image and remove padding
            prediction_patches = np.stack(prediction_patches, axis=0)
            # prediction_patches = np.array(prediction_patches) # I have no idea why but sometimes model predict outputs a np array and sometimes a tensor array???
            prediction_patches = prediction_patches.reshape([1, cols, rows, 1, self.patch_size, self.patch_size])
            unpatch_image = unpatch_img(prediction_patches, [1, padded_image.shape[1], padded_image.shape[2]], overlap=self.patch_overlap, mode=self.unpatch_mode)
            #prediction_image = unpatchify(prediction_patches, [image.shape[0], image.shape[1], 1])
            prediction_image = unpatch_image[:,:map_width,:map_height]
            #saveGeoTiff("testdata/debug/sample.tif", prediction_image.astype(np.uint8) * 255, None, None)

            # Add legend to prediction mask
            map_prediction[prediction_image >= map_confidence] = legend_index
            map_confidence = np.maximum(map_confidence, prediction_image)
            
            gc.collect() # This is needed otherwise gpu memory is not freed up on each loop

            legend_index += 1
            lgd_time = time() - lgd_stime
            log.debug("\t\tExecution time for {} legend: {:.2f} seconds. {:.2f} patches per second".format(label, lgd_time, (rows*cols)/lgd_time))

        # Minimum confidence threshold for a prediction
        map_prediction[map_confidence < 0.333] = 0
        
        return map_prediction

class flat_iceberg_model_multiclass(pipeline_pytorch_model):
    def __init__(self):
        super().__init__()
        self.name = 'flat iceberg'
        self.checkpoint = '/projects/bbym/shared/models/flat-iceberg/multi_class.pt'
    
        # Modifiable parameters
        self.device = torch.device("cuda")
        self.batch_size = 256
        self.patch_size = 256
        self.patch_overlap = 32
        self.unpatch_mode = 'discard'

    # @override
    def load_model(self):
        self.model = OneshotYOLO()
        self.model.load(self.checkpoint)
        self.model.eval()

    # @override
    def inference(self, image):
        """Image data is in CHW format. legend_images is a dictionary of label to map_unit label images in CHW format."""

        # Get the size of the map
        map_width, map_height, map_channels = image.shape

        # Reshape maps with 1 channel images (greyscale) to 3 channels for inference
        if map_channels == 1: # This is tmp fix!
            image = np.concatenate([image,image,image], axis=2)        

        # Generate patches
        # Pad image so we get a size that can be evenly divided into patches.
        right_pad = self.patch_size - (map_width % self.patch_size)
        bottom_pad = self.patch_size - (map_height % self.patch_size)
        padded_image = np.pad(image, ((0,0), (0, bottom_pad), (0, right_pad)), mode='constant', constant_values=0)
        map_patches = patchify(padded_image, (3, self.patch_size, self.patch_size), step=self.patch_size-self.patch_overlap)

        cols = map_patches.shape[1]
        rows = map_patches.shape[2]

        # Flatten row col dims
        map_patches = map_patches.reshape(-1, 3, self.patch_size, self.patch_size)
        
        log.debug(f"\tMap size: {map_width}, {map_height} patched into : {rows} x {cols} = {rows*cols} patches")
        map_prediction = np.zeros((1, map_height, map_width), dtype=np.float32)
        map_confidence = np.zeros((1, map_height, map_width), dtype=np.float32)


        log.debug(f'\t\tInferencing legend: {label}')
        lgd_stime = time()


        # Perform Inference in batches
        prediction_patches = []
        with torch.no_grad():
            for i in range(0, len(map_patches), self.batch_size):
                prediction = self.model(map_patches[i:i+self.batch_size])
                prediction_patches += prediction
        
        # Merge patches back into single image and remove padding
        prediction_patches = np.stack(prediction_patches, axis=0)
        # prediction_patches = np.array(prediction_patches) # I have no idea why but sometimes model predict outputs a np array and sometimes a tensor array???
        prediction_patches = prediction_patches.reshape([1, cols, rows, 1, self.patch_size, self.patch_size])
        unpatch_image = unpatch_img(prediction_patches, [1, padded_image.shape[1], padded_image.shape[2]], overlap=self.patch_overlap, mode=self.unpatch_mode)
        #prediction_image = unpatchify(prediction_patches, [image.shape[0], image.shape[1], 1])
        prediction_image = unpatch_image[:,:map_width,:map_height]
        #saveGeoTiff("testdata/debug/sample.tif", prediction_image.astype(np.uint8) * 255, None, None)


            
        gc.collect() # This is needed otherwise gpu memory is not freed up on each loop


        lgd_time = time() - lgd_stime
        # log.debug("\t\tExecution time for {} legend: {:.2f} seconds. {:.2f} patches per second".format(label, lgd_time, (rows*cols)/lgd_time))


        
        return map_prediction
