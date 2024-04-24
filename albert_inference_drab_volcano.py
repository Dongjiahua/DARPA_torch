import gc
import cv2
#import nvtx
import logging
import numpy as np

import torch
from torchvision import transforms

from time import time
# from src.patching import unpatch_img
from patchify import patchify, unpatchify
from inference_multi_class import MulticlassYOLO
import json

log = logging.getLogger('DARPA_CMAAS_PIPELINE')

def unpatch_img(patches, shape, overlap=0, mode='discard'):
        # Check unpatch_method
        choices = ['and','or','discard']
        if not mode in choices:
            msg = f'unpatchify mode "{mode}" is not a valid option. Valid options are : {choices}'
            log.error(msg)
            raise ValueError(msg)
        
        _, cols, rows, channels, patch_size, patch_size = patches.shape
        h_over = int(overlap/2)

        if mode == 'and':
            image = np.full(shape, 255, dtype=np.uint8)
            #image = np.full((shape[0], shape[1]+overlap, shape[2]+overlap), 255, dtype=np.uint8)
            for col in range(0,cols):
                for row in range(0,rows):
                    y_min = col*(patch_size-overlap) 
                    y_max = (col+1)*patch_size - (col*overlap)
                    x_min = row*(patch_size-overlap)
                    x_max = (row+1)*patch_size - (row*overlap)
                    #log.debug(f':, {y_min}:{y_max}, {x_min}:{x_max}')
                    image[:, y_min:y_max, x_min:x_max] &= patches[0, col, row, :, :, :]
            #image = image[:,h_over:shape[1]+h_over, h_over:shape[2]+h_over]
        if mode == 'or':
            image = np.zeros(shape, dtype=np.uint8)
            #image = np.zeros((shape[0], shape[1]+overlap, shape[2]+overlap), dtype=np.uint8)
            for col in range(0,cols):
                for row in range(0,rows):
                    y_min = col*(patch_size-overlap) 
                    y_max = (col+1)*patch_size - (col*overlap)
                    x_min = row*(patch_size-overlap)
                    x_max = (row+1)*patch_size - (row*overlap)
                    #log.debug(f':, {y_min}:{y_max}, {x_min}:{x_max}')
                    image[:, y_min:y_max, x_min:x_max] |= patches[0, col, row, :, :, :]
            #image = image[:,h_over:shape[1]+h_over, h_over:shape[2]+h_over]
        if mode == 'discard':
            image = np.zeros(shape, dtype=np.float32)
            
            for col in range(0,cols):
                for row in range(0,rows):
                    y_min = h_over + col*(patch_size-overlap)
                    y_max = h_over + (col+1)*(patch_size-overlap)
                    x_min = h_over + row*(patch_size-overlap)
                    x_max = h_over + (row+1)*(patch_size-overlap)
                    #log.debug(f'{col},{row} = :, {y_min}:{y_max}, {x_min}:{x_max}')
                    py_min = h_over
                    py_max = patch_size-h_over
                    px_min = h_over
                    px_max = patch_size-h_over
                    
                    if row == 0: 
                        x_min -= h_over
                        px_min = 0
                    if row == rows-1:
                        x_max += h_over
                        px_max = patch_size
                    if col == 0:
                        y_min -= h_over
                        py_min = 0
                    if col == cols-1:
                        y_max += h_over
                        py_max = patch_size
                    image[:, y_min:y_max, x_min:x_max] = patches[0, col, row, :, py_min:py_max, px_min:px_max]
        return image

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
    

class drab_volcano_model(pipeline_pytorch_model):
    def __init__(self):
        super().__init__()
        self.name = 'drab volcano'
        self.checkpoint = '/projects/bbym/shared/models/drab-volcano/best.pt'
        self.label_name_path = '/projects/bbym/shared/models/drab-volcano/legend_name2class.json'
        # Modifiable parameters
        self.device = torch.device("cuda")
        self.batch_size = 256
        self.patch_size = 256
        self.patch_overlap = 32
        self.unpatch_mode = 'discard'

    # @override
    def load_model(self):
        self.model = MulticlassYOLO()
        self.model.load(self.checkpoint)
        self.model.eval()

        label_names = json.load(open(self.label_name_path))
        self.name_to_class = {name: int(label_names[i]["id"]) for i in range(len(label_names)) for name in label_names[i]["names"]}

    # @override
    def inference_with_refactor(self, image, legend_names=None):
        """Image data is in CHW format. legend_images is a dictionary of label to map_unit label images in CHW format."""

        # Get the size of the map
        map_height, map_width, map_channels = image.shape

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

        print(map_patches.shape)
        # Flatten row col dims
        map_patches = map_patches.reshape(-1, 3, self.patch_size, self.patch_size)
        
        log.debug(f"\tMap size: {map_width}, {map_height} patched into : {rows} x {cols} = {rows*cols} patches")



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
        
        if legend_names is not None:
            log.debug(f"Refactoring prediction to known legend names!")
            idx = 1
            final_prediction = np.zeros_like(prediction_image)
            for name in legend_names:
                if name not in self.name_to_class:
                    log.warning(f"Unknown legend name: {name}, return the unrefactored prediction")
                    return prediction_image
                final_prediction[prediction_image == self.name_to_class[name]] = idx
                idx += 1
        else:
            final_prediction = prediction_image
            
        gc.collect() # This is needed otherwise gpu memory is not freed up on each loop
        
        return prediction_image

    # @override
    def inference(self, image, legend_images=None):
        """Image data is in CHW format. legend_images is a dictionary of label to map_unit label images in CHW format."""         

        # Get the size of the map
        map_height, map_width, map_channels = image.shape

        # Reshape maps with 1 channel images (greyscale) to 3 channels for inference
        if map_channels == 1: # This is tmp fix!
            image = np.concatenate([image,image,image], axis=0)        

        # Generate patches
        # Pad image so we get a size that can be evenly divided into patches.
        right_pad = self.patch_size - (map_width % self.patch_size)
        bottom_pad = self.patch_size - (map_height % self.patch_size)
   
        
        # pad the H, W, 3 image, pad at the right and bottom
        padded_image = np.pad(image, ((0,0), (0, bottom_pad), (0, right_pad)), mode='constant', constant_values=0)

        map_patches = patchify(padded_image, (3, self.patch_size, self.patch_size), step=self.patch_size-self.patch_overlap)
        # print(map_patches.shape)
        cols = map_patches.shape[0]
        rows = map_patches.shape[1]

        # Flatten row col dims
        map_patches = map_patches.reshape(-1, 3, self.patch_size, self.patch_size)
        
        # transpose BCHW to BHWC
        map_patches = np.transpose(map_patches, (0, 2, 3, 1))

        log.debug(f"\tMap size: {map_width}, {map_height} patched into : {rows} x {cols} = {rows*cols} patches")



        # Perform Inference in batches
        prediction_patches = []
        with torch.no_grad():
            for i in range(0, len(map_patches), self.batch_size):
                prediction,_ = self.model(map_patches[i:i+self.batch_size])
                prediction_patches += prediction
        
        # Merge patches back into single image and remove padding
        prediction_patches = np.stack(prediction_patches, axis=0)
        # print(prediction_patches.shape)
        # prediction_patches = np.array(prediction_patches) # I have no idea why but sometimes model predict outputs a np array and sometimes a tensor array???
        prediction_patches = prediction_patches.reshape([1, cols, rows, 1, self.patch_size, self.patch_size])

        unpatch_image = unpatch_img(prediction_patches, [1, padded_image.shape[1], padded_image.shape[2]], overlap=self.patch_overlap, mode=self.unpatch_mode)
        #prediction_image = unpatchify(prediction_patches, [image.shape[0], image.shape[1], 1])
        prediction_image = unpatch_image[:,:map_width,:map_height]
        #saveGeoTiff("testdata/debug/sample.tif", prediction_image.astype(np.uint8) * 255, None, None)
        
        
        final_prediction = prediction_image
            
        gc.collect() # This is needed otherwise gpu memory is not freed up on each loop
        
        return prediction_image


if __name__ == '__main__':
    import os
    from PIL import Image
    import matplotlib.pyplot as plt


    # Load the model
    model = drab_volcano_model()
    model.load_model()

    # Load the image
    image_path = "/projects/bbym/shared/data/sortedDatasets/validation/images/NM_Capitan_189999_1934_48000_geo_mosaic.tif"

    image = Image.open(image_path)
    
    image = np.array(image)
    print(image.shape)
    # Perform inference
    prediction = model.inference(image)

    prediction+=1
    # draw non-zero pixels as circles
    nonzero = np.nonzero(prediction)
    plt.scatter(nonzero[1], nonzero[0], s=1)
    plt.imsave("prediction.png", prediction)