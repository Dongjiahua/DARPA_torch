from typing import Any

import argparse
import torch 
from ultralytics import YOLO
import PIL
import os 
from glob import glob
import json 
import numpy as np
import patchify
import cv2 
import imageio
from itertools import chain
from patchify import patchify, unpatchify
from metric_eval import eval_f1, single_eval
import gc
import math 
import time 
import csv
import sys
import contextlib
import io
from inference import OneshotYOLO
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str,default="/media/jiahua/FILE/uiuc/NCSA/all_patched/all_patched_data/training", help='Root train data path')
    parser.add_argument('--val_data', type=str, default="/media/jiahua/FILE/uiuc/NCSA/all_patched/all_patched_data/validation",  help='Root val data path')
    parser.add_argument('--fct_cfg', type=str, default="/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/config/fct.yaml", help='fct config')
    parser.add_argument('--out_dir', type=str, default="", help='output_dir')
    parser.add_argument('--model', type=str, default="unet_cat", help='backbone model')
    parser.add_argument('--patches', type=int, default=1, help='Patch size.')
    parser.add_argument('--input_size', type=int, default=112, help='Patch size.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size.')
    
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=30, help='Total training epochs.')
    parser.add_argument('--pretrained', action='store_true', help='Whether use pretrained model.')
    parser.add_argument('--freeze', action='store_true',  help='Whether freeze layers.')

    return parser.parse_args()

symbol = [None]
def train():
    args = parse_args()
    torch.manual_seed(0)

    if args.out_dir!="":
        os.makedirs(args.out_dir,exist_ok=True)

    working_dir = "/u/jiahuad2/data/validation_shirui"
    
    model_path = "/u/jiahuad2/codebase/yolo_inference/runs/oneshot-detect/train6/weights/best.pt"

    
    
    tifPaths = glob(working_dir+'/*.tif')
    sorted(tifPaths)

    if not os.path.exists(os.path.join(working_dir, 'Inference')):
        os.mkdir(os.path.join(working_dir, 'Inference'))
    os.makedirs(os.path.join(working_dir, 'Geo'),exist_ok=True)
    count = 0
    polyScore = {}
    f1s = []
    # model = YOLO("yolov8n.yaml")  # build a new model from scratch
    # model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
    model = OneshotYOLO()
    model.load(model_path)

    for tifPath in tifPaths:    
        # if "AR_Maumee" not in tifPath and "CO_Silverton" not in tifPath: continue
        tifFile = tifPath.split('/')[-1]
        jsonFile = tifFile.split('.')[0]+'.json'

        with open(os.path.join(working_dir, jsonFile), 'r') as f:
            jsonData = json.load(f)

        for label_dir in jsonData['shapes']:
            legend = label_dir['label']
            for e in symbol:
                # print(legend,e)
                if legend.endswith("pt") and (e is None or legend.endswith(e)) :
                    
                    print(f"({count+1}/135)",end=' ')
                    score, filename = run(working_dir,model, tifFile, legend, args)
                    if score!=-1:
                        count+=1
                        polyScore[filename] = score
                        f1s.append(score[2])
                        print(f"Average f1: {np.mean(f1s):.3}, Median f1: {np.median(f1s):.3}, Current f1: {score[2]:.3}")
                    
        prefix = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        csv_file =  f"{prefix}_pt.csv"
        os.makedirs("./exp/csv",exist_ok=True)
    polyScore["all"] = (np.mean(f1s), np.median(f1s), np.std(f1s))
    with open(os.path.join("./exp/csv", csv_file), 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in polyScore.items():
            writer.writerow([key, value])

    # eval_f1()

    
patch_dims = (256,256)

    
def build_patch(working_dir, filename, legend, patch_dims = (256,256)):
    """
    filename = 'VA_Lahore_bm.tif'   
    legend = 'SOa_poly'
    """
    
    filePath = os.path.join(working_dir, filename)
    segmentation_file = filePath.split('.')[0]+'_'+legend+'.tif' 
    patch_dims = patch_dims
    
    map_patches_dir = os.path.join(working_dir, filename.split('.')[0]+'_map_patches')
   
    map_im =  cv2.imread(filePath)

    patch_overlap = 32
    patch_step = patch_dims[1]-patch_overlap

    # To patchify, the (width - patch_width) mod step_size = 0
    shift_x = (map_im.shape[0]-patch_dims[0])%patch_step
    shift_y = (map_im.shape[1]-patch_dims[1])%patch_step
    shift_x_left = shift_x//2
    shift_x_right = shift_x - shift_x_left
    shift_y_left = shift_y//2
    shift_y_right = shift_y - shift_y_left

    shift_coord =  [shift_x_left, shift_x_right, shift_y_left, shift_y_right]

    map_im_cut = map_im[shift_x_left:map_im.shape[0]-shift_x_right, shift_y_left:map_im.shape[1]-shift_y_right,:]
    
    map_patchs = patchify(map_im_cut, (*patch_dims,3), patch_step)
    map_cut_shape = map_im_cut.shape
    del map_im_cut
    map_shape = map_patchs.shape
    del map_patchs
    # print(map_patches_dir)
    # if not os.path.exists(map_patches_dir):
    #     os.mkdir(map_patches_dir)  
    #     for i in range(map_patchs.shape[0]):
    #         for j in range(map_patchs.shape[1]):
    #             imageio.imwrite(os.path.join(map_patches_dir, '{0:02d}_{1:02d}.png'.format(i,j)), (map_patchs[i][j][0]).astype(np.uint8))


    ## work on cropping the legend and save it to a subfolder "filename_legend"
    legend_dir = os.path.join(working_dir, filename.split('.')[0]+'_legend')
    
    if not os.path.exists(legend_dir):
        os.mkdir(legend_dir)

    json_file = filePath.split('.')[0]+'.json'
    with open(json_file, 'r') as f:
        jsonData = json.load(f)
        
    point_coord = []
    
    for label_dict in jsonData['shapes']:
        if label_dict['label'] == legend:
            point_coord = label_dict['points']
    if not point_coord: raise Exception("!!!The provided legend does not exist: ", filename, legend)
    flatten_list = list(chain.from_iterable(point_coord))
    
    if point_coord[0][0] >= point_coord[1][0] or point_coord[0][1] >= point_coord[1][1]:
        # print("Coordinate right is less than left:  ", filename, legend, point_coord)
        x_low = min(int(point_coord[0][0]), int(point_coord[1][0]))
        x_hi = max(int(point_coord[0][0]), int(point_coord[1][0]))
        y_low = min(int(point_coord[0][1]), int(point_coord[1][1]))
        y_hi = max(int(point_coord[0][1]), int(point_coord[1][1]))
    elif (len(flatten_list)!=4):
        x_coord = [x[0] for x in point_coord]
        y_coord = [x[1] for x in point_coord]
        x_low, y_low, x_hi, y_hi = int(min(x_coord)), int(min(y_coord)), int(max(x_coord)), int(max(y_coord))
        # print("Point Coordinates number is not 4: ", filename, legend)
    else: x_low, y_low, x_hi, y_hi = [int(x) for x in flatten_list]
        
    legend_coor =  [(x_low, y_low), (x_hi, y_hi)]
    shift_pixel  = 4
    im_crop = map_im[y_low+shift_pixel:y_hi-shift_pixel, x_low+shift_pixel:x_hi-shift_pixel] # need to resize
    # print(os.path.join(legend_dir, legend+'.png'))
    im_crop_resize = cv2.resize(im_crop, dsize=patch_dims, interpolation=cv2.INTER_CUBIC)

    imageio.imwrite(os.path.join(legend_dir, legend+'.png'), (im_crop_resize).astype(np.uint8))
    
    # replace_point_legend_by_template(os.path.join(legend_dir, legend+'.png'), legend) 
    
    return legend_coor, shift_coord, map_cut_shape, map_shape

def map_mask(working_dir, filename, pad_unpatch_predicted_threshold):
    filePath = os.path.join(working_dir, filename)
    imarray = cv2.imread(filePath)
    gray = cv2.cvtColor(imarray, cv2.COLOR_BGR2GRAY)  # greyscale image
    # Detect Background Color
    pix_hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    background_pix_value = np.argmax(pix_hist, axis=None)

    # Flood fill borders
    height, width = gray.shape[:2]
    corners = [[0,0],[0,height-1],[width-1, 0],[width-1, height-1]]
    for c in corners:
        cv2.floodFill(gray, None, (c[0],c[1]), 255)

    # AdaptiveThreshold to remove noise
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    # Edge Detection
    thresh_blur = cv2.GaussianBlur(thresh, (11, 11), 0)
    canny = cv2.Canny(thresh_blur, 0, 200)
    canny_dilate = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Keeping only the largest detected contour.
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    wid, hight = pad_unpatch_predicted_threshold.shape[0], pad_unpatch_predicted_threshold.shape[1]
    mask = np.zeros([wid, hight])
    mask = cv2.fillPoly(mask, pts=[contour], color=(1,1,1)).astype(int)
    masked_img = cv2.bitwise_and(pad_unpatch_predicted_threshold, mask)
    
    return masked_img

def unpatch_img(patches, shape, overlap=0, mode='discard'):
        # Check unpatch_method
        choices = ['and','or','discard']
        if not mode in choices:
            msg = f'unpatchify mode "{mode}" is not a valid option. Valid options are : {choices}'
            raise ValueError(msg)
        
        _, cols, rows, channels, patch_size, patch_size = patches.shape
        h_over = int(overlap/2)

        if mode == 'and':
            image = np.full(shape, 255, dtype=np.float32)
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
            image = np.zeros(shape, dtype=np.uint8)
            
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


def inference(model, image, legend_img, batch_size=16, patch_size=256, patch_overlap=32):

    # Make sure model is set to inference mode
    #self.model.cuda()
    model.eval() 

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

    # Flatten row col dims
    patches = patches.reshape(-1, patch_size, patch_size, 3) 
    print(f"number of patches: {len(patches)}")
    # Resize the legend patch
    norm_legend_img = cv2.resize(legend_img, (patch_size, patch_size))

    # Reshape maps with 1 channel legends (greyscale) to 3 channels for inference
    if map_channels == 1: # This is tmp fix!
        norm_legend_img = np.stack([norm_legend_img,norm_legend_img,norm_legend_img], axis=2)

    # Create legend array to merge with patches
    norm_legend_patches = np.array([norm_legend_img for i in range(rows*cols)])

    # Perform Inference in batches
    prediction_patches = []
    prediction_confs = []
    with torch.no_grad():
        for i in range(0, len(norm_legend_patches), batch_size):
            prediction, confs = model(patches[i:i+batch_size], norm_legend_patches[i:i+batch_size])
            
            prediction_patches += prediction
            prediction_confs += confs

    # Merge patches back into single image and remove padding
    prediction_patches = np.stack(prediction_patches, axis=0)
    prediction_confs = np.stack(prediction_confs, axis=0)
    # prediction_patches = np.array(prediction_patches) # I have no idea why but sometimes model predict outputs a np array and sometimes a tensor array???
    prediction_patches = prediction_patches.reshape([rows, cols, 1, patch_size, patch_size, 1])
    unpatch_image = unpatch_img(prediction_patches.transpose(2,0,1,5,3,4), [1, image.shape[0], image.shape[1]], overlap=patch_overlap)
    #prediction_image = unpatchify(prediction_patches, [image.shape[0], image.shape[1], 1])
    prediction_image = unpatch_image[:,:map_width,:map_height]
    
    prediction_confs = prediction_confs.reshape([rows, cols, 1, patch_size, patch_size, 1])
    unpatch_confs = unpatch_img(prediction_confs.transpose(2,0,1,5,3,4), [1, image.shape[0], image.shape[1]], overlap=patch_overlap)
    confs_image = unpatch_confs[:,:map_width,:map_height]
    
    
    #saveGeoTiff("testdata/debug/sample.tif", prediction_image.astype(np.uint8) * 255, None, None)

    # Convert prediction result to a binary format using a threshold
    #prediction_mask = (prediction_image > 0.5).astype(np.uint8)
    #predictions[label] = prediction_mask
    prediction = prediction_image.transpose(1,2,0).squeeze()
    prediction = (prediction > 0.5).astype(np.uint8)
    confs = confs_image.transpose(1,2,0).squeeze()
    return prediction, confs


def run(working_dir, model, tifFile, legend, args):
    True_Folder = "/u/jiahuad2/data/validation_rasters"
    write_filename = tifFile.split('.')[0]+'_'+legend+'.tif'
    if write_filename not in os.listdir(True_Folder):
        print(f"No GT for {write_filename}!", legend)
        return -1,-1
    
    write_filePath = os.path.join(working_dir, 'Inference', write_filename)
    write_geo_filePath = os.path.join(working_dir, 'Geo', write_filename)
    # if os.path.isfile(write_filePath):
    #     print(f"Already Finished {write_filename}!", legend)
    #     return
    map_img = cv2.imread(os.path.join(working_dir, tifFile))
    map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
    legend_path = os.path.join(working_dir, tifFile.split('.')[0]+'_legend', legend+'.png')
    legend_img = cv2.imread(legend_path)
    legend_img = cv2.cvtColor(legend_img, cv2.COLOR_BGR2RGB)
    start = time.time()
    patched_predicted = inference(model, map_img, legend_img, batch_size=256, patch_size=256, patch_overlap=32)
    print(f"Finished Inference in {time.time()-start:.3f} seconds, map_size: {map_img.shape}")
    print("Finished Inference")
    patched_predicted = patched_predicted.astype(int)
    masked_img = map_mask(working_dir, tifFile, patched_predicted)

    gc.collect()
    # expand one more dimension and repeat the pixel value in the third axis
    final_seg = np.repeat(masked_img[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    cv2.imwrite(write_filePath, final_seg)  
    # write_geoTIFF(filePath,write_geo_filePath,final_seg[:,:,0]) 
    
    precision, recall, f_score = single_eval(write_filename)

    print(f"Finished {write_filename}!", legend)
    return (precision, recall, f_score), write_filename
        
        
        
    
if __name__ == "__main__":
    train()
    
    # model = build_model("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/config/fct.yaml")

        
        