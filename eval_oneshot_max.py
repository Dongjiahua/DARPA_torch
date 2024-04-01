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
    
    model_path = "/u/jiahuad2/codebase/yolo_inference/runs/oneshot-detect/train55/weights/last.pt"

    
    
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

        legends = [label_dir['label'] for label_dir in jsonData['shapes'] if label_dir['label'].endswith("pt") or label_dir['label'].endswith("pt")]
        if len(legends) == 0:
            continue
        scores, filenames = run(working_dir,model, tifFile, legends, args)
        if type(scores) == int:
            continue
        for score, filename in zip(scores, filenames):
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
            image = np.zeros(shape, dtype=np.float32)
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


def inference(model, image, legend_imgs, batch_size=16, patch_size=256, patch_overlap=32):

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
    
    all_predictions = []
    all_confs = []
    for legend_img in legend_imgs:

        # Resize the legend patch
        original_norm_legend_img = cv2.resize(legend_img, (patch_size, patch_size))
        # rotate the legend patch
        rot_predictions = []
        rot_confs = []
        for rot in range(4):
            # new_norm_legend_img = np.rot90(norm_legend_img, rot)
            norm_legend_img = np.rot90(original_norm_legend_img, rot)
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
            prediction = (prediction > 0.3).astype(int)
            confs = confs_image.transpose(1,2,0).squeeze()
            rot_predictions.append(prediction)
            rot_confs.append(confs)
        rot_predictions = np.stack(rot_predictions, axis=0)
        rot_confs = np.stack(rot_confs, axis=0)
        prediction, confs = post_process_to_1(rot_predictions, rot_confs, threshold=5, soft=0)
        all_predictions.append(prediction)
        all_confs.append(confs)
    return all_predictions, all_confs

def post_process(prediction, confs, threshold=5, soft=0):
    final_prediction = np.zeros_like(prediction)
    for b in range(prediction.shape[0]):
        points_coord = np.where(prediction[b] == 1)
        for i in range(len(points_coord[0])):
            y, x = points_coord[0][i], points_coord[1][i]
            if y < threshold or x < threshold or y > prediction.shape[1]-threshold or x > prediction.shape[2]-threshold:
                continue
            # search other layers to see if there has points larger in the same position
            local_max = confs[:,y-threshold:y+threshold,x-threshold:x+threshold].max()
            # print(local_max)
            if confs[b,y,x]+soft>=local_max:
                final_prediction[b,y,x] = 1
    return final_prediction
                

def post_process_to_1(prediction, confs, threshold=5, soft=0):
    final_prediction = np.zeros_like(prediction[0])
    final_confs = np.zeros_like(confs[0])
    for b in range(prediction.shape[0]):
        points_coord = np.where(prediction[b] == 1)
        for i in range(len(points_coord[0])):
            y, x = points_coord[0][i], points_coord[1][i]
            if y < threshold or x < threshold or y > prediction.shape[1]-threshold or x > prediction.shape[2]-threshold:
                continue
            # search other layers to see if there has points larger in the same position
            local_max = confs[:,y-threshold:y+threshold,x-threshold:x+threshold].max()
            current_max = final_confs[y-threshold:y+threshold,x-threshold:x+threshold].max()
            # print(local_max)
            if confs[b,y,x]>=local_max and confs[b,y,x]>current_max:
                final_prediction[y,x] = 1
                final_confs[y,x] = confs[b,y,x]
    return final_prediction, final_confs
      
def run(working_dir, model, tifFile, legends, args):
    True_Folder = "/u/jiahuad2/data/validation_rasters"
    predicts_all, confs_all = [], []
    new_legends = []
    for legend in legends:
        write_filename = tifFile.split('.')[0]+'_'+legend+'.tif'
        if write_filename not in os.listdir(True_Folder):
            print(f"No GT for {write_filename}!", legend)
        else:
            new_legends.append(legend)
    legends = new_legends
    if len(legends) == 0:
        return -1, -1
    legend_imgs =  []
    for legend in legends:
        map_img = cv2.imread(os.path.join(working_dir, tifFile))
        map_img = cv2.cvtColor(map_img, cv2.COLOR_BGR2RGB)
        legend_path  =os.path.join(working_dir, tifFile.split('.')[0]+'_legend', legend+'.png')
        legend_img = cv2.imread(legend_path)
        legend_img = cv2.cvtColor(legend_img, cv2.COLOR_BGR2RGB)
        legend_imgs.append(legend_img)
    patched_predicted, confs = inference(model, map_img, legend_imgs, batch_size=256, patch_size=256, patch_overlap=32)
    confs_all += (confs)
    predicts_all += (patched_predicted)
    
    predicts_all = np.stack(predicts_all, axis=0)
    confs_all = np.stack(confs_all, axis=0)
    predicts_all = post_process(predicts_all, confs_all, threshold=5, soft=0.05)
    scores, files = [], []
    for patched_predicted, legend in zip(predicts_all, legends):
        write_filename = tifFile.split('.')[0]+'_'+legend+'.tif'
        # print(write_filename)
        write_filePath = os.path.join(working_dir, 'Inference', write_filename)
        masked_img = map_mask(working_dir, tifFile, patched_predicted)

        # expand one more dimension and repeat the pixel value in the third axis
        final_seg = np.repeat(masked_img[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        cv2.imwrite(write_filePath, final_seg)  
        # write_geoTIFF(filePath,write_geo_filePath,final_seg[:,:,0]) 
        
        precision, recall, f_score = single_eval(write_filename)

        print(f"Finished {write_filename}!", legend)
        scores.append((precision, recall, f_score))
        files.append(write_filename)
    return scores, files
        
        
        
    
if __name__ == "__main__":
    train()
    
    # model = build_model("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/config/fct.yaml")

        
        