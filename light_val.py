from typing import Any
import lightning.pytorch as pl
import argparse
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch 
from model import  DARPA_DET
from data.dataset import MAPData
from data.dataset_det import DetData, collect_fn_det
from data.test_data import TestData
from torch.utils.data import DataLoader
import torchmetrics
import os 
from tqdm import tqdm
from detectron2.engine import DefaultTrainer as d2Trainer
from glob import glob
import json 
import numpy as np
import patchify
import cv2 
import imageio
from itertools import chain
from patchify import patchify, unpatchify
from metric_eval import eval_f1, single_eval
import  tracemalloc
import gc
import math 
import time 
import csv
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str,default="/media/jiahua/FILE/uiuc/NCSA/all_patched/all_patched_data/training", help='Root train data path')
    parser.add_argument('--val_data', type=str, default="/media/jiahua/FILE/uiuc/NCSA/all_patched/all_patched_data/validation",  help='Root val data path')
    parser.add_argument('--fct_cfg', type=str, default="/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/config/fct.yaml", help='fct config')
    parser.add_argument('--out_dir', type=str, default="output_all", help='output_dir')
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

    
def train():
    args = parse_args()
    torch.manual_seed(0)

    if args.out_dir!="":
        os.makedirs(args.out_dir,exist_ok=True)

    working_dir = "/media/jiahua/FILE/uiuc/NCSA/all_patched/validation_shirui"
    
    model = DARPA_DET.load_from_checkpoint("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/exp/lightning_logs/version_436/checkpoints/epoch=13-step=5488.ckpt",args=args)
    model = model.cuda()
    model.eval()
    
    
    tifPaths = glob(working_dir+'/*.tif')
    sorted(tifPaths)

    if not os.path.exists(os.path.join(working_dir, 'Inference')):
        os.mkdir(os.path.join(working_dir, 'Inference'))
    count = 0
    polyScore = {}
    f1s = []
    for tifPath in tifPaths:    
        tifFile = tifPath.split('/')[-1]
        jsonFile = tifFile.split('.')[0]+'.json'

        with open(os.path.join(working_dir, jsonFile), 'r') as f:
            jsonData = json.load(f)

        for label_dir in jsonData['shapes']:
            legend = label_dir['label']
            if legend.endswith('_pt'):
                
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

def run(working_dir, model, tifFile, legend, args):
    True_Folder = "/media/jiahua/FILE/uiuc/NCSA/all_patched/validation_rasters"
    write_filename = tifFile.split('.')[0]+'_'+legend+'.tif'
    if write_filename not in os.listdir(True_Folder):
        print(f"No GT for {write_filename}!", legend)
        return -1,-1
    write_filePath = os.path.join(working_dir, 'Inference', write_filename)
    # if os.path.isfile(write_filePath):
    #     print(f"Already Finished {write_filename}!", legend)
    #     return
    try:
        legend_coor, shift_coord, map_im_cut_dims, map_patchs_dims = build_patch(working_dir, tifFile, legend, patch_dims = (256,256))
    except:
        print(f"Failed {write_filename}!", legend)
        return
    gc.collect()
    patchNames = sorted(glob(os.path.join(working_dir, tifFile.split('.')[0]+'_map_patches/*')))
    legend_path  =os.path.join(working_dir, tifFile.split('.')[0]+'_legend', legend+'.png')
    val_dataset = TestData(patchNames, legend_path, args)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False,num_workers=8, collate_fn=collect_fn_det)
    patched_predicted = np.zeros((map_patchs_dims[0], map_patchs_dims[1], 1, 256, 256, 1))
    index = 0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            output, keypoints = model(data)
            for kpts_patches in keypoints:
                kpts  = kpts_patches[0]
                dim_i = index//map_patchs_dims[1]
                
                dim_j = index%map_patchs_dims[1]
                for kpt in kpts:
                    patched_predicted[dim_i, dim_j, 0, int(kpt[1]), int(kpt[0]), 0] = 1
                index+=1
                

    patched_predicted = unpatchify(patched_predicted, (map_im_cut_dims[0], map_im_cut_dims[1], 1))
    patched_predicted = np.pad(patched_predicted, [(shift_coord[0], shift_coord[1]), (shift_coord[2], shift_coord[3]), (0,0)], mode='constant')
    gc.collect()
    patched_predicted = patched_predicted.astype(int)
    masked_img = map_mask(working_dir, tifFile, patched_predicted)
    del patched_predicted, val_dataset, val_loader
    gc.collect()
    # expand one more dimension and repeat the pixel value in the third axis
    final_seg = np.repeat(masked_img[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

    cv2.imwrite(write_filePath, final_seg)    
    print(f"Finished {write_filename}!", legend)
    precision, recall, f_score = single_eval(write_filename)
    return (precision, recall, f_score), write_filename
        
        
        
    
if __name__ == "__main__":
    train()
    
    # model = build_model("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/config/fct.yaml")

        
        