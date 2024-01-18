from metrics import feature_f_score
import os 
import cv2 
import numpy as np 
import csv 
import time 
import sys
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
    
def eval_f1():
    Pred_Folder = "/media/jiahua/FILE/uiuc/NCSA/all_patched/OK_250K_inference"
    True_Folder = "/media/jiahua/FILE/uiuc/NCSA/all_patched/validation_rasters"
    True_FileList = os.listdir(True_Folder)
    Pred_FileList = os.listdir(Pred_Folder)
    total_f1 = 0
    count = 0
    polyScore = {}
    for pt_file_name in True_FileList:
        if '_pt.tif' in pt_file_name :
            if pt_file_name not in Pred_FileList:
                print(f"File {pt_file_name} not found in prediction folder")
                continue
            trueSegPath = os.path.join(True_Folder, pt_file_name)
            predicted_path = os.path.join(Pred_Folder, pt_file_name)
            mapName = '_'.join(pt_file_name.split('_')[0:-2])+'.tif'
            mapPath = os.path.join('/media/jiahua/FILE/uiuc/NCSA/all_patched/validation_shirui', mapName)
            
            truSeg_im = cv2.imread(trueSegPath)[...,0]
            
            predicted_seg_im = cv2.imread(predicted_path)[...,0]

            map_im = cv2.imread(mapPath)
            precision, recall, f_score = feature_f_score(map_im, predicted_seg_im, truSeg_im,feature_type="pt")
            polyScore[pt_file_name] = (precision, recall, f_score)
            total_f1+=f_score
            count+=1
            print(f"file {pt_file_name} f1 score: {f_score}")
            print(f"Count: {count}, average f1 score: {total_f1/count}")
        
    prefix = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    csv_file =  f"{prefix}_pt.csv"
    os.makedirs("./exp/csv",exist_ok=True)
    with open(os.path.join("./exp/csv", csv_file), 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in polyScore.items():
            writer.writerow([key, value])
            
def single_eval(pt_file_name):
    blockPrint()
    Pred_Folder = "/u/jiahuad2/data/validation_shirui/Inference"
    True_Folder = "/u/jiahuad2/data/validation_rasters"
    True_FileList = os.listdir(True_Folder)
    total_f1 = 0
    count = 0
    if '_pt.tif' in pt_file_name:
        trueSegPath = os.path.join(True_Folder, pt_file_name)
        predicted_path = os.path.join(Pred_Folder, pt_file_name)
        mapName = '_'.join(pt_file_name.split('_')[0:-2])+'.tif'
        mapPath = os.path.join('/u/jiahuad2/data/validation_shirui', mapName)
        
        truSeg_im = cv2.imread(trueSegPath)[...,0]
        
        predicted_seg_im = cv2.imread(predicted_path)[...,0]

        map_im = cv2.imread(mapPath)
        precision, recall, f_score = feature_f_score(map_im, predicted_seg_im, truSeg_im,feature_type="pt")
        enablePrint()
        return precision, recall, f_score
    else:
        raise ValueError("Not a pt file")
        polyScore[pt_file_name] = (precision, recall, f_score)
        total_f1+=f_score
        count+=1
        print(f"file {pt_file_name} f1 score: {f_score}")
        print(f"Count: {count}, average f1 score: {total_f1/count}")
        
    prefix = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    csv_file =  f"{prefix}_pt.csv"
    os.makedirs("./exp/csv",exist_ok=True)
    with open(os.path.join("./exp/csv", csv_file), 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in polyScore.items():
            writer.writerow([key, value])

if __name__=="__main__":
    eval_f1()
    # single_eval("VA_Lahore_250K_pt.tif")