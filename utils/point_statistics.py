import os
import cv2

import glob
import json

import pandas as pd
import numpy as np

# LegendFilePath = '/home/shared/DARPA/all_patched_data/training/small_point/legend/'
filePath = '/projects/bbym/shared/all_patched_data/validation/'

LegendFiles = glob.glob(filePath+'*.png')
jsonFiles = glob.glob(filePath+'*.json')

labelfilePath = '/home/shared/DARPA/data/validation_rasters/'

# create a dataFrame to summarize the points statistics information
labelsDict = {}
for i, jsonFile in enumerate(jsonFiles):
    filename = jsonFile.split('/')[-1]
    with open(jsonFile, 'r') as f:
        jsonData = json.load(f)
    temp_list = [x['label'] for x in jsonData['shapes'] if x['label'].endswith('_pt')]
    if temp_list:
        name_number_list = []
        for point_name in temp_list:
            
            labelFileName = os.path.join(labelfilePath, filename.split('.')[0] + '_' + point_name + '.tif')
            try:
                segment_label = cv2.imread(labelFileName)
                total_points = sum(sum(segment_label[:,:,2]))
                name_number_list.append((point_name,total_points))
            except:
                print("file not found: ", labelFileName)
                
        if name_number_list:
            labelsDict[filename] = name_number_list
            
print(labelsDict)