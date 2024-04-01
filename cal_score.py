import pandas as pd
import numpy as np

# read a 3 column csv file

file_path = "/u/jiahuad2/codebase/yolo_inference/exp/csv/20240208_155904_pt.csv"
df = pd.read_csv(file_path, header=None)
map_names = df[0].values[:-1]
map_scores = df[1].values[:-1]
map_scores = np.array([eval(x) for x in map_scores]) 

sub_set_name = ["1_pt.tif","2_pt.tif", "3_pt.tif", "4_pt.tif", "5_pt.tif"]
subset_scores = []
subset_scores = [map_scores[i] for i in range(len(map_scores)) if any(sub_set_name in map_names[i] for sub_set_name in sub_set_name)]
subset_scores = np.array(subset_scores)
print(f"from {len(map_scores)} to {len(subset_scores)}")
print(f"median precision: {np.median(subset_scores[:,0])}, average precision: {np.mean(subset_scores[:,0])}")
print(f"median recall: {np.median(subset_scores[:,1])}, average recall: {np.mean(subset_scores[:,1])}")
print(f"median f1: {np.median(subset_scores[:,2])}, average f1: {np.mean(subset_scores[:,2])}")