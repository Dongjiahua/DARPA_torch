import os 
import json
import glob
import yaml
eval_legend_dir = os.listdir('/u/jiahuad2/codebase/yolo_inference/yolo_data/legend_base')
train_legend_dir = os.listdir('/u/jiahuad2/codebase/yolo_inference/yolo_data/legend_base_train')

eval_legend_dir = sorted([x.split('.')[0] for x in eval_legend_dir])
train_legend_dir = sorted([x.split('.')[0] for x in train_legend_dir])

annotation_json = "/u/jiahuad2/codebase/yolo_inference/yolo_data/legend_name2class.json"
with open(annotation_json) as f:
    legend_name2class = json.load(f)
legend_names = [item["names"] for item in legend_name2class]
legend_names = [item for sublist in legend_names for item in sublist]

def check_set(legend_dir, legend_names):
    for legend in legend_dir:
        if legend not in legend_names:
            raise Exception(f"Error: {legend} not in legend_names")

print("Checking eval set")
check_set(eval_legend_dir, legend_names)
print("Checking train set")
check_set(train_legend_dir, legend_names)

cfgpath = "/u/jiahuad2/codebase/yolo_inference/config/h5_15.yaml"
with open(cfgpath) as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

names = {item["id"]:item["class"] for item in legend_name2class}
cfg['names'] = names

# write new yaml
new_path = "/u/jiahuad2/codebase/yolo_inference/config/h5_15_multiclass.yaml"
with open(new_path, 'w') as f:
    yaml.dump(cfg, f)
