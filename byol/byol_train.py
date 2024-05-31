import os
import argparse
import multiprocessing
from pathlib import Path
from PIL import Image

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset

from byol import BYOL, SimCLR
import pytorch_lightning as pl
import json 
from h5image import H5Image
import glob
import random
import numpy as np
# test model, a resnet 50

def h5setup(path):
    mapsh5i = {}
    # exclude_map = set(['CO_SanchezRes','CA_NV_LasVegas','CO_VailE100K','CA_Cambria','ID_LakeWalcott','AZ_PioRico_Nogales'])
    for file in glob.glob(os.path.join(path, "*")):
        mapname = os.path.basename(file).replace(".hdf5", "")
        mapsh5i[mapname] = H5Image(file, "r")
    print("Setup complete. Number of maps loaded:", len(mapsh5i))
    return mapsh5i

resnet = models.resnet18(pretrained=True)

# arguments

parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--image_folder', type=str, default="/u/jiahuad2/codebase/yolo_inference/config/legend_name2class.json",
                       help='path to your folder of images for self-supervised learning')

args = parser.parse_args()

# constants
MODEL = "SimCLR"
BATCH_SIZE = 128
EPOCHS     = 1000
LR         = 3e-4
NUM_GPUS   = 2
IMAGE_SIZE = 64
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = multiprocessing.cpu_count()

# pytorch lightning module

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        if MODEL == "SimCLR":
            self.learner = SimCLR(net, **kwargs)
        elif MODEL == "BYOL":
            self.learner = BYOL(net, **kwargs)

    def forward(self, images, labels):
        return self.learner(images, labels)

    def training_step(self, data, _):
        images, labels = data
        loss = self.forward(images, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, data, _):
        images, labels = data
        loss = self.forward(images, labels)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

    def on_epoch_end(self):
        avg_loss = self.trainer.callback_metrics['loss']
        print(f"Avg Loss: {avg_loss}")
# images dataset

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, json_path, image_size, split = 'train'):
        super().__init__()
        with open(json_path) as f:
            data = json.load(f)
        self.cache_paths = []
        self.cache_labels = []
        self.cache_images = []
        self.transform = transforms.Compose([
            
            transforms.Resize(image_size),
            # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.3),
            transforms.ToTensor()
        ])
        for i, instance in enumerate(data):
            self.cache_paths.append(os.path.join("/u/jiahuad2/codebase/yolo_inference",instance["example"]))
            self.cache_labels.append(instance['id'])
            img = Image.open(self.cache_paths[-1])
            img = img.convert('RGB')
            img = self.transform(img)
            self.cache_images.append(img)
            
        self.name_to_class = {name: int(data[i]["id"]) for i in range(len(data)) for name in data[i]["names"]}

        h5_data_path = "/u/jiahuad2/codebase/yolo_inference/yolo_data/h5_15"
        self.h5_path = os.path.join(h5_data_path, split)
        self.h5_files = h5setup(self.h5_path)
        self.files = self.build_files(self.h5_files)
        self.split = split
        
    def build_files(self, h5_files):
        
        maps = list(h5_files.keys())
        valids = []
        raw = []
        for map_name in maps:
            try:
                patches = h5_files[map_name].get_patches(map_name)
            except:
                print(f"Map {map_name} has no patches")
                continue
            try:
                for legend in [layer for layer in patches if layer.endswith('_pt') ]:
                    legend = h5_files[map_name].get_legend(map_name, legend)
            except:
                print(f"Map {map_name} has no legend: {legend}")
                continue
            try:
                cor = h5_files[map_name].get_map_corners(map_name)
                all_patches = [(map_name, layer, x, y) for layer in h5_files[map_name].get_layers(map_name) if layer.endswith('pt') for x in range(cor[0][0], cor[1][0]) for y in range(cor[0][1], cor[1][1])]
                raw.extend(all_patches)
            except:
                print(f"Map {map_name} has no corners")
           
            valid_patches = [(map_name, layer, x, y) for layer in patches if layer.endswith('_pt') for x, y in patches[layer]]
            valids.extend(valid_patches)
            self.len_valids = len(valids)
        
        return valids
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        index = index % len(self.files)
        # for 10% probability, select a image from the cache
        if random.random() < 0 and self.split == 'train':
            return self.cache_images[index], self.cache_labels[index]

        map_name, legend, row, col = self.files[index]
        label_patch = self.h5_files[map_name].get_patch(row, col, map_name, legend)
        point_coords = np.where(label_patch==1)
        point_coords = np.stack(point_coords, axis=1)
        # randomly select a point from the point_coords
        y, x = point_coords[np.random.choice(point_coords.shape[0])]
        im = self.h5_files[map_name].get_patch(row, col,  map_name)
        # pad the image with white images and crop a patch from given x,y with IMAGE_SIZE
        im = np.pad(im, ((IMAGE_SIZE//2, IMAGE_SIZE//2), (IMAGE_SIZE//2, IMAGE_SIZE//2), (0, 0)), constant_values=255)
        im = im[x:x+IMAGE_SIZE, y:y+IMAGE_SIZE]
        im = Image.fromarray(im)
        im = im.convert('RGB')
        im = self.transform(im)
        label = self.name_to_class[legend]
        return im, label
              
# main

if __name__ == '__main__':
    
    train_ds = ImagesDataset(args.image_folder, IMAGE_SIZE, "train")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    val_ds = ImagesDataset(args.image_folder, IMAGE_SIZE, "val")
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)
    model = SelfSupervisedLearner(
        resnet,
        image_size = IMAGE_SIZE,
        hidden_layer = 'avgpool',
        projection_size = 256,
        projection_hidden_size = 4096,
        moving_average_decay = 0.99
    )

    trainer = pl.Trainer(
        accelerator="gpu", devices=1,
        max_epochs = EPOCHS,
        accumulate_grad_batches = 1,
        sync_batchnorm = True
    )

    trainer.fit(model, train_loader, val_loader)