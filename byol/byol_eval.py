import os
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
# Assuming the rest of your definitions (dataset, model, etc.) are as above
# test model, a resnet 50
import argparse
from byol import BYOL
BATCH_SIZE = 16
EPOCHS     = 100
LR         = 3e-4
NUM_GPUS   = 2
IMAGE_SIZE = 64
IMAGE_EXTS = ['.jpg', '.png', '.jpeg']
NUM_WORKERS = 8
resnet = models.resnet18(pretrained=True)

# arguments

parser = argparse.ArgumentParser(description='byol-lightning-test')

parser.add_argument('--image_folder', type=str, required = True,
                       help='path to your folder of images for self-supervised learning')

args = parser.parse_args()
class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, net, **kwargs):
        super().__init__()
        self.learner = BYOL(net, **kwargs)

    def forward(self, images):
        return self.learner(images,  return_embedding = True,
        return_projection = False)

    def training_step(self, images, _):
        loss = self.forward(images)
        self.log("loss", loss)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()

# images dataset

def expand_greyscale(t):
    return t.expand(3, -1, -1)

class ImagesDataset(Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.paths = []

        for path in Path(f'{folder}').glob('**/*'):
            _, ext = os.path.splitext(path)
            if ext.lower() in IMAGE_EXTS:
                self.paths.append(path)

        print(f'{len(self.paths)} images found')

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(expand_greyscale)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        index = index % len(self.paths)
        path = self.paths[index]
        img = Image.open(path)
        img = img.convert('RGB')
        image_name = str(path).split('/')[-1].split('.')[0]
        return self.transform(img), str(image_name)
# Step 1: Load the trained model
model_checkpoint = '/u/jiahuad2/codebase/yolo_inference/byol/lightning_logs/version_2/checkpoints/epoch=33-step=10098.ckpt'
trained_model = SelfSupervisedLearner.load_from_checkpoint(model_checkpoint, net=resnet, image_size = IMAGE_SIZE)
trained_model.eval()
trained_model.freeze()

# Make sure to move your model to the appropriate device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
trained_model.to(device)

# Step 2: Feature Extraction
# Assuming your dataset `ImagesDataset` is defined as above
dataset = ImagesDataset(args.image_folder, IMAGE_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

def extract_features(dataloader, model):
    features = []
    model.eval()
    names = []
    with torch.no_grad():
        for imgs, paths in dataloader:
            imgs = imgs.to(device)
            feats = model(imgs)
            # print(feats)
            features.append(feats.cpu().numpy())
            names.extend(paths)
    return np.concatenate(features, axis=0), names

features, names = extract_features(dataloader, trained_model)

# Step 3: PCA Transformation to 2D
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)

# Step 4: Plotting
plt.figure(figsize=(30, 30))
for i, (x, y) in enumerate(pca_result):
    plt.scatter(x, y, alpha=0.5)
    plt.text(x + 0.05, y + 0.05, os.path.basename(names[i]), fontsize=9)
plt.title('PCA Features Plot')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.savefig('pca_features.png')
