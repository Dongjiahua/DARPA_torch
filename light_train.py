from typing import Any
import lightning.pytorch as pl
import argparse
import torch 
from model.unet import UNet
from data.dataset import MAPData
from torch.utils.data import DataLoader
import torchmetrics
import os 
from tqdm import tqdm

class Trainer(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args