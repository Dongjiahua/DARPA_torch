from .backbones import model_dict
import lightning.pytorch as pl
import torchmetrics
import torch 
import os 
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import EventStorage
from utils.metrics import KeypointAPMetrics,Keypoint,DetectedKeypoint
from utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from typing import Callable, Dict, List, Tuple
from utils.vis import visualize_pred_kpts,  visualize_pred
from utils.extract import update_channel_ap_metrics

class DARPA_DET(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.model = model_dict[args.model](args)

        self.loss_total = 0
        self.count = 0
        self.val_loss_total = 0
        self.val_count = 0
        # self.model = MAP_UNet(n_channels=6,n_classes=1,pretrained=args.pretrained,freeze=args.freeze)
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.train_f1 = torchmetrics.F1Score(task = "binary")
        self.val_f1 = torchmetrics.F1Score(task = "binary")
        self.criterion = torch.nn.BCELoss()
        self.ap_metrics = KeypointAPMetrics()
        self.val_ap_metrics = KeypointAPMetrics()
        self.max_keypoints = 20
        self.minimal_keypoint_pixel_distance = 5
        
    def training_step(self, batch, batch_idx):
        
        map, legend, seg, instance = batch['map_img'], batch['legend_img'], batch['seg_img'], batch['instance']
        
        gt_keypoints = batch["keypoints"]
        seg = seg.unsqueeze(1)
        with EventStorage() as storage:
            output = self.model(map, legend,instance)
        update_channel_ap_metrics(output[:,0,...], gt_keypoints, self.ap_metrics)
        loss = self.criterion(output, seg)
        return loss
    
    def on_train_epoch_end(self) -> None:

        print(f"train_ap_epoch: {self.ap_metrics.compute()}")
        self.ap_metrics.reset()
        self.loss_total = 0
        self.count = 0
    
    def on_validation_epoch_end(self) -> None:
        print(f"val_ap_epoch: {self.val_ap_metrics.compute()}")
        self.val_ap_metrics.reset()

        
    def validation_step(self, batch, batch_idx):
        map, legend, seg, instance = batch['map_img'], batch['legend_img'], batch['seg_img'], batch['instance']
        

        output = self.model(map, legend,instance)
        seg = seg.unsqueeze(1)
        # pred = torch.sigmoid(output)
        gt_keypoints = batch["keypoints"]
        update_channel_ap_metrics(output[:,0,...], gt_keypoints, self.val_ap_metrics)
        if self.args.out_dir!="":
            visualize_pred(batch, output, batch_idx, self.current_epoch, self.args.out_dir)
        
        loss = self.criterion(output, seg)

        return loss

    
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer