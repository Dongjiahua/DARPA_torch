from .FCT import build_model
import lightning.pytorch as pl
import torchmetrics
import torch 
import os 
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import EventStorage
from utils.metrics import KeypointAPMetrics,Keypoint,DetectedKeypoint
from utils.heatmap import get_keypoints_from_heatmap_batch_maxpool
from typing import Callable, Dict, List, Tuple

class DARPA_DET(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.model = build_model("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/config/fct.yaml")
        self.loss_total = 0
        self.count = 0
        self.val_loss_total = 0
        self.val_count = 0
        # self.model = MAP_UNet(n_channels=6,n_classes=1,pretrained=args.pretrained,freeze=args.freeze)
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.train_f1 = torchmetrics.F1Score(task = "binary")
        self.val_f1 = torchmetrics.F1Score(task = "binary")
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.ap_metrics = KeypointAPMetrics()
        self.val_ap_metrics = KeypointAPMetrics()
        self.max_keypoints = 20
        self.minimal_keypoint_pixel_distance = 1
        
    def training_step(self, batch, batch_idx):
        
        map, legend, seg, instance = batch['map_img'], batch['legend_img'], batch['seg_img'], batch['instance']
        
        gt_keypoints = batch["keypoints"]
        seg = seg.unsqueeze(1)
        with EventStorage() as storage:
            output = self.model(map, legend,instance)
        pred = torch.sigmoid(output)
        self.update_channel_ap_metrics(pred[:,0,...], gt_keypoints, self.ap_metrics)
        # loss = sum(losses.values())

        # print(loss.item())
        
        loss = self.criterion(output, seg)
        self.loss_total += loss.item()
        self.count += 1
        # acc = self.train_acc(output, seg)
        # f1 = self.train_f1(output, seg)
        # self.log("train_f1",f1)
        # self.log("train_loss", loss)
        # self.log("acc", acc)
        return loss
    
    def on_train_epoch_end(self) -> None:
        # self.log('train_acc_epoch', self.train_acc.compute())
        # self.log("train_f1_epoch",self.train_f1.compute())
        # print(f"train_acc_epoch: {self.train_acc.compute()}, train_f1_epoch: {self.train_f1.compute()}")
        # self.train_acc.reset()
        # self.train_f1.reset()
        # print(f"train_loss_epoch: {self.loss_total/self.count}")
        print(f"train_ap_epoch: {self.ap_metrics.compute()}")
        self.ap_metrics.reset()
        self.loss_total = 0
        self.count = 0
    
    def on_validation_epoch_end(self) -> None:
        print(f"val_ap_epoch: {self.val_ap_metrics.compute()}")
        self.val_ap_metrics.reset()
        # print(f"val_loss_epoch: {self.val_loss_total/self.val_count}")
        # self.val_loss_total = 0
        # self.val_count = 0
        # self.log('val_acc_epoch', self.val_acc.compute())
        # self.log("val_f1_epoch",self.val_f1.compute())
        # print(f"val_acc_epoch: {self.val_acc.compute()}, val_f1_epoch: {self.val_f1.compute()}")
        # self.val_acc.reset()
        # self.val_f1.reset()
        
    def validation_step(self, batch, batch_idx):
        map, legend, seg, instance = batch['map_img'], batch['legend_img'], batch['seg_img'], batch['instance']
        

        output = self.model(map, legend,instance)
        seg = seg.unsqueeze(1)
        pred = torch.sigmoid(output)
        gt_keypoints = batch["keypoints"]
        self.update_channel_ap_metrics(pred[:,0,...], gt_keypoints, self.val_ap_metrics)
        if self.args.out_dir!="":
            self.visualize_pred(batch,output,batch_idx)
        # loss = sum(losses.values())

        # print(loss.item())
        
        loss = self.criterion(output, seg)
        self.val_loss_total += loss.item()
        self.val_count += 1
        # acc = self.val_acc(output, seg)
        # f1 = self.val_f1(output, seg)
        # self.log("train_f1",f1)
        # self.log("train_loss", loss)
        # self.log("acc", acc)
        return loss

    @torch.no_grad()
    def visualize_pred(self, batch, output, batch_idx):
        import numpy as np
        import matplotlib.pyplot as plt
        mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
        std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
        map, legend, seg = batch['map_img'][0], batch['legend_img'][0], batch['seg_img'][0]
    
        map, legend, seg = map.permute(1,2,0).cpu().detach().numpy(), legend.permute(1,2,0).cpu().detach().numpy(), seg.cpu().detach().numpy().squeeze()
        map = map*std+mean
        legend = legend*std+mean
        map[map>1] = 1
        legend[legend>1] = 1
        map[map<0] = 0
        legend[legend<0] = 0
        pred = torch.sigmoid(output[0]).permute(1,2,0).squeeze().cpu().detach().numpy()
        
        # import matplotlib.pyplot as plt
        f, axarr = plt.subplots(2,2)
        axarr[0,0].imshow(np.array(map))
        axarr[0,1].imshow(np.array(legend))
        axarr[1,0].imshow(np.array(seg),cmap="gray")
        axarr[1,1].imshow(np.array(pred),cmap="gray")
        
        # axarr[2].imshow(seg_img,cmap="gray")        
        # plt.show()
        f.savefig(os.path.join(self.args.out_dir,f"epoch_{self.current_epoch}_step_{batch_idx}.png"))
        plt.close(f)
        
    def extract_detected_keypoints_from_heatmap(self, heatmap: torch.Tensor) -> List[DetectedKeypoint]:
        """
        Extract keypoints from a single channel prediction and format them for AP calculation.

        Args:
        heatmap (torch.Tensor) : H x W tensor that represents a heatmap.
        """
        if heatmap.dtype == torch.float16:
            # Maxpool_2d not implemented for FP16 apparently
            heatmap_to_extract_from = heatmap.float()
        else:
            heatmap_to_extract_from = heatmap

        keypoints, scores = get_keypoints_from_heatmap_batch_maxpool(
            heatmap_to_extract_from, self.max_keypoints, self.minimal_keypoint_pixel_distance, return_scores=True
        )
        detected_keypoints = [
            [[] for _ in range(heatmap_to_extract_from.shape[1])] for _ in range(heatmap_to_extract_from.shape[0])
        ]
        for batch_idx in range(len(detected_keypoints)):
            for channel_idx in range(len(detected_keypoints[batch_idx])):
                for kp_idx in range(len(keypoints[batch_idx][channel_idx])):
                    detected_keypoints[batch_idx][channel_idx].append(
                        DetectedKeypoint(
                            keypoints[batch_idx][channel_idx][kp_idx][0],
                            keypoints[batch_idx][channel_idx][kp_idx][1],
                            scores[batch_idx][channel_idx][kp_idx],
                        )
                    )

        return detected_keypoints
    
    def update_channel_ap_metrics(
        self, predicted_heatmaps: torch.Tensor, gt_keypoints: List[torch.Tensor], validation_metric: KeypointAPMetrics
    ):
        """
        Updates the AP metric for a batch of heatmaps and keypoins of a single channel (!)
        This is done by extracting the detected keypoints for each heatmap and combining them with the gt keypoints for the same frame, so that
        the confusion matrix can be determined together with the distance thresholds.

        predicted_heatmaps: N x H x W tensor with the batch of predicted heatmaps for a single channel
        gt_keypoints: List of size N, containing K_i x 2 tensors with the ground truth keypoints for the channel of that sample
        """

        # log corner keypoints to AP metrics for all images in this batch
        formatted_gt_keypoints = [
            [Keypoint(int(k[0]), int(k[1])) for k in frame_gt_keypoints] for frame_gt_keypoints in gt_keypoints
        ]
        batch_detected_channel_keypoints = self.extract_detected_keypoints_from_heatmap(
            predicted_heatmaps.unsqueeze(1)
        )
        batch_detected_channel_keypoints = [batch_detected_channel_keypoints[i][0] for i in range(len(gt_keypoints))]
        for i, detected_channel_keypoints in enumerate(batch_detected_channel_keypoints):
            validation_metric.update(detected_channel_keypoints, formatted_gt_keypoints[i])


        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer