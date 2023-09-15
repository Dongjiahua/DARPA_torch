from .FCT import build_model
import lightning.pytorch as pl
import torchmetrics
import torch 
import os 
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import EventStorage

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
        
        
    def training_step(self, batch, batch_idx):
        
        map, legend, seg, instance = batch['map_img'], batch['legend_img'], batch['seg_img'], batch['instance']
        
        with EventStorage() as storage:
            output = self.model(map, legend,instance)
        # loss = sum(losses.values())
        # self.loss_total += loss.item()
        # self.count += 1
        # print(loss.item())
        
        loss = self.criterion(output, seg)
        acc = self.train_acc(output, seg)
        f1 = self.train_f1(output, seg)
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
        
    def validation_step(self, batch, batch_idx):
        map, legend, seg, instance = batch['map_img'], batch['legend_img'], batch['seg_img'], batch['instance']
        

        output = self.model(map, legend,instance)
        self.visualize_pred(batch,output,batch_idx)
        # loss = sum(losses.values())
        # self.val_loss_total += loss.item()
        # self.val_count += 1
        # print(loss.item())
        
        loss = self.criterion(output, seg)
        acc = self.val_acc(output, seg)
        f1 = self.val_f1(output, seg)
        # self.log("train_f1",f1)
        # self.log("train_loss", loss)
        # self.log("acc", acc)
        return loss
        


    def on_train_epoch_end(self) -> None:
        # self.log('train_acc_epoch', self.train_acc.compute())
        # self.log("train_f1_epoch",self.train_f1.compute())
        print(f"train_acc_epoch: {self.train_acc.compute()}, train_f1_epoch: {self.train_f1.compute()}")
        self.train_acc.reset()
        self.train_f1.reset()
        # print(f"train_loss_epoch: {self.loss_total/self.count}")
        # self.loss_total = 0
        # self.count = 0
    
    def on_validation_epoch_end(self) -> None:
        # print(f"val_loss_epoch: {self.val_loss_total/self.val_count}")
        # self.val_loss_total = 0
        # self.val_count = 0
        # self.log('val_acc_epoch', self.val_acc.compute())
        # self.log("val_f1_epoch",self.val_f1.compute())
        print(f"val_acc_epoch: {self.val_acc.compute()}, val_f1_epoch: {self.val_f1.compute()}")
        self.val_acc.reset()
        self.val_f1.reset()
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer