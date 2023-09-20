from model.backbones.unet import MAP_UNet
import lightning.pytorch as pl
import torchmetrics
import torch 
import os 

class DARPA_SEG(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args

        self.model = MAP_UNet(args)
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary")
        self.train_f1 = torchmetrics.F1Score(task = "binary")
        self.val_f1 = torchmetrics.F1Score(task = "binary")
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
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
        
    def training_step(self, batch, batch_idx):
        map, legend, seg = batch['map_img'], batch['legend_img'], batch['seg_img']
        model_input = torch.cat([map,legend],dim=1)
        output = self.model(model_input)
        
        output = torch.nn.functional.interpolate(output,size=seg.shape[-2:],mode="nearest")
        
        loss = self.criterion(output, seg)
        acc = self.train_acc(output, seg)
        f1 = self.train_f1(output, seg)
        self.log("train_f1",f1)
        self.log("train_loss", loss)
        self.log("acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        map, legend, seg = batch['map_img'], batch['legend_img'], batch['seg_img']
        model_input = torch.cat([map,legend],dim=1)
        output = self.model(model_input)
        
        output = torch.nn.functional.interpolate(output,size=seg.shape[-2:],mode="nearest")
        if self.args.out_dir!="":
            self.visualize_pred(batch,output,batch_idx)
        # image = torch.sigmoid(output[0]).squeeze().cpu().detach().numpy().permute(1,2,0)
        # import matplotlib.pyplot as plt
        # plt.imshow(image,cmap="gray")
        # plt.show()
        loss = self.criterion(output, seg)
        acc = self.val_acc(output, seg)
        f1 = self.val_f1(output, seg)
        self.log("val_f1",f1)
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        


    def on_train_epoch_end(self) -> None:
        self.log('train_acc_epoch', self.train_acc.compute())
        self.log("train_f1_epoch",self.train_f1.compute())
        print(f"train_acc_epoch: {self.train_acc.compute()}, train_f1_epoch: {self.train_f1.compute()}")
        self.train_acc.reset()
        self.train_f1.reset()
    
    def on_validation_epoch_end(self) -> None:
        self.log('val_acc_epoch', self.val_acc.compute())
        self.log("val_f1_epoch",self.val_f1.compute())
        print(f"val_acc_epoch: {self.val_acc.compute()}, val_f1_epoch: {self.val_f1.compute()}")
        self.val_acc.reset()
        self.val_f1.reset()
        
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer