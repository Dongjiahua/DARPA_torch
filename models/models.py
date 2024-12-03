# from statistics import correlation
from calendar import c
import pytorch_lightning as pl
# import segmentation_models_pytorch as smp
from sympy import fu
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from adamp import AdamP
from torchmetrics.functional import jaccard_index, f1_score
import torchvision.models as models
from .res_unet import Resnet_Unet
from torch.nn.init import kaiming_normal_, constant_
import matplotlib.pyplot as plt
import numpy as np
# from .superpixel_utils import *
import wandb 
import os 

# util function for generating interactive image mask from components
def wb_mask(bg_img, pred_mask, true_mask):
  return wandb.Image(bg_img, masks={
    "prediction" : {"mask_data" : pred_mask, "class_labels" : labels()},
    "ground truth" : {"mask_data" : true_mask, "class_labels" : labels()}})

def labels():
  l = {}
  for i, label in enumerate(['bg', 'target']):
    l[i] = label
  return l

@torch.no_grad()
def conv_feat(feat, legend):
    """
    input: torch.Tensor, shape (B, 3, H, W)
    legend: torch.Tensor, shape (B, 3, H, W)
    resize the legend to (B,3, 32, 32) and use it as convolutional layer to extract features
    """
    h, w = 33, 33
    B, C = feat.size(0), feat.size(1)
    kernel = torch.nn.functional.interpolate(legend, size=(h, w), mode='bilinear', align_corners=False)
    kernel = kernel.reshape(B*C, 1, h, w)
    feat = feat.reshape(1, B*C, feat.size(2), feat.size(3))
    out = torch.nn.functional.conv2d(feat, kernel, groups=B*C, padding=(h//2, w//2)) # (1, B, H, W)
    out = out.reshape(B, C, feat.size(2), feat.size(3)).sum(dim=1, keepdim=True)
    return out
    
@torch.no_grad()
def sub_conv_feat(feat, legend, max_div=None):
    """
    input: torch.Tensor, shape (B, 3, H, W)
    legend: torch.Tensor, shape (B, 3, H, W)
    resize the legend to (B,3, 32, 32) and use it as convolutional layer to extract features
    """
    h, w = 19, 19
    B, C, H, W = feat.shape
    kernel = torch.nn.functional.interpolate(legend, size=(h, w), mode='bilinear', align_corners=False) #  (B, 3, 33, 33)
    unfold_feat = F.unfold(feat, kernel_size=(h, w), padding=(h//2, w//2)).reshape(B, 3, h, w, H*W) # (B, 3,33, 33, H*W)
    kernel = kernel.reshape(B, C,  h, w, 1)
    sub = (unfold_feat - kernel).abs().sum(dim=(1, 2, 3), keepdim=True).reshape(B, 1, H*W)
    out = F.fold(sub, output_size=(H, W), kernel_size=(1, 1))
    out = -out
    # if max_div is not None:
    #     out = max_div + out
    # print(out.shape)


    return out

class SegmentationModel(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        model_name = args.model
        n_channels = 6
        self.superpixel_weights = args.superpixel #'/u/xiyuez2/xiyuez2/Darpa_Unet/models/SpixelNet_bsd_ckpt.tar'
        self.superpixel = (len(self.superpixel_weights) > 0)


        if args.edge:
            n_channels -= 2
        if self.superpixel:
            n_channels += 9
            self.superpixel_model = SpixelNet(batchNorm=True)
            # load ckpt to model
            self.superpixel_model.load_state_dict(torch.load(self.superpixel_weights)["state_dict"])
        else:
            self.superpixel_model = None


        if model_name == "Unet":
            self.model = UNet(n_channels=n_channels,n_classes=2) # was 6*5 when adding the sin preprocess
        elif model_name == "Unet_bound":
            self.model = UNet_Bound(n_channels=12,n_classes=2) # was 6*5 when adding the sin preprocess
        elif model_name == "Unet_S":
            self.model = UNet_S(n_channels=n_channels,n_classes=2) # was 6*5 when adding the sin preprocess
        elif model_name == "Unet_conv":
            self.model = UNet_conv(n_channels=7,n_classes=2)
        elif model_name == "UNet_S1_conv":
            self.model = UNet_S1_conv(n_channels=7,n_classes=2)
        elif model_name == "Unet_D":
            self.model = UNet_D(n_channels=3,n_classes=2) # was 6*5 when adding the sin preprocess
        elif model_name == "Unet_S1":
            self.model = UNet_S1(n_channels=n_channels,n_classes=2) # was 6*5 when adding the sin preprocess
        elif model_name == "Unet_S2":
            self.model = UNet_S2(n_channels=n_channels,n_classes=2) # was 6*5 when adding the sin preprocess
        elif model_name == "Unet_S3":
            self.model = UNet_S3(n_channels=n_channels,n_classes=2) # was 6*5 when adding the sin preprocess
        elif model_name == "Unet_S3_C":
            self.model = UNet_S3_C(n_channels=3,n_classes=2) # was 6*5 when adding the sin preprocess
        elif model_name == "Unet_S4":
            self.model = UNet_S4(n_channels=n_channels,n_classes=2) # was 6*5 when adding the sin preprocess
        elif model_name == "Unet2B":
            self.model = UNet2Branch(n_channels=3,n_classes=2)
        elif model_name == "Unet2BMP":
            self.model = UNet2Branch_MPool(n_channels=3,n_classes=2)
        elif model_name == "Resnet":
            self.model = Resnet_Unet(n_channels=n_channels,n_classes=2)
        elif model_name == "Unet_COCO":
            self.model = UNet_COCO(n_channels=3,n_classes=81)
        else:
            raise NotImplementedError
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.mean= torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        self.std= torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]

    def forward(self, x):
        x = self.model(x)
        return x

    def log_images_and_masks(self, outputs, images, legends, gt, fuse_gt, val=True, a_images=None):

        images = images * self.std.cuda() + self.mean.cuda()
        legends = legends * self.std.cuda() + self.mean.cuda()
        pred_masks = outputs.argmax(dim=1)
        masks = gt.long()
        plots = []
        if a_images is not None:
            a_images = [a * self.std.cuda() + self.mean.cuda() for a in a_images]
        # 将结果存储到wandb中
        # os.make
        for j in range(images.size(0)):
            img = images[j].cpu().numpy().transpose(1, 2, 0).squeeze()  # 转换为CHW
            legend = legends[j].cpu().numpy().transpose(1, 2, 0).squeeze()
            pred_mask = pred_masks[j].cpu().numpy().squeeze()
            true_mask = masks[j].cpu().numpy().squeeze()
            fuse_mask = fuse_gt[j].cpu().numpy().squeeze()
            a_image = [a[j].cpu().numpy().transpose(1, 2, 0).squeeze() for a in a_images] if a_images is not None else []
            # make a image by ploting them
            total_imgs = 5+len(a_image)
            fig, axes = plt.subplots(1, total_imgs)
            # print(img.max(), img.shape)
            axes[0].imshow(img)
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(legend)
            axes[1].set_title("Legend")
            axes[1].axis("off")

            axes[2].imshow(pred_mask, cmap="gray")
            axes[2].set_title("Pred")
            axes[2].axis("off")

            axes[3].imshow(true_mask, cmap="gray")
            axes[3].set_title("GT")
            axes[3].axis("off")

            axes[4].imshow(fuse_mask, cmap="gray")
            axes[4].set_title("Fuse GT")
            axes[4].axis("off")

            for i, a_img in enumerate(a_image):
                axes[5+i].imshow(a_img)
                axes[5+i].set_title(f"Boundary {i}")
                axes[5+i].axis("off")
                
            plt.tight_layout()

            os.makedirs("tmp", exist_ok=True)
            filename = f"tmp/{j}.png"
            plt.savefig(filename)
            plt.close(fig)

            plots.append(filename)
        if val:
            self.logger.log_image(key = f"val/example", images = plots)
        else:
            self.logger.log_image(key = f"train/example", images = plots)

            # break # ONLY ONE IMAGE PER BATCH


    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'adamp':
            optimizer = AdamP(self.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-2)

        if self.args.scheduler == "reducelr":
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, mode="max", verbose=True)
            return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/jaccard_index_value"}

        elif self.args.scheduler == "cosineanneal":
            scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-5,
                                                                 last_epoch=-1, verbose=True)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, train_batch, batch_idx):
        if self.superpixel:
            map_img, legend_img, mask, superpixel = train_batch["map_img"],train_batch["legend_img"],train_batch["GT"],train_batch["superpixel"]
            
            superpixel_img = super_pixel_inference(downsize = 32, model = self.superpixel_model, img_= map_img)
            
            # superpixel_img = torch.from_numpy(np.array(superpixel_img)).float().cuda()
            map_img = torch.cat((map_img, superpixel_img),axis=1)

        else:
            map_img, legend_img, mask = train_batch["map_img"],train_batch["legend_img"],train_batch["GT"]
        
        if self.args.encode_boundary:
            a_channel = train_batch["a_channel"]
            a_images = [a_channel[:,:3,:,:], a_channel[:,3:,:,:]]
        else:
            a_images = None
        fuse_mask, gt_mask = train_batch["color_seg"], train_batch["seg"]
        mask = mask.long()
        fuse_mask = fuse_mask.long()
        gt_mask = gt_mask.long()

        if a_images is not None: 
            outputs = self.model(map_img,legend_img, a_channel)
        else:
            outputs = self.model(map_img,legend_img)
        if (batch_idx+1)%2000 == 10 and self.model.n_classes == 2:
            self.log_images_and_masks(outputs, map_img, legend_img, gt_mask, fuse_mask, val=False, a_images=a_images)
        loss = self.criterion(outputs, mask)
        jaccard_index_value = jaccard_index(outputs.argmax(dim=1), gt_mask, task="multiclass", num_classes=self.model.n_classes)
        f1_score_value = f1_score(outputs.argmax(dim=1), gt_mask, task="multiclass", num_classes=self.model.n_classes)
        self.log('train/loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('train/jaccard_index_value', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)
        self.log('train/f1_score_value', f1_score_value, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('train/jaccard_index_value_fuse', jaccard_index(outputs.argmax(dim=1), fuse_mask, task="multiclass", num_classes=self.model.n_classes), on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('train/f1_score_value_fuse', f1_score(outputs.argmax(dim=1), fuse_mask, task="multiclass", num_classes=self.model.n_classes), on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "jaccard_index_value": jaccard_index_value}

    def validation_step(self, val_batch, batch_idx):
        if self.superpixel:
            map_img, legend_img, mask, superpixel = val_batch["map_img"],val_batch["legend_img"],val_batch["GT"],val_batch["superpixel"]
            
            superpixel_img = super_pixel_inference(downsize = 32, model = self.superpixel_model, img_= map_img)
            # for i in range(len(map_img)):
            #     superpixel_img.append(super_pixel_inference(downsize = 16, model = self.superpixel_model, img_= map_img[i]))    
            # superpixel_img = torch.from_numpy(np.array(superpixel_img)).float().cuda()
            map_img = torch.cat((map_img, superpixel_img), axis=1)
        else:
            map_img, legend_img, mask = val_batch["map_img"], val_batch["legend_img"], val_batch["GT"]
        fuse_mask, gt_mask = val_batch["color_seg"], val_batch["seg"]
        fuse_mask = fuse_mask.long()
        gt_mask = gt_mask.long()
        mask = mask.long()
        if self.args.encode_boundary:
            a_channel = val_batch["a_channel"]
            a_images = [a_channel[:,:3,:,:], a_channel[:,3:,:,:]]
        else:
            a_images = None
        if a_images is not None:
    
            outputs = self.model(map_img,legend_img, a_channel)
        else:
            outputs = self.model(map_img,legend_img)
        if (batch_idx+1)%2000 == 10 and self.model.n_classes == 2:
            self.log_images_and_masks(outputs, map_img, legend_img, gt_mask, fuse_mask, val=True, a_images=a_images)
        loss = self.criterion(outputs, mask)
        jaccard_index_value = jaccard_index(outputs.argmax(dim=1), gt_mask,task="multiclass", num_classes=self.model.n_classes)

        self.log('val/loss', loss, on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('val/jaccard_index_value', jaccard_index_value, on_epoch=True, on_step=True, prog_bar=True,
                 sync_dist=True)
        self.log('val/f1_score_value', f1_score(outputs.argmax(dim=1), gt_mask, task="multiclass", num_classes=self.model.n_classes), on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('val/jaccard_index_value_fuse', jaccard_index(outputs.argmax(dim=1), fuse_mask, task="multiclass", num_classes=self.model.n_classes), on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        self.log('val/f1_score_value_fuse', f1_score(outputs.argmax(dim=1), fuse_mask, task="multiclass", num_classes=self.model.n_classes), on_epoch=True, on_step=True, prog_bar=True, sync_dist=True)
        return {"loss": loss, "jaccard_index_value": jaccard_index_value}

    # on each validation epoch end, if f1 is better than previous best, save the model
    def on_validation_epoch_end(self):
        save_dir = os.path.join("coco_checkpoints", self.args.name)
        os.makedirs(save_dir, exist_ok=True)
        if self.current_epoch == 0:
            self.best_f1 = 0
        if self.best_f1 < self.trainer.callback_metrics["val/f1_score_value"]:
            self.best_f1 = self.trainer.callback_metrics["val/f1_score_value"]
            print(f"Updated Best F1: {self.best_f1}")
            torch.save(self.model.state_dict(), os.path.join(save_dir, f"{self.best_f1:.4f}.pth"))
    
    def test_step(self, val_batch, batch_idx):
        if self.superpixel:
            map_img, legend_img, mask, superpixel = val_batch["map_img"].cuda(), val_batch["legend_img"].cuda(), val_batch["GT"].cuda(), val_batch["superpixel"].cuda()
            
            superpixel_img = super_pixel_inference(downsize = 32, model = self.superpixel_model, img_= map_img)
            # for i in range(len(map_img)):
            #     superpixel_img.append(super_pixel_inference(downsize = 16, model = self.superpixel_model, img_= map_img[i]))    
            # superpixel_img = torch.from_numpy(np.array(superpixel_img)).float().cuda()
            map_img = torch.cat((map_img, superpixel_img),axis=1)
        else:
            map_img, legend_img, mask = val_batch["map_img"].cuda(), val_batch["legend_img"].cuda(), val_batch["GT"].cuda()
        
        mask = mask.long()
        if self.args.encode_boundary:
            a_channel = val_batch["a_channel"]
            a_images = [a_channel[:,:3,:,:], a_channel[:,3:,:,:]]
        else:
            a_channel = None
        if a_channel is not None:
            outputs = self.model(map_img,legend_img, a_channel)
        else:
            outputs = self.model(map_img,legend_img)
        # loss = self.criterion(outputs, mask)
        # jaccard_index_value = jaccard_index(outputs.argmax(dim=1), mask, num_classes=2)
        return outputs[:,-1,]


    # def on_validation_epoch_end(self):
    #     self.log_images_and_masks("val")


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

def predict_param(in_planes, channel=3):
    return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)

def predict_mask(in_planes, channel=9):
    return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)

def predict_feat(in_planes, channel=20, stride=1):
    return  nn.Conv2d(in_planes, channel, kernel_size=3, stride=stride, padding=1, bias=True)

def predict_prob(in_planes, channel=9):
    return  nn.Sequential(
        nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Softmax(1)
    )
#***********************************************************************

def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1)
        )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1)
    )

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SingleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.single_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.single_conv(x)
    
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, double_conv = True):
        super().__init__()
        if not double_conv:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                SingleConv(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, double_conv = True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if double_conv:
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = DoubleConv(in_channels, out_channels)
        elif not double_conv:
            if bilinear:
                self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
                self.conv = SingleConv(in_channels, out_channels, in_channels // 2)
            else:
                self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
                self.conv = SingleConv(in_channels, out_channels)
                
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Fuse_Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, fuse_channel = 0):
        super().__init__()
        if fuse_channel == 0:
            fuse_channel = in_channels
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels // 2 + fuse_channel, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class HED(nn.Module):
    def __init__(self):
        super(HED, self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.features = list(vgg.features.children())
        
        self.conv1 = nn.Sequential(*self.features[:5])
        self.conv2 = nn.Sequential(*self.features[5:10])
        self.conv3 = nn.Sequential(*self.features[10:17])
        self.conv4 = nn.Sequential(*self.features[17:24])
        self.conv5 = nn.Sequential(*self.features[24:31])
        
        self.side1 = nn.Conv2d(64, 1, 1)
        self.side2 = nn.Conv2d(128, 1, 1)
        self.side3 = nn.Conv2d(256, 1, 1)
        self.side4 = nn.Conv2d(512, 1, 1)
        self.side5 = nn.Conv2d(512, 1, 1)
        
        self.fuse = nn.Conv2d(5, 1, 1)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        
        side1 = self.side1(conv1)
        side2 = self.side2(conv2)
        side3 = self.side3(conv3)
        side4 = self.side4(conv4)
        side5 = self.side5(conv5)
        
        side1 = nn.functional.interpolate(side1, size=x.size()[2:], mode='bilinear', align_corners=True)
        side2 = nn.functional.interpolate(side2, size=x.size()[2:], mode='bilinear', align_corners=True)
        side3 = nn.functional.interpolate(side3, size=x.size()[2:], mode='bilinear', align_corners=True)
        side4 = nn.functional.interpolate(side4, size=x.size()[2:], mode='bilinear', align_corners=True)
        side5 = nn.functional.interpolate(side5, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        fuse = self.fuse(torch.cat((side1, side2, side3, side4, side5), 1))
        
        return side1, side2, side3, side4, side5, fuse
    
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, map, legend):
        # assert False
        x = torch.cat((map, legend),axis=1)
        # print(x.size(),map.size(),legend.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

class UNet_COCO(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_COCO, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, map, legend):
        
        # x = torch.cat((map, legend),axis=1)
        # print(x.size(),map.size(),legend.size())
        x1 = self.inc(map)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits

class UNet_Bound(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_Bound, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, map, legend, a_channel):
        
        x = torch.cat((map, legend, a_channel),axis=1)
        # print(x.size(),map.size(),legend.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # logits = torch.argmax(logits, dim=1)
        # print(logits.size())
        # viz_logits =  torch.argmax(logits, dim=1).cpu().numpy()
        # print(viz_logits.shape)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class UNet_S(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_S, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, map, legend):
        
        x = torch.cat((map, (map-legend).abs()),axis=1)
        # print(x.size(),map.size(),legend.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # logits = torch.argmax(logits, dim=1)
        # print(logits.size())
        # viz_logits =  torch.argmax(logits, dim=1).cpu().numpy()
        # print(viz_logits.shape)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class UNet_conv(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_conv, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, map, legend):
        x = (map - legend).abs().sum(dim=1, keepdim=True)
        # normalize x to 0-1
        x = (x - x.min()) / (x.max() - x.min()+1e-6)
        x[x>0.02] = 1
        x[x<=0.02] = 0
        x = torch.cat((map, legend, x),axis=1)
        # print(x.size(),map.size(),legend.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # logits = torch.argmax(logits, dim=1)
        # print(logits.size())
        # viz_logits =  torch.argmax(logits, dim=1).cpu().numpy()
        # print(viz_logits.shape)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        
class UNet_D(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, map, legend):

        x = (map - legend).abs()
        # print(x.size(),map.size(),legend.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # logits = torch.argmax(logits, dim=1)
        # print(logits.size())
        # viz_logits =  torch.argmax(logits, dim=1).cpu().numpy()
        # print(viz_logits.shape)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNet_S1_conv(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_S1_conv, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (SingleConv(n_channels, 64))
        self.down1 = (Down(64, 128, double_conv=False))
        self.down2 = (Down(128, 256, double_conv=False))
        self.down3 = (Down(256, 512, double_conv=False))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, double_conv=False))
        self.up1 = (Up(1024, 512 // factor, bilinear, double_conv=False))
        self.up2 = (Up(512, 256 // factor, bilinear, double_conv=False))
        self.up3 = (Up(256, 128 // factor, bilinear, double_conv=False))
        self.up4 = (Up(128, 64, bilinear, double_conv=False))
        self.outc = (OutConv(64, n_classes))
        self.std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        self.mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        self.max_div = (1 - self.mean)/self.std

    def forward(self, map, legend):
        # x = torch.cat((map, legend),axis=1)
        # mean, std = self.mean.to(map.device), self.std.to(map.device)
        # max_div = self.max_div.to(map.device)
        x = (map - legend).abs().sum(dim=1, keepdim=True)
        # x = sub_conv_feat(map, legend, max_div)
        
        # map = map*std + mean
        # legend = legend*std + mean
        # # save image of map, legend and x
        # map = map[0].cpu().numpy().transpose(1, 2, 0)
        # legend = legend[0].cpu().numpy().transpose(1, 2, 0)
        # x = x[0].cpu().numpy().transpose(1, 2, 0)
        # plt.imshow(map)
        # plt.savefig("temp/map.png")
        # plt.imshow(legend)
        # plt.savefig("temp/legend.png")
        # plt.imshow(x, cmap='gray')
        # plt.savefig("temp/x.png")
        # assert False

        x = torch.cat((map, x, legend),axis=1)
        # print(x.size(),map.size(),legend.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # logits = torch.argmax(logits, dim=1)
        # print(logits.size())
        # viz_logits =  torch.argmax(logits, dim=1).cpu().numpy()
        # print(viz_logits.shape)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class UNet_S1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_S1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (SingleConv(n_channels, 64))
        self.down1 = (Down(64, 128, double_conv=False))
        self.down2 = (Down(128, 256, double_conv=False))
        self.down3 = (Down(256, 512, double_conv=False))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor, double_conv=False))
        self.up1 = (Up(1024, 512 // factor, bilinear, double_conv=False))
        self.up2 = (Up(512, 256 // factor, bilinear, double_conv=False))
        self.up3 = (Up(256, 128 // factor, bilinear, double_conv=False))
        self.up4 = (Up(128, 64, bilinear, double_conv=False))
        self.outc = (OutConv(64, n_classes))

    def forward(self, map, legend):
        x = torch.cat((map, legend),axis=1)
        # print(x.size(),map.size(),legend.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # logits = torch.argmax(logits, dim=1)
        # print(logits.size())
        # viz_logits =  torch.argmax(logits, dim=1).cpu().numpy()
        # print(viz_logits.shape)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        
class UNet_S2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_S2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (SingleConv(n_channels, 32))
        self.down1 = (Down(32, 64, double_conv=False))
        self.down2 = (Down(64, 128, double_conv=False))
        self.down3 = (Down(128, 256, double_conv=False))
        factor = 2 if bilinear else 1
        self.down4 = (Down(256, 512 // factor, double_conv=False))
        self.up1 = (Up(512, 256 // factor, bilinear, double_conv=False))
        self.up2 = (Up(256, 128 // factor, bilinear, double_conv=False))
        self.up3 = (Up(128, 64 // factor, bilinear, double_conv=False))
        self.up4 = (Up(64, 32, bilinear, double_conv=False))
        self.outc = (OutConv(32, n_classes))

    def forward(self, map, legend):

        x = torch.cat((map, legend),axis=1)
        # print(x.size(),map.size(),legend.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # logits = torch.argmax(logits, dim=1)
        # print(logits.size())
        # viz_logits =  torch.argmax(logits, dim=1).cpu().numpy()
        # print(viz_logits.shape)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class UNet_S3_C(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_S3_C, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (SingleConv(n_channels, 16))
        self.down1 = (Down(16, 32, double_conv=False))
        self.down2 = (Down(32, 64, double_conv=False))
        self.down3 = (Down(64, 128, double_conv=False))
        factor = 2 if bilinear else 1
        self.up1 = (Up(128, 64 // factor, bilinear, double_conv=False))
        self.up2 = (Up(64, 32 // factor, bilinear, double_conv=False))
        self.up3 = (Up(32, 16, bilinear, double_conv=False))
        self.outc = (OutConv(16, n_classes))
        self.l_inc = (SingleConv(3, 16))   
        self.l_down1 = (Down(16, 16, double_conv=False))
        self.l_down2 = (Down(16, 16, double_conv=False))
        self.l_down3 = (Down(16, 16, double_conv=False))
        
        # self.c_down = (Down(1, 1, double_conv=False))
        self.c_up1 = (Up(65, 64, bilinear=True, double_conv=False))
        self.c_up2 = (Up(64, 32, bilinear, double_conv=False))
        self.c_up3 = (Up(32, 16, bilinear, double_conv=False))
        
        
    def forward(self, map, legend):

        # x = torch.cat((map, legend),axis=1)
        # print(x.size(),map.size(),legend.size())
        x = map
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        
        # resize legend to (B, C, 32, 32)
        legend = F.interpolate(legend, size=(32, 32), mode='bilinear', align_corners=True)
        l1 = self.l_inc(legend)
        l2 = self.l_down1(l1)
        l3 = self.l_down2(l2)
        l4 = self.l_down3(l3)

        # use average pooling to reduce the size of the legend
        l = F.adaptive_avg_pool2d(l4, (1, 1)).expand_as(x)
        # print(l.shape)
        x_l = nn.functional.cosine_similarity(x, l, dim=1, eps=1e-8).unsqueeze(1)
        # x_l = self.c_down(x_l)
        # downsample by 8 times 
        x_l = F.interpolate(x_l, size=(x_l.shape[2]//8, x_l.shape[3]//8), mode='bilinear', align_corners=True)

        x_l = self.c_up1(x_l, x3)

        x_l = self.c_up2(x_l, x2)
        x_l = self.c_up3(x_l, x1)
        
        
        logits = self.outc(x_l)
        # logits = torch.argmax(logits, dim=1)
        # print(logits.size())
        # viz_logits =  torch.argmax(logits, dim=1).cpu().numpy()
        # print(viz_logits.shape)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        
class UNet_S3(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_S3, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (SingleConv(n_channels, 16))
        self.down1 = (Down(16, 32, double_conv=False))
        self.down2 = (Down(32, 64, double_conv=False))
        self.down3 = (Down(64, 128, double_conv=False))
        factor = 2 if bilinear else 1
        self.up1 = (Up(128, 64 // factor, bilinear, double_conv=False))
        self.up2 = (Up(64, 32 // factor, bilinear, double_conv=False))
        self.up3 = (Up(32, 16, bilinear, double_conv=False))
        self.outc = (OutConv(16, n_classes))

    def forward(self, map, legend):

        x = torch.cat((map, legend),axis=1)
        # print(x.size(),map.size(),legend.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        # logits = torch.argmax(logits, dim=1)
        # print(logits.size())
        # viz_logits =  torch.argmax(logits, dim=1).cpu().numpy()
        # print(viz_logits.shape)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class UNet_S4(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_S4, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (SingleConv(n_channels, 16))
        self.down1 = (Down(16, 32, double_conv=False))
        self.down2 = (Down(32, 64, double_conv=False))
        factor = 2 if bilinear else 1
        self.up1 = (Up(64, 32 // factor, bilinear, double_conv=False))
        self.up2 = (Up(32, 16, bilinear, double_conv=False))
        self.outc = (OutConv(16, n_classes))

    def forward(self, map, legend):

        x = torch.cat((map, legend),axis=1)
        # print(x.size(),map.size(),legend.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        # logits = torch.argmax(logits, dim=1)
        # print(logits.size())
        # viz_logits =  torch.argmax(logits, dim=1).cpu().numpy()
        # print(viz_logits.shape)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        
class UNet_F1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_F1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        # self.outc = (OutConv(1, n_classes))

        self.c_inc = (DoubleConv(1, 64))
        self.c_down1 = (Down(64, 128))
        self.c_down2 = (Down(128, 256))
        self.c_up1 = (Up(256, 128 // factor, bilinear))
        self.c_up2 = (Up(128, 64, bilinear))
        self.c_outc = (OutConv(64, n_classes))

    def forward(self, map, legend):
        
        # print("debug in train",map.size(),legend.size())    
        # print("debug in train",map.min(),legend.min())
        # print("debug in train",map.max(),legend.max())
        # print("debug in train",map.mean(),legend.mean())
        # # normalize map and legend to 0-1
        # viz_map = (map - map.min()) / (map.max() - map.min())
        # viz_legend =  (map - legend.min()) / (legend.max() - legend.min())

        # save map and legend into images on disk
        # viz_map = viz_map.cpu().numpy()
        # viz_legend = viz_legend.cpu().numpy()
        # viz_map = viz_map[0,:,:,:]
        # viz_legend = viz_legend[0,:,:,:]
        # viz_map = np.transpose(viz_map,(1,2,0))
        # viz_legend = np.transpose(viz_legend,(1,2,0))
        # plt.imshow(viz_map)
        # plt.savefig("map.png")
        # plt.close()
        # plt.imshow(viz_legend)
        # plt.savefig("legend.png")
        # plt.close()

        # x = torch.cat((map, legend),axis=1) 
        x = map
        # print(x.size(),map.size(),legend.size())
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_l = legend
        x1_l = self.inc(x_l)
        x2_l = self.down1(x1_l)
        x3_l = self.down2(x2_l)
        x4_l = self.down3(x3_l)
        x5_l = self.down4(x4_l)


        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        x_l = self.up1(x5_l, x4_l)
        x_l = self.up2(x_l, x3_l)
        x_l = self.up3(x_l, x2_l)
        x_l = self.up4(x_l, x1_l)

        c = torch.nn.functional.cosine_similarity(x, x_l, dim=1)  
        c = c.unsqueeze(1)
        c1 = self.c_inc(c)
        c2 = self.c_down1(c1)
        c3 = self.c_down2(c2)
        c = self.c_up1(c3, c2)
        c = self.c_up2(c, c1)
        logits = self.c_outc(c)
        # logits = torch.argmax(logits, dim=1)
        # print(logits.size())
        # viz_logits =  torch.argmax(logits, dim=1).cpu().numpy()
        # print(viz_logits.shape)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

class UNet2Branch_MPool(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet2Branch_MPool, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder for map
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        
        # encoder for legend
        self.incl = (DoubleConv(n_channels, 64))
        self.downl1 = (Down(64, 128))
        self.downl2 = (Down(128, 256))
        self.downl3 = (Down(256, 512))
        self.downl4 = (Down(512, 1024 // factor))

        # decoder
        # self.up1 = (Fuse_Up(1024*2, 512 // factor, bilinear,fuse_channel = 512))
        
        # self.fuseconv = DoubleConv(1024 * 2, 1024)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, map, legend):
        # x = torch.cat((map, legend),axis=1)
        # print(x.size(),map.size(),legend.size())

        map1 = self.inc(map)
        map2 = self.down1(map1)
        map3 = self.down2(map2)
        map4 = self.down3(map3)
        map5 = self.down4(map4)

        legend1 = self.incl(legend)
        legend2 = self.downl1(legend1)
        legend3 = self.downl2(legend2)
        legend4 = self.downl3(legend3)
        legend5 = self.downl4(legend4)

        legend_pool = F.adaptive_avg_pool2d(legend5, (map5.size(2), map5.size(3)))
        # fused5 = torch.cat((map5,legend_pool),axis=1)
        fused5 = map5 * legend_pool
        # fused5 = torch.cat((map5,legend5),axis=1)
        # fused4 = torch.cat((map4,legend4),axis=1)
        # fused3 = torch.cat((map3,legend3),axis=1)
        # fused2 = torch.cat((map2,legend2),axis=1)
        # fused1 = torch.cat((map1,legend1),axis=1)
        # fused5 = self.fuseconv(fused5)

        x = self.up1(fused5, map4)
        x = self.up2(x, map3)
        x = self.up3(x, map2)
        x = self.up4(x, map1)
        logits = self.outc(x)
        # logits = torch.argmax(logits, dim=1)
        # print(logits.size())
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        
        self.incl = torch.utils.checkpoint(self.incl)
        self.downl1 = torch.utils.checkpoint(self.downl1)
        self.downl2 = torch.utils.checkpoint(self.downl2)
        self.downl3 = torch.utils.checkpoint(self.downl3)
        self.downl4 = torch.utils.checkpoint(self.downl4)

        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)
        
class UNet2Branch(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet2Branch, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # encoder for map
        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        
        # encoder for legend
        self.incl = (DoubleConv(n_channels, 64))
        self.downl1 = (Down(64, 128))
        self.downl2 = (Down(128, 256))
        self.downl3 = (Down(256, 512))
        self.downl4 = (Down(512, 1024 // factor))

        # decoder
        # self.up1 = (Fuse_Up(1024*2, 512 // factor, bilinear,fuse_channel = 512))
        
        self.fuseconv = DoubleConv(1024 * 2, 1024)
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, map, legend):
        # x = torch.cat((map, legend),axis=1)
        # print(x.size(),map.size(),legend.size())

        map1 = self.inc(map)
        map2 = self.down1(map1)
        map3 = self.down2(map2)
        map4 = self.down3(map3)
        map5 = self.down4(map4)

        legend1 = self.incl(legend)
        legend2 = self.downl1(legend1)
        legend3 = self.downl2(legend2)
        legend4 = self.downl3(legend3)
        legend5 = self.downl4(legend4)

        fused5 = torch.cat((map5,legend5),axis=1)
        # fused4 = torch.cat((map4,legend4),axis=1)
        # fused3 = torch.cat((map3,legend3),axis=1)
        # fused2 = torch.cat((map2,legend2),axis=1)
        # fused1 = torch.cat((map1,legend1),axis=1)
        fused5 = self.fuseconv(fused5)

        x = self.up1(fused5, map4)
        x = self.up2(x, map3)
        x = self.up3(x, map2)
        x = self.up4(x, map1)
        logits = self.outc(x)
        # logits = torch.argmax(logits, dim=1)
        # print(logits.size())
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        
        self.incl = torch.utils.checkpoint(self.incl)
        self.downl1 = torch.utils.checkpoint(self.downl1)
        self.downl2 = torch.utils.checkpoint(self.downl2)
        self.downl3 = torch.utils.checkpoint(self.downl3)
        self.downl4 = torch.utils.checkpoint(self.downl4)

        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class UNet_resnet18(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        
        self.base_model = models.resnet18()
        self.base_model.load_state_dict(torch.load("../input/resnet18/resnet18.pth"))
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

    def forward(self, map, legend):
        input = torch.cat((map, legend),axis=1)
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

class SpixelNet(nn.Module):
    expansion = 1
    def __init__(self, batchNorm=True):
        super(SpixelNet,self).__init__()

        self.batchNorm = batchNorm
        self.assign_ch = 9

        self.conv0a = conv(self.batchNorm, 3, 16, kernel_size=3)
        self.conv0b = conv(self.batchNorm, 16, 16, kernel_size=3)

        self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv1b = conv(self.batchNorm, 32, 32, kernel_size=3)

        self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv2b = conv(self.batchNorm, 64, 64, kernel_size=3)

        self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3b = conv(self.batchNorm, 128, 128, kernel_size=3)

        self.conv4a = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4b = conv(self.batchNorm, 256, 256, kernel_size=3)

        self.deconv3 = deconv(256, 128)
        self.conv3_1 = conv(self.batchNorm, 256, 128)
        self.pred_mask3 = predict_mask(128, self.assign_ch)

        self.deconv2 = deconv(128, 64)
        self.conv2_1 = conv(self.batchNorm, 128, 64)
        self.pred_mask2 = predict_mask(64, self.assign_ch)

        self.deconv1 = deconv(64, 32)
        self.conv1_1 = conv(self.batchNorm, 64, 32)
        self.pred_mask1 = predict_mask(32, self.assign_ch)

        self.deconv0 = deconv(32, 16)
        self.conv0_1 = conv(self.batchNorm, 32 , 16)
        self.pred_mask0 = predict_mask(16,self.assign_ch)

        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out1 = self.conv0b(self.conv0a(x)) #5*5
        out2 = self.conv1b(self.conv1a(out1)) #11*11
        out3 = self.conv2b(self.conv2a(out2)) #23*23
        out4 = self.conv3b(self.conv3a(out3)) #47*47
        out5 = self.conv4b(self.conv4a(out4)) #95*95

        out_deconv3 = self.deconv3(out5)
        concat3 = torch.cat((out4, out_deconv3), 1)
        out_conv3_1 = self.conv3_1(concat3)

        out_deconv2 = self.deconv2(out_conv3_1)
        concat2 = torch.cat((out3, out_deconv2), 1)
        out_conv2_1 = self.conv2_1(concat2)

        out_deconv1 = self.deconv1(out_conv2_1)
        concat1 = torch.cat((out2, out_deconv1), 1)
        out_conv1_1 = self.conv1_1(concat1)

        out_deconv0 = self.deconv0(out_conv1_1)
        concat0 = torch.cat((out1, out_deconv0), 1)
        out_conv0_1 = self.conv0_1(concat0)
        mask0 = self.pred_mask0(out_conv0_1)
        prob0 = self.softmax(mask0)

        return prob0

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

def super_pixel_inference(downsize, model, img_):
    # may get 4 channel (alpha channel) for some format
    # img is of shape (3, H, W)
    b, _, H, W = img_.shape
    H_, W_  = int(np.ceil(H/16.)*16), int(np.ceil(W/16.)*16)

    # get spixel id
    n_spixl_h = int(np.floor(H_ / downsize))
    n_spixl_w = int(np.floor(W_ / downsize))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    spix_idx_tensor = np.repeat(
      np.repeat(spix_idx_tensor_, downsize, axis=1), downsize, axis=2)

    spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float).cuda()
    n_spixel =  int(n_spixl_h * n_spixl_w)

    img = img_
    img1 = img_
    ori_img = img_
    
    # compute output
    output = model(img1)

    spixel_viz = output #np.zeros((len(output), 3, H_, W_))
    # for i in range(len(output)):
    # # assign the spixel map
    #     curr_spixl_map = update_spixl_map(spixeIds, output[i].unsqueeze(0))
    #     ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=( H_,W_), mode='nearest').type(torch.int)

    #     mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1[i].cuda().unsqueeze(0).dtype).view(3, 1, 1).cuda()
    #     # import pdb; pdb.set_trace()
    #     spixel_viz[i], spixel_label_map = get_spixel_image((ori_img[i] + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(), n_spixels= n_spixel,  b_enforce_connect=True)
        
    # plt.imshow(spixel_viz.transpose(1,2,0))
    # plt.savefig("superpixel_img.png")
    # plt.close()
    # print(np.shape(spixel_viz))
    return spixel_viz