""" Full assembly of the parts to form the complete network """

from .unet_parts import *


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
    
        

    def forward(self, x):
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

class MAP_UNet(nn.Module):
    def __init__(self, args):
        super(MAP_UNet, self).__init__()
        freeze = args.freeze
        pretrained = args.pretrained
        n_channels = 6
        n_classes = 1
        self.unet = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=pretrained, scale=0.5)
        if freeze:
            if pretrained:
                for param in self.unet.parameters():
                    param.requires_grad = False
            else:
                raise Exception("Cannot freeze untrained model")
        self.unet.inc = (DoubleConv(n_channels, 64))
        self.unet.outc = (OutConv(64, n_classes))
    
    def forward(self, x, legend, instance):
        x = torch.cat([x,legend],dim=1)
        x = self.unet(x)
        x = F.interpolate(x,size=(256,256),mode="bilinear")
        x = torch.sigmoid(x)
        return x
    
class Sim_UNet(nn.Module):
    def __init__(self, args):
        super(Sim_UNet, self).__init__()
        freeze = args.freeze
        pretrained = args.pretrained
        n_channels = 3
        n_classes = 1
        self.unet = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana', pretrained=pretrained, scale=0.5)
        
        if freeze:
            if pretrained:
                for param in self.unet.parameters():
                    param.requires_grad = False
            else:
                raise Exception("Cannot freeze untrained model")
        self.unet.outc = nn.Identity()
        import copy 
        self.unet_t = copy.deepcopy(self.unet)
        for param in self.unet_t.parameters():
            param.requires_grad = False
        self.outc = OutConv(64, n_classes)
    
    def forward(self, x, legend, instance):
        legend = F.interpolate(legend,size=(32,32),mode="bilinear")
        x = self.unet(x)
        with torch.no_grad():
            legend = self.unet_t(legend)
        # legend = self.unet_t(legend)
        # x = torch.cat([x,legend],dim=1)
        x = F.interpolate(x,size=(256,256),mode="bilinear")
        legend = F.adaptive_avg_pool2d(legend, (1,1))
        # x = self.outc(x)
        # x = F.normalize(x,dim=1)
        # legend = F.normalize(legend,dim=1)
        x = x*legend
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x
            
            