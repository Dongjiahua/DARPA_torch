import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import random
from torchvision import transforms

class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, masks):
        if random.random() < self.p:
            img = F.hflip(img)
            masks = [F.hflip(mask) for mask in masks]
        return img, masks

class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, masks):
        if random.random() < self.p:
            img = F.vflip(img)
            masks = [F.vflip(mask) for mask in masks]
        return img, masks

class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, img, masks):
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.size)
        img = F.crop(img, i, j, h, w)
        masks = [F.crop(mask, i, j, h, w) for mask in masks]
        return img, masks

class RandomResizedCrop:
    def __init__(self, size, scale=(0.8, 1.0)):
        self.size = size
        self.scale = scale

    def __call__(self, img, masks):
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=self.scale, ratio=(1.0, 1.0))
        img = F.resized_crop(img, i, j, h, w, self.size)
        masks = [F.resized_crop(mask, i, j, h, w, self.size) for mask in masks]
        return img, masks

class Identity:
    def __call__(self, img, masks):
        return img, masks
    
class RandomRotation:
    def __init__(self, degrees):
        self.degrees = transforms.RandomRotation(degrees).degrees

    def __call__(self, img, masks):
        angle = transforms.RandomRotation.get_params(self.degrees)
        img = F.rotate(img, angle)
        masks = [F.rotate(mask, angle) for mask in masks]
        return img, masks


class Color_Jitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
        self.color_jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)


    def __call__(self, imgs):
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.color_jitter.get_params(
            self.color_jitter.brightness, self.color_jitter.contrast, self.color_jitter.saturation, self.color_jitter.hue
        )
        result = []
        for img in imgs:
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)
            
            result.append(img)

        
        return result
    
# 自定义数据集
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths_list, transform=None):
        self.image_paths = image_paths
        self.mask_paths_list = mask_paths_list
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        masks = [Image.open(mask_path).convert('L') for mask_path in self.mask_paths_list[idx]]

        if self.transform:
            image, masks = self.transform(image, masks)

        return F.to_tensor(image), [F.to_tensor(mask) for mask in masks] if masks else []

# 定义数据增广组合
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, masks):
        for t in self.transforms:
            img, masks = t(img, masks)
        return img, masks

if __name__=='__main__':

# 使用自定义增强方法
    train_transform = Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomCrop((256, 256)),
        RandomResizedCrop((256, 256)),
        RandomRotation(15),
        ColorJitter(),
    ])

