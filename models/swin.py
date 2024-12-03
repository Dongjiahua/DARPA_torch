    
import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
# from mmcv.runner import init_dist
# from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        

    def forward(self, map, legend):

        return 
    
def build_swin(in_chans = 7):
    model = dict(
    backbone=dict(
        in_chans = in_chans,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False
    ),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],
        num_classes=150
    ),)


    model = build_segmentor(
        model,
        train_cfg=None,
        test_cfg=None)
    return model


if __name__ == '__main__':
    print(build_swin(in_chans = 7))

