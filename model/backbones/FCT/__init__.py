"""
Created on Wednesday, September 28, 2022

@author: Guangxing Han
"""
from .fsod import FsodRCNN, FsodRes5ROIHeads, FsodFastRCNNOutputLayers, FsodRPN
from .config import get_cfg
from detectron2.engine import DefaultTrainer as d2Trainer
from detectron2.checkpoint import DetectionCheckpointer

_EXCLUDE = {"torch", "ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]

def build_model(args):
    cfg_path = args.fct_cfg
    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    
    cfg.freeze()
    model = d2Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
    return model
