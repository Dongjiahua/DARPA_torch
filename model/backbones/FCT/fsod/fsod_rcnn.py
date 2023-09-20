# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modified on Wednesday, September 28, 2022

@author: Guangxing Han
"""
import logging
import numpy as np
import torch
from torch import nn

from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from .fsod_roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from detectron2.modeling.poolers import ROIPooler
import torch.nn.functional as F

from .fsod_fast_rcnn import FsodFastRCNNOutputs

import os

import matplotlib.pyplot as plt
import pandas as pd

from detectron2.data.catalog import MetadataCatalog
import detectron2.data.detection_utils as utils
import pickle
import sys

__all__ = ["FsodRCNN"]


@META_ARCH_REGISTRY.register()
class FsodRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES

        self.support_way = cfg.INPUT.FS.SUPPORT_WAY
        self.support_shot = cfg.INPUT.FS.SUPPORT_SHOT
        self.logger = logging.getLogger(__name__)
        self.support_dir = cfg.OUTPUT_DIR

        self.evaluation_dataset = 'voc'
        self.evaluation_shot = 10
        self.keepclasses = 'all1'
        self.test_seeds = 0
        self.final_out = nn.Conv2d(320,1,1,1)

    def init_support_features(self, evaluation_dataset, evaluation_shot, keepclasses, test_seeds):
        self.evaluation_dataset = evaluation_dataset
        self.evaluation_shot = evaluation_shot
        self.keepclasses = keepclasses
        self.test_seeds = test_seeds

        if self.evaluation_dataset == 'voc':
            self.init_model_voc()
        elif self.evaluation_dataset == 'coco':
            self.init_model_coco()

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, images, support_images, instances):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """

        
        # images, support_images = self.preprocess_image(batched_inputs)
        # images, support_images = batched_inputs
        # if "instances" in batched_inputs[0]:
        #     for x in batched_inputs:
        #         x['instances'].set('gt_classes', torch.full_like(x['instances'].get('gt_classes'), 0))
            
        #     gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        # else:
        #     gt_instances = None



        x,y = self.backbone.forward_with_two_branch(images, support_images)['res4']

        
        query_images = ImageList.from_tensors([images[i] for i in range(len(images))])
        query_features_res4 = x 
        query_features = {'res4': query_features_res4}
        
        pos_support_features = y
        pos_support_features_pool = pos_support_features.mean(dim=[2, 3], keepdim=True)
        #TODO: check the correctness of the following line
        pos_correlation = query_features_res4*pos_support_features_pool # attention map
        pos_map_out = self.final_out(torch.nn.functional.interpolate(query_features_res4,size=(256,256),mode="bilinear"))
        pos_map_out = torch.sigmoid(pos_map_out)
        return pos_map_out
        pos_features = {'res4': pos_correlation} 
        # query_gt_instances = instances
        # pos_proposals, pos_anchors, pos_pred_objectness_logits, pos_gt_labels, pos_pred_anchor_deltas, pos_gt_boxes = self.proposal_generator(query_images, pos_features, query_gt_instances)
        # # print(pos_proposals[0].shape)
        
        # # print(pos_pred_proposal_deltas)
        
        # # rpn loss
        # outputs_pred_objectness_logits = pos_pred_objectness_logits
        # outputs_pred_anchor_deltas = pos_pred_anchor_deltas 
        
        # outputs_anchors = pos_anchors 


        # outputs_gt_boxes = pos_gt_boxes 
        # outputs_gt_labels = pos_gt_labels

        # if True:
        # # if self.training:
        #     proposal_losses = self.proposal_generator.losses(
        #         outputs_anchors, outputs_pred_objectness_logits, outputs_gt_labels, outputs_pred_anchor_deltas, outputs_gt_boxes)
        #     proposal_losses = {k: v * self.proposal_generator.loss_weight for k, v in proposal_losses.items()}
        # else:
        #     proposal_losses = {}

        # # # detector loss
        # # pos_pred_class_logits, pos_pred_proposal_deltas, pos_detector_proposals = self.roi_heads(query_images, query_features, pos_support_features, pos_proposals, query_gt_instances)
        # # detector_pred_class_logits = pos_pred_class_logits
        # # detector_pred_proposal_deltas = pos_pred_proposal_deltas
        # # detector_proposals = pos_detector_proposals
        
        # # # if self.training:
        # # if True:
        # #     predictions = detector_pred_class_logits, detector_pred_proposal_deltas
        # #     detector_losses = self.roi_heads.box_predictor.losses(predictions, detector_proposals)
            
        # rpn_loss_rpn_cls.append(proposal_losses['loss_rpn_cls'])
        # rpn_loss_rpn_loc.append(proposal_losses['loss_rpn_loc'])
        # # detector_loss_box_reg.append(detector_losses['loss_box_reg'])
        
        # proposal_losses = {}
        # detector_losses = {}

        # proposal_losses['loss_rpn_cls'] = torch.stack(rpn_loss_rpn_cls).mean()
        # proposal_losses['loss_rpn_loc'] = torch.stack(rpn_loss_rpn_loc).mean()
        # # detector_losses['loss_cls'] = torch.stack(detector_loss_cls).mean() 
        # # detector_losses['loss_box_reg'] = torch.stack(detector_loss_box_reg).mean()

        # losses = {}
        # losses.update(proposal_losses)
        # losses.update(detector_losses)
        return losses


    def inference(self, images, support_images, do_postprocess=True):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        
        x,y = self.backbone.forward_with_two_branch(images, support_images)['res4']

        query_images = ImageList.from_tensors([images[i] for i in range(len(images))])
        query_features_res4 = x 
        query_features = {'res4': query_features_res4}
        
        pos_support_features = y
        pos_support_features_pool = pos_support_features.mean(0, True).mean(dim=[2, 3], keepdim=True)
        #TODO: check the correctness of the following line
        pos_correlation = F.conv2d(query_features_res4, pos_support_features_pool.permute(1,0,2,3), groups=query_features_res4.shape[1]) # attention map

        pos_features = {'res4': pos_correlation}
        pos_proposals, pos_anchors, pos_pred_objectness_logits, pos_gt_labels, pos_pred_anchor_deltas, pos_gt_boxes = self.proposal_generator(query_images, pos_features, None)

        support_proposals_dict = {}
        support_box_features_dict = {}
        proposal_num_dict = {}
        query_features_dict = {}

        for cls_id, support_images in self.support_dict['image'].items():
            query_images = ImageList.from_tensors([images[0]]) # one query image

            features_dict = self.backbone.forward_with_two_branch(query_images.tensor, support_images.tensor)

            query_features_res4 = features_dict['res4'][0] # one query feature for attention rpn
            query_features = {'res4': query_features_res4} # one query feature for rcnn

            # support branch ##################################
            support_features_res4 = features_dict['res4'][1]
            support_features = {'res4': support_features_res4}
            pos_support_features = self.roi_heads.roi_pooling(support_features, self.support_dict['box'][cls_id])
            pos_support_features_pool = pos_support_features.mean(0, True).mean(dim=[2, 3], keepdim=True)

            correlation = F.conv2d(query_features_res4, pos_support_features_pool.permute(1,0,2,3), groups=query_features_res4.shape[1]) # attention map

            support_correlation = {'res4': correlation} # attention map for attention rpn

            proposals, _ = self.proposal_generator(query_images, support_correlation, None)
            support_proposals_dict[cls_id] = proposals
            support_box_features_dict[cls_id] = pos_support_features
            query_features_dict[cls_id] = query_features

            if cls_id not in proposal_num_dict.keys():
                proposal_num_dict[cls_id] = []
            proposal_num_dict[cls_id].append(len(proposals[0]))

            # del support_box_features
            del correlation
            # del res4_avg
            del query_features_res4

        results, _ = self.roi_heads.eval_with_support(query_images, query_features_dict, support_proposals_dict, support_box_features_dict)
        
        if do_postprocess:
            return FsodRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        if self.training:
            # support images
            support_images = [x['support_images'].to(self.device) for x in batched_inputs]
            support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
            support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)

            return images, support_images
        else:
            return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
