# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path
from typing import Union
from ultralytics.engine.model import Model
from ultralytics.models import yolo  # noqa
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OneshotDetectionModel, PoseModel, SegmentationModel,H5_DetectionModel


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""
        
  
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            'classify': {
                'model': ClassificationModel,
                'trainer': yolo.classify.ClassificationTrainer,
                'validator': yolo.classify.ClassificationValidator,
                'predictor': yolo.classify.ClassificationPredictor, },
            'detect': {
                'model': DetectionModel,
                'trainer': yolo.detect.DetectionTrainer,
                'validator': yolo.detect.DetectionValidator,
                'predictor': yolo.detect.DetectionPredictor, },
            'hd5-detect': {
                'model': H5_DetectionModel,
                'trainer': yolo.detect.H5_DetectionTrainer,
                'validator': yolo.detect.H5_DetectionValidator,
                'predictor': yolo.detect.OneShotDetectionPredictor, },
            'oneshot-detect': {
                'model': OneshotDetectionModel,
                'trainer': yolo.detect.OneshotDetectionTrainer,
                'validator': yolo.detect.OneShotDetectionValidator,
                'predictor': yolo.detect.OneShotDetectionPredictor, },
            'segment': {
                'model': SegmentationModel,
                'trainer': yolo.segment.SegmentationTrainer,
                'validator': yolo.segment.SegmentationValidator,
                'predictor': yolo.segment.SegmentationPredictor, },
            'pose': {
                'model': PoseModel,
                'trainer': yolo.pose.PoseTrainer,
                'validator': yolo.pose.PoseValidator,
                'predictor': yolo.pose.PosePredictor, }, }
