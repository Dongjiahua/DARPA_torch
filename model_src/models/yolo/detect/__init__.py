# Ultralytics YOLO 🚀, AGPL-3.0 license

from .predict import DetectionPredictor, OneShotDetectionPredictor
from .train import DetectionTrainer, OneshotDetectionTrainer
from .val import DetectionValidator, OneShotDetectionValidator

__all__ = 'DetectionPredictor', 'OneShotDetectionPredictor','DetectionTrainer', 'DetectionValidator','OneShotDetectionValidator', 'OneshotDetectionTrainer'
