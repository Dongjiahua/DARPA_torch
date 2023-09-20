import torch 
from .heatmap import get_keypoints_from_heatmap_batch_maxpool
from .metrics import KeypointAPMetrics, DetectedKeypoint, Keypoint
from typing import Callable, Dict, List, Tuple


def update_channel_ap_metrics(
        predicted_heatmaps: torch.Tensor, gt_keypoints: List[torch.Tensor], validation_metric: KeypointAPMetrics
    ):
    """
    Updates the AP metric for a batch of heatmaps and keypoins of a single channel (!)
    This is done by extracting the detected keypoints for each heatmap and combining them with the gt keypoints for the same frame, so that
    the confusion matrix can be determined together with the distance thresholds.

    predicted_heatmaps: N x H x W tensor with the batch of predicted heatmaps for a single channel
    gt_keypoints: List of size N, containing K_i x 2 tensors with the ground truth keypoints for the channel of that sample
    """

    # log corner keypoints to AP metrics for all images in this batch
    formatted_gt_keypoints = [
        [Keypoint(int(k[0]), int(k[1])) for k in frame_gt_keypoints] for frame_gt_keypoints in gt_keypoints
    ]
    batch_detected_channel_keypoints = extract_detected_keypoints_from_heatmap(
        predicted_heatmaps.unsqueeze(1)
    )
    # print(batch_detected_channel_keypoints)
    batch_detected_channel_keypoints = [batch_detected_channel_keypoints[i][0] for i in range(len(gt_keypoints))]
    for i, detected_channel_keypoints in enumerate(batch_detected_channel_keypoints):
        validation_metric.update(detected_channel_keypoints, formatted_gt_keypoints[i])
        
def extract_detected_keypoints_from_heatmap(heatmap: torch.Tensor, max_keypoints = 20, minimal_keypoint_pixel_distance=5) -> List[DetectedKeypoint]:
    """
    Extract keypoints from a single channel prediction and format them for AP calculation.

    Args:
    heatmap (torch.Tensor) : H x W tensor that represents a heatmap.
    """
    if heatmap.dtype == torch.float16:
        # Maxpool_2d not implemented for FP16 apparently
        heatmap_to_extract_from = heatmap.float()
    else:
        heatmap_to_extract_from = heatmap

    keypoints, scores = get_keypoints_from_heatmap_batch_maxpool(
        heatmap_to_extract_from, max_keypoints, minimal_keypoint_pixel_distance, return_scores=True
    )
    detected_keypoints = [
        [[] for _ in range(heatmap_to_extract_from.shape[1])] for _ in range(heatmap_to_extract_from.shape[0])
    ]
    for batch_idx in range(len(detected_keypoints)):
        for channel_idx in range(len(detected_keypoints[batch_idx])):
            for kp_idx in range(len(keypoints[batch_idx][channel_idx])):
                detected_keypoints[batch_idx][channel_idx].append(
                    DetectedKeypoint(
                        keypoints[batch_idx][channel_idx][kp_idx][0],
                        keypoints[batch_idx][channel_idx][kp_idx][1],
                        scores[batch_idx][channel_idx][kp_idx],
                    )
                )

    return detected_keypoints