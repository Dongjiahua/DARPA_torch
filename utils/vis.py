import torch 
from typing import List, Tuple, Dict
from utils.heatmap import generate_channel_heatmap,get_keypoints_from_heatmap_batch_maxpool
import os 

@torch.no_grad()
def visualize_pred(batch, output, batch_idx, epoch, out_dir, postix="val"):
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
    pred = output[0].permute(1,2,0).squeeze().cpu().detach().numpy()
    
    # import matplotlib.pyplot as plt
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(np.array(map))
    axarr[0,1].imshow(np.array(legend))
    axarr[1,0].imshow(np.array(seg),cmap="gray")
    axarr[1,1].imshow(np.array(pred),cmap="gray")
    
    # axarr[2].imshow(seg_img,cmap="gray")        
    # plt.show()
    f.savefig(os.path.join(out_dir,f"epoch_{epoch}_step_{postix}_{batch_idx}.png"))
    plt.close(f)

@torch.no_grad()
def visualize_pred_kpts(batch, output, batch_idx, epoch, out_dir, postix="val"):
    import numpy as np
    import matplotlib.pyplot as plt
    mean = np.array([0.485, 0.456, 0.406]).reshape(1,1,3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1,1,3)
    map, legend, seg = batch['map_img'][0], batch['legend_img'][0], batch['seg_img'][0]

    map, legend, seg = map.permute(1,2,0).cpu(), legend.permute(1,2,0).cpu().detach().numpy(), seg.cpu().detach().numpy().squeeze()
    map = map*torch.tensor(std)+torch.tensor(mean)
    legend = legend*std+mean
    map[map>1] = 1
    legend[legend>1] = 1
    map[map<0] = 0
    legend[legend<0] = 0
    pred = output[0]
    map = torch.nn.functional.interpolate(map.unsqueeze(0).permute(0,3,1,2),size=pred.shape[-2:],mode="bilinear").squeeze(0).permute(1,2,0)
    keypoints, scores = get_keypoints_from_heatmap_batch_maxpool(
            pred.unsqueeze(0), 20, 5, return_scores=True
        )
    pred = overlay_image_with_keypoints(map.unsqueeze(0).permute(0,3,1,2), [torch.tensor(keypoints[0])], sigma=3)[0].permute(1,2,0).cpu().detach().numpy()
    # import matplotlib.pyplot as plt
    f, axarr = plt.subplots(2,2)
    axarr[0,0].imshow(np.array(map))
    axarr[0,1].imshow(np.array(legend))
    axarr[1,0].imshow(np.array(seg),cmap="gray")
    axarr[1,1].imshow(np.array(pred),cmap="gray")
    
    # axarr[2].imshow(seg_img,cmap="gray")        
    # plt.show()
    f.savefig(os.path.join(out_dir,f"epoch_{epoch}_step_{postix}_{batch_idx}.png"))
    plt.close(f)
        
def overlay_image_with_keypoints(images: torch.Tensor, keypoints: List[torch.Tensor], sigma: float=3) -> torch.Tensor:
    """
    images N x 3 x H x W
    keypoints list of size N with Tensors C x 2


    Returns:
        torch.Tensor: N x 3 x H x W
    """

    image_size = images.shape[2:]
    alpha = 0.5
    keypoint_color = torch.Tensor([240.0, 240.0, 10.0]) / 255.0
    keypoint_color = keypoint_color.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    overlayed_images = []
    for i in range(images.shape[0]):

        heatmaps = generate_channel_heatmap(image_size, keypoints[i], sigma=sigma, device="cpu")  # C x H x W
        heatmaps = heatmaps.unsqueeze(0)  # 1 xC x H x W
        colorized_heatmaps = keypoint_color * heatmaps
        combined_heatmap = torch.max(colorized_heatmaps, dim=1)[0]  # 3 x H x W
        combined_heatmap[combined_heatmap < 0.1] = 0.0  # avoid glare

        overlayed_image = images[i] * alpha + combined_heatmap
        overlayed_image = torch.clip(overlayed_image, 0.0, 1.0)
        overlayed_images.append(overlayed_image)
    overlayed_images = torch.stack(overlayed_images)
    return overlayed_images

def overlay_image_with_keypoints(images: torch.Tensor, keypoints: List[torch.Tensor], sigma: float=3) -> torch.Tensor:
    """
    images N x 3 x H x W
    keypoints list of size N with Tensors C x 2


    Returns:
        torch.Tensor: N x 3 x H x W
    """

    image_size = images.shape[2:]
    alpha = 0.5
    keypoint_color = torch.Tensor([240.0, 240.0, 10.0]) / 255.0
    keypoint_color = keypoint_color.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    overlayed_images = []
    for i in range(images.shape[0]):

        heatmaps = generate_channel_heatmap(image_size, keypoints[i], sigma=sigma, device="cpu")  # C x H x W
        heatmaps = heatmaps.unsqueeze(0)  # 1 xC x H x W
        colorized_heatmaps = keypoint_color * heatmaps
        combined_heatmap = torch.max(colorized_heatmaps, dim=1)[0]  # 3 x H x W
        combined_heatmap[combined_heatmap < 0.1] = 0.0  # avoid glare

        overlayed_image = images[i] * alpha + combined_heatmap
        overlayed_image = torch.clip(overlayed_image, 0.0, 1.0)
        overlayed_images.append(overlayed_image)
    overlayed_images = torch.stack(overlayed_images)
    return overlayed_images

def predict_final_map(output):
    keypoints, scores = get_keypoints_from_heatmap_batch_maxpool(
            output, 20, 5, return_scores=True
        )
    final_maps = torch.zeros(output.shape[0],*output.shape[-2:])
    for i, kpts in enumerate(keypoints):
        kpts = kpts[0]
        for point in kpts:
            x,y = point 
            final_maps[i,y,x] = 1
    return final_maps



