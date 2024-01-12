
import argparse
import torch 
import os 
import numpy as np
import cv2 
import math 
from data.h5Image import H5Image
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rasterio
import time
from model_src import RTDETR, SAM, YOLO

def save_plot_as_png(prediction_result, map_name, legend, outputPath):
    """
    This function visualizes and saves the True Segmentation, Predicted Segmentation, Full Map, 
    and the Legend in a single image.

    Parameters:
    - prediction_result: 2D numpy array representing the predicted segmentation.
    - map_name: string, the name of the map.
    - legend: string, the name of the legend.
    - outputPath: string, the directory where the output image will be saved.

    Returns:
    - None. The output image is saved in the specified directory.
    """

    global h5_image  # Using a global variable to access the h5 image object

    # Fetching the true segmentation layer
    true_seg = h5_image.get_layer(map_name, legend)
    
    # Fetching the full map
    full_map = h5_image.get_map(map_name)
    
    # Fetching the legend patch from h5 image
    legend_patch = h5_image.get_legend(map_name, legend)
    
    # Resize the legend to the specified dimensions
    legend_resized = cv2.resize(legend_patch, (256,256))

    # Convert the legend to uint8 range [0, 255] if its dtype is float32
    if legend_resized.dtype == np.float32:
        legend_resized = (legend_resized * 255).numpy().astype(np.uint8)

    # Construct the output image path
    output_image_path = os.path.join(outputPath, f"{map_name}_{legend}_visual.png")

    # Create a figure with 4 subplots: true segmentation, predicted segmentation, full map, and legend
    fig, axarr = plt.subplots(1, 4, figsize=(20,5))

    # Using GridSpec for custom sizing of the subplots
    gs = gridspec.GridSpec(1, 4, width_ratios=[1,1,1,1])

    # Display the true segmentation
    ax0 = plt.subplot(gs[0])
    point_size = 10

    true_seg = np.array(true_seg)
    y_indices, x_indices = np.where(true_seg == 1)
    ax0.imshow(true_seg, cmap='gray')  # 假设mask是灰度的，所以使用灰度色图
    ax0.scatter(x_indices, y_indices, s=point_size, color='red', marker='o')

    ax0.set_title('True segmentation')
    ax0.axis('off')

    # Display the predicted segmentation
    ax1 = plt.subplot(gs[1])
    y_indices, x_indices = np.where(prediction_result == 1)
    ax1.imshow(prediction_result, cmap='gray')  # 假设mask是灰度的，所以使用灰度色图
    ax1.scatter(x_indices, y_indices, s=point_size, color='red', marker='o')
    
    ax1.set_title('Pred segmentation')
    ax1.axis('off')
    
    # Display the full map
    ax2 = plt.subplot(gs[2])
    ax2.imshow(full_map)
    ax2.set_title('Map')
    ax2.axis('off')

    # Display the resized legend
    ax3 = plt.subplot(gs[3])
    ax3.imshow(legend_resized)
    ax3.set_title('Legend')
    ax3.axis('off')

    # Adjust layout to ensure there's no overlap
    plt.tight_layout()

    # Save the combined visualization to the specified path
    plt.savefig(output_image_path)


def prediction_mask(prediction_result, map_name):
    """
    Apply a mask to the prediction image to isolate the area of interest.

    Parameters:
    - prediction_result: numpy array, The output of the model after prediction.
    - map_name: str, The name of the map used for prediction.

    Returns:
    - masked_img: numpy array, The masked prediction image.
    """
    global h5_image

    # Get the map array corresponding to the given map name
    map_array = np.array(h5_image.get_map(map_name))
    print("map_array", map_array.shape)

    # Convert the RGB map array to grayscale for further processing
    gray = cv2.cvtColor(map_array, cv2.COLOR_BGR2GRAY)

    # Identify the most frequent pixel value, which will be used as the background pixel value
    pix_hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    background_pix_value = np.argmax(pix_hist, axis=None)

    # Flood fill from the corners to identify and modify the background regions
    height, width = gray.shape[:2]
    corners = [[0,0],[0,height-1],[width-1, 0],[width-1, height-1]]
    for c in corners:
        cv2.floodFill(gray, None, (c[0],c[1]), 255)

    # Adaptive thresholding to remove small noise and artifacts
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    # Detect edges using the Canny edge detection method
    thresh_blur = cv2.GaussianBlur(thresh, (11, 11), 0)
    canny = cv2.Canny(thresh_blur, 0, 200)
    canny_dilate = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # Detect contours in the edge-detected image
    contours, hierarchy = cv2.findContours(canny_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Retain only the largest contour
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    
    # Create an empty mask of the same size as the prediction_result
    wid, hight = prediction_result.shape[0], prediction_result.shape[1]
    mask = np.zeros([wid, hight])
    mask = cv2.fillPoly(mask, pts=[contour], color=(1)).astype(np.uint8)

    # Convert prediction result to a binary format using a threshold
    prediction_result_int = (prediction_result > 0.5).astype(np.uint8)

    # Apply the mask to the thresholded prediction result
    masked_img = cv2.bitwise_and(prediction_result_int, mask)

    return masked_img

def save_results(prediction, map_name, legend, outputPath):
    """
    Save the prediction results to a specified output path.

    Parameters:
    - prediction: The prediction result (should be a 2D or 3D numpy array).
    - map_name: The name of the map.
    - legend: The legend associated with the prediction.
    - outputPath: The directory where the results should be saved.
    """

    global h5_image

    output_image_path = os.path.join(outputPath, f"{map_name}_{legend}.tif")

    # Convert the prediction to an image
    # Note: The prediction array may need to be scaled or converted before saving as an image
    # prediction_image = Image.fromarray((prediction*255).astype(np.uint8))

    # Save the prediction as a tiff image
    # prediction_image.save(output_image_path, 'TIFF')

    prediction_image = (prediction*255).astype(np.uint8)

    prediction_image = np.expand_dims(prediction_image, axis=0)

    rasterio.open(output_image_path, 'w', driver='GTiff', compress='lzw',
                height = prediction_image.shape[1], width = prediction_image.shape[2], count = prediction_image.shape[0], dtype = prediction_image.dtype,
                crs = h5_image.get_crs(map_name, legend), transform = h5_image.get_transform(map_name, legend)).write(prediction_image)



class Inference(torch.nn.Module):
    def __init__(self, model, args) -> None:
        super().__init__()
        self.model = model
        
    def process_image(self,image):  
        image = Image.fromarray(image)
        # zero pad the image to make it square, the length of the longer side, pad on the right and bottom
        if image.width != image.height:
            if image.width > image.height:
                pad = (0, 0, 0, image.width - image.height)
            elif image.width < image.height:
                pad = (0, 0, image.height - image.width,0)
                
            image = transforms.functional.pad(image, pad, 0, 'constant')         
        return image
        
    
    def forward(self, legend_patch, map_patch):
        img_size = map_patch.shape[:2]

        legend_patch = self.process_image(legend_patch)
        map_patch = self.process_image(map_patch)


        patched_predicted = np.zeros(img_size)
        input = {
            "img": map_patch,
            "legend": legend_patch
        }
        output = self.model(input, verbose=False)[0].boxes

        xywh = output.xywh
        cls = output.cls.cpu().numpy()
        kpts = xywh.cpu().numpy()[:,:2]
        kpts = np.round(kpts).astype(int)
        for i, kpt in enumerate(kpts):
            if int(cls[i]) == 0 and kpt[0] >= 0 and kpt[0] < img_size[1] and kpt[1] >= 0 and kpt[1] < img_size[0]:
                patched_predicted[int(kpt[1]), int(kpt[0])] = 1
        
        return patched_predicted
        
        
def perform_inference(legend_patch, map_patch, model):
    return model(legend_patch, map_patch)

def main(args):
    """
    Main function to orchestrate the map inference process.

    Parameters:
    - args: Command-line arguments.
    """
    global h5_image

    # Load the HDF5 file using the H5Image class
    print("Loading the HDF5 file.")
    h5_image = H5Image(args.mapPath, mode='r', patch_border=0)

    # Get map details
    print("Getting map details.")
    map_name = h5_image.get_maps()[0]
    print(f"Map Name: {map_name}")
    all_map_legends = h5_image.get_layers(map_name)
    print(f"All Map Legends: {all_map_legends}")
    

    map_legends = [legend for legend in all_map_legends if "_pt" in legend]
    # Get the size of the map
    map_width, map_height, _ = h5_image.get_map_size(map_name)
    
    # Calculate the number of patches based on the patch size and border
    num_rows = math.ceil(map_width / h5_image.patch_size)
    num_cols = math.ceil(map_height / h5_image.patch_size)
    
    # Build the Inference model
    model = YOLO(args.modelPath)
    model = Inference(model,args)
    # model.eval()    

    for legend in (map_legends):
        print(f"Processing legend: {legend}")
        start = time.time()
        full_prediction = np.zeros((map_width, map_height))
        legend_patch = h5_image.get_legend(map_name, legend)
        for row in range(num_rows):
            for col in range(num_cols):
                map_patch = h5_image.get_patch(row, col, map_name)

                prediction = perform_inference(legend_patch, map_patch, model)
                # print(f"Prediction for patch ({row}, {col}) completed.")

                
                # Calculate starting indices for rows and columns
                x_start = row * h5_image.patch_size
                y_start = col * h5_image.patch_size

                # Calculate ending indices for rows and columns
                x_end = x_start + h5_image.patch_size
                y_end = y_start + h5_image.patch_size

                # Adjust the ending indices if they go beyond the image size
                x_end = min(x_end, map_width)
                y_end = min(y_end, map_height)

                # Adjust the shape of the prediction if necessary
                prediction_shape_adjusted = prediction[:x_end-x_start, :y_end-y_start]

                # Assign the prediction to the correct location in the full_prediction array
                full_prediction[x_start:x_end, y_start:y_end] = prediction_shape_adjusted
       # Mask out the map background pixels from the prediction
        print("Applying mask to the full prediction.")
        masked_prediction = prediction_mask(full_prediction, map_name)

        os.makedirs(args.outputPath, exist_ok=True)
        save_plot_as_png(masked_prediction, map_name, legend, args.outputPath)

        # Save the results
        print("Saving results.")
        save_results(masked_prediction, map_name, legend, args.outputPath)
        print(f"Processing legend: {legend} completed, time: {time.time()-start}")


    
if __name__ == "__main__":
    # Command-line interface setup
    parser = argparse.ArgumentParser(description="Perform inference on a given map.")
    parser.add_argument("--mapPath", required=True, help="Path to the hdf5 file.")
    parser.add_argument("--outputPath", required=True, help="Path to save the inference results. ")
    parser.add_argument("--modelPath", default="./inference_model/Unet-attentionUnet.h5", help="Path to the trained model. Default is './inference_model/Unet-attentionUnet.h5'.")
    
    parser.add_argument('--model', type=str, default="unet_cat", help='backbone model')
    parser.add_argument('--patches', type=int, default=1, help='Patch size.')
    parser.add_argument('--input_size', type=int, default=112, help='Patch size.')
    parser.add_argument('--pretrained', action='store_true', help='Whether use pretrained model.')
    parser.add_argument('--freeze', action='store_true',  help='Whether freeze layers.')
    args = parser.parse_args()
    main(args)

    #python inference.py --mapPath '/projects/bbym/shared/data/commonPatchData/256/NV_SilverPeak_321289_1963_62500_geo_mosaic.hdf5' --outputPath './inference' --modelPath './checkpoint.ckpt'
    # model = build_model("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/config/fct.yaml")
    # python inference.py --mapPath '/media/jiahua/FILE/uiuc/NCSA/outputForTA4/NV_SilverPeak_321289_1963_62500_geo_mosaic.hdf5' --outputPath './inference' --modelPath './best.pt'

        
        