# import rembg
import numpy as np
from PIL import Image
import cv2
import os
import copy
from tqdm import tqdm

def thresholding(img):
    img =cv2.imread(img)

    # First Convert to Grayscale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
    ret,baseline = cv2.threshold(img_grey,127,255,cv2.THRESH_TRUNC)
 
    ret,background = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY)
 
    ret,foreground = cv2.threshold(baseline,126,255,cv2.THRESH_BINARY_INV)
 
    foreground = cv2.bitwise_and(img,img, mask=foreground)  # Update foreground with bitwise_and to extract real foreground
    # cv2.imwrite("foreground.png",foreground)
    # Convert black and white back into 3 channel greyscale
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    # cv2.imwrite("background.png",background)

    # Combine the background and foreground to obtain our final image
    sharpened = background+foreground
    # cv2.imwrite("method_2.png",sharpened)
    return sharpened

def remove_bg(img, file_name=None,output_dir=None):
        # join output filename
        if file_name:
            output_path = os.path.join(output_dir, file_name)

        # convert image to 4-channel
        # img.save('original.png')
        img = img.convert("RGBA")
        # img.save('test2.png')
        newData = []
        for item in img.getdata():
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                newData.append((255, 255, 255, 0))
            else:
                newData.append(item)
        img.putdata(newData)
        if file_name:
            img.save(output_path, "PNG")
        return img

def merge_bg(front,back,location):
    front = front.convert("RGBA")
    back = back.convert("RGBA")
    # Paste the frontImage at (width, height)
    back.paste(front, (location[0], location[1]), front)
    back =back.convert("RGB")
    return back

def data_augment(dataset_path,output_path):
    output_dir = output_path
    root_dir = dataset_path
    data_type = "point"
    map_path_dir = os.listdir(os.path.join(root_dir,data_type,"map_patches"))
    legend_path = ['_'.join(x.split('_')[0:-2])+'.png' for x in map_path_dir]
    image_size = (224,224)

    # parse data path
    map_path = [os.path.join(root_dir,data_type,"map_patches",x) for x in map_path_dir]
    legend_path = [os.path.join(root_dir,data_type,"legend",x) for x in legend_path]
    seg_path = [os.path.join(root_dir,data_type,"seg_patches",x) for x in map_path_dir]

    # create folder for legend and image patches
    output_map_path = os.path.join(output_dir,data_type,"map_patches")
    if not os.path.exists(output_map_path):
        os.makedirs(output_map_path)
    output_legend_path = os.path.join(output_dir,data_type,"legend")
    if not os.path.exists(output_legend_path):
        os.makedirs(output_legend_path)

    for i in tqdm(range(len(map_path))):
        # get original imgs
        map_img = Image.open(map_path[i])
        legend_img = Image.open(legend_path[i])
        seg_img = Image.open(seg_path[i])
        legend_name = os.path.basename(legend_path[i]).split('/')[-1]
        map_name =  os.path.basename(map_path[i]).split('/')[-1]

        # map_img.save('map1.png')
        # legend_img.save('legend1.png')
        # seg_img.save('seg1.png')
        # find location of legend on map_img

        seg_img = np.array(seg_img)
        annotations, keypoints= get_bbox(seg_img)

        # print(annotations)      # x_left, y_top, x_right, y_bottom

        # sharped legend
        sharpend_legend = thresholding(legend_path[i])
        # convert color from cv2 -> Image
        sharpend_legend = cv2.cvtColor(sharpend_legend, cv2.COLOR_BGR2RGB)
        sharpend_legend = Image.fromarray(sharpend_legend)
        bgrm_legend = remove_bg(sharpend_legend, legend_name, output_legend_path)

        # paste all augmented_legend to the map
        for anno in annotations:
            x = int(anno[2]) - int(anno[0])
            y = int(anno[3]) - int(anno[1])
            resized_legend = bgrm_legend.resize((x,y))
            map_img = merge_bg(resized_legend,map_img,location=(int(anno[0]),int(anno[1])))

        output_map = os.path.join(output_map_path, map_name)
        map_img.save(output_map)
        # exit(1)

def get_bbox(seg_img,frac = 0.05):
        '''
        return:
            point_annotation: (N,2)
        '''
        points = np.array(np.where(seg_img==1)).T
        # if len(points.shape)==1:
        #     points = points.reshape(1,-1)
        # print(points.shape)
        
        # use boxes around the points as annotations
        point_annotation = np.zeros((len(points),4))
        for i,point in enumerate(points):
            y, x = point[0],point[1]
            range = int(seg_img.shape[0]*frac)
            point_annotation[i,:] = [x-range,y-range,x+range,y+range]
        return point_annotation, points[:,[1,0]]




if __name__ == "__main__":
    

    trainingPath = '/home/shared/DARPA/all_patched_data/training'
    validationPath = '/home/shared/DARPA/all_patched_data/validation'
    # change the path 
    savedPath = '/home/wwwwwzr409/DARPA_torch/data_augmented'


    # create the output folder if not exists
    if not os.path.exists(savedPath):
        os.makedirs(savedPath)

    data_augment(trainingPath,savedPath)