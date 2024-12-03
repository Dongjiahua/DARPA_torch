import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
import PIL
import os 
import torch.nn.functional as F
import torchvision.transforms.functional as aug_f
import random
import cv2
import json 
from itertools import chain
from patchify import patchify, unpatchify
import matplotlib.pyplot as plt
import math
import copy
import flow_transforms
from h5image import H5Image
import glob
from random import choice, sample
import scipy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from sklearn.cluster import KMeans
from collections import Counter
from skimage import color
from new_trans import RandomHorizontalFlip, RandomVerticalFlip, RandomCrop, RandomResizedCrop, RandomRotation, Color_Jitter, Compose, Identity
from pycocotools.coco import COCO

def calculate_cdf(hist):
    cdf = hist.cumsum()
    cdf_normalized = 255 * (cdf - cdf.min()) / (cdf.max() - cdf.min())
    # print(cdf_normalized.max(), cdf_normalized.min())
    return cdf, cdf_normalized

def apply_cdf(image, cdf_normalized):
    img_flat = image.flatten()
    img_equalized = np.interp(img_flat, np.arange(256), cdf_normalized)
    return img_equalized.reshape(image.shape).astype(np.uint8)
    
def generate_point(mask):
    # mask should be of shape h, w
    # sample a point
    if np.sum(mask) <= 0:
        return None
    
    while True:
        point = (np.random.randint(0, mask.shape[0]), np.random.randint(0, mask.shape[1]))
        
        if mask[point]:
            # print(point)
            return point

def process_map(map_name, mapsh5i, train):
    cor = mapsh5i[map_name].get_map_corners(map_name) if train else \
        mapsh5i[map_name].h5f[map_name].attrs['corners']
    map_hist = np.zeros((256))
    for row in range(cor[0][0], cor[1][0]):
        for column in range(cor[0][1], cor[1][1]):
            map_img = mapsh5i[map_name].get_patch(row, column, map_name)
            Lab_img = cv2.cvtColor(map_img, cv2.COLOR_RGB2YCrCb)
            L_hist, _ = np.histogram(Lab_img[:,:,0].flatten(), bins=256, range=(0, 256))
            map_hist += L_hist
    cdf, cdf_normalized = calculate_cdf(map_hist)
    return map_name, cdf_normalized

def generate_heatmap(mask):

    h,w = mask.shape
    point = generate_point(mask)
    if point is None:
        return np.zeros((h,w))
    x = np.arange(0, np.max((h,w)))
    y = np.arange(0, np.max((h,w)))
    X, Y = np.meshgrid(x, y)
    Z = ((X - point[1])**2 + (Y - point[0])**2)**0.5

    # Normalize Z to 0 - 1
    Z = Z / (h**2 + w**2)**0.5

    # # viz mask and Z and save them as png
    # plt.imshow(mask)
    # plt.savefig("mask.png")
    # plt.imshow(Z[:h,:w])
    # plt.savefig("Z.png")
    
    # raise

    return Z[:h,:w]

def setup_h5im(train = True):
    mapsh5i = {}
    exclude_map = set(['CO_SanchezRes','CA_NV_LasVegas','CO_VailE100K','CA_Cambria','ID_LakeWalcott','AZ_PioRico_Nogales'])
    
    if train:
        map_files = sorted(glob.glob("/projects/bcxi/commonPatchData/256/32/*.hdf5"))
    else:
        map_files = sorted(glob.glob("/projects/bcxi/commonPatchData/validation/256/32/*.hdf5"))

    for file in map_files:
        mapname = os.path.basename(file).replace(".hdf5", "")
        if mapname not in exclude_map:
            mapsh5i[mapname] = H5Image(file, "r")
    print("Setup complete. Number of maps loaded:", len(mapsh5i))
    return mapsh5i


def unpatch_get_dice(items):
    print("lauch process")
    # vars, index = items
    preds, preds_max, legend_gt, legend_name, cut_shape, map_mask_im, shift_coord, viz, thre = items
    # thre = thre[index]
    print("fecthed vars")
    preds_thre = (preds & (preds_max > thre)).astype(np.int8)
    # preds = preds > 0.5
    final_dice = []
    print("start calcualte dice")
    
    for i in range(len(preds)):
        cur_legend_name = legend_name[i]
        print("for legend:", cur_legend_name)
        cur_gt = legend_gt[i]

        unpatch_predicted = unpatchify(preds_thre[i], (cut_shape[0], cut_shape[1], 1))
        pad_unpatch_predicted = np.pad(unpatch_predicted, [(shift_coord[0], shift_coord[1]), (shift_coord[2], shift_coord[3]), (0,0)], mode='constant')
        pad_unpatch_predicted = pad_unpatch_predicted.astype(int)
        
        # print("debugging", map.shape, pad_unpatch_predicted.shape)
        cur_masked_img = cv2.bitwise_and(pad_unpatch_predicted, map_mask_im) 
        # map_mask(map, pad_unpatch_predicted)

        intersection = np.logical_and(cur_masked_img.flatten(), cur_gt.flatten()).sum()
        gtsum = cur_gt.sum() 
        union = cur_masked_img.sum() + gtsum
        dice = (2.0 * intersection) / union
        print("dice:", dice, 'intersection:', intersection/gtsum, 'union:', union/gtsum)
        final_dice.append(dice)
        if viz:
            viz_seg = np.repeat(cur_masked_img[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            viz_gt = np.repeat(cur_gt[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
            cv2.imwrite("eval_viz/"+cur_legend_name+"_pred.tif", viz_seg) 
            cv2.imwrite("eval_viz/"+cur_legend_name+"_gt.tif", viz_gt) 
            plt.figure(figsize=(20,20))
            plt.imshow(viz_seg[:,:,0])
            plt.savefig("eval_viz/"+cur_legend_name+"_pred.png")
            plt.figure(figsize=(20,20))
            plt.imshow(viz_gt[:,:,0])
            plt.savefig("eval_viz/"+cur_legend_name+"_gt.png")
        print("final dice:", np.mean(final_dice))
    return final_dice

def generate_map_patches(mapsh5i,percent_valid = 0.75, train=True):
    map_patches = []
    for map_name in mapsh5i.keys():
        try:
            patches = mapsh5i[map_name].get_patches(map_name) 
        except:
            print(f"error when getting patches for {map_name}")
            continue
        

        valid_patches = [(map_name, layer, x, y) for layer in patches if layer.endswith('_poly') for x, y in patches[layer]]
        valid_layers = [key for key in patches.keys() if key.endswith('_poly')]
        n = len(valid_patches)
        if percent_valid <1:
            try:
                cor = mapsh5i[map_name].get_map_corners(map_name) if train else \
                    mapsh5i[map_name].h5f[map_name].attrs['corners']
                random_patches = [(map_name,choice(valid_layers), 
                            sample(range(cor[0][0], cor[1][0]), 1)[0], 
                            sample(range(cor[0][1], cor[1][1]), 1)[0]) 
                            for _ in range(len(valid_patches))]
            except:
                print(f"error when getting map corners for {map_name}")
                random_patches = []

        else:
            random_patches = []

        
        map_patches += sample(valid_patches, int(n * percent_valid)) + sample(random_patches, int(n * (1 - percent_valid)))

    return map_patches

def sincolor(image, position):
    # image is of shape 3 H W
    pe = position[image.long()] # 3 H W n
    pe = pe.transpose(1,-1) # 3 n H W
    pe = pe.flatten(start_dim=0, end_dim=1) # 3n H W 
    return torch.cat([pe,image],dim = 0) # 3n+3 H W

def canny(image):
    # image is of shape 3 H W
    edges = cv2.Canny(image,100,200)
    return torch.cat([edges,image],dim = 0) # 4 H W

def positionalencoding1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe

def cut_map(image,patch_dims = (256,256),patch_overlap=32):
    # patch parameter
    patch_dims = (256,256)
    patch_overlap = 32
    patch_step = patch_dims[1]-patch_overlap
    # cut boundary of the map 
    shift_x = (image.shape[0]-patch_dims[0])%patch_step
    shift_y = (image.shape[1]-patch_dims[1])%patch_step
    shift_x_left = shift_x//2
    shift_x_right = shift_x - shift_x_left
    shift_y_left = shift_y//2
    shift_y_right = shift_y - shift_y_left
    # recoder the shift_coord
    shift_coord =  [shift_x_left, shift_x_right, shift_y_left, shift_y_right]
    
    map_im_cut = image[shift_x_left:image.shape[0]-shift_x_right, shift_y_left:image.shape[1]-shift_y_right,:]
    cut_shape = np.shape(map_im_cut)
    map_patchs = patchify(map_im_cut, (*patch_dims,3), patch_step)
    patch_shape = np.shape(map_patchs)
    map_patchs = map_patchs.reshape((-1,patch_dims[0],patch_dims[1],3))
    return map_patchs, shift_coord, cut_shape, patch_shape


class COCOData(data.Dataset):
    def __init__(self, img_root, ann_file):
        self.img_root = img_root
        self.ann_file = ann_file
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_root, img['file_name'])
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert("RGB")
        mask = np.zeros((img.size[1], img.size[0]))
        for ann in anns:
            mask = np.maximum(self.coco.annToMask(ann), mask)
        mask = Image.fromarray(mask)
        img = self.img_transform(img)
        mask = self.mask_transform(mask).squeeze()
        legend_img = torch.zeros_like(img)
        a_channel = torch.tensor([0.0])
        gt = mask
        color_seg, seg_img = mask, mask
        
        return {
            "map_img": img,    # 3 H W
            "legend_img": legend_img, # 1 3 H W - > 3 H W 
            "a_channel": a_channel,
            "GT": gt, #.type(torch.LongTensor),    # 1 H W
            "color_seg": color_seg, #.type(torch.LongTensor),    # 1 H W
            "seg": seg_img, #.type(torch.LongTensor),    # 1 H W
            'qry_ori_size': torch.tensor((256,256)),
            'superpixel': img
        }
        
    def __len__(self):
        return len(self.ids)
    
    def get_img_info(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img = self.coco.loadImgs(img_id)[0]
        return img
        
class MAPData(data.Dataset):
    '''
    return:
        map_img: map image (3,resize_size,resize_size)
        legend_img: legend image (3,resize_size,resize_size)
        seg_img: segmentation image (3,resize_size,resize_size)
    '''
    def __init__(self, data_path=None,type="poly",range=None,resize_size = 256 , train = True, filp_rate = 0, color_jitter_rate = 0, edge = False, update_percent_valid_every = 1, heatmap = '', color = "rgb", hist=False, fuse=False, args=None):
        print("augmentation rate:", filp_rate, color_jitter_rate)
        self.color = color
        self.edge_only = edge
        self.heatmap = heatmap
        self.resize_size = resize_size
        self.train = train
        self.filp_rate = filp_rate
        self.color_jitter_rate = color_jitter_rate
        # if train:
        self.color_transform = Color_Jitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.5)
        if self.train:
            self.data_transforms = Compose([
                RandomHorizontalFlip(),
                RandomVerticalFlip(),
                RandomCrop((256, 256)),
                RandomResizedCrop((256, 256), scale=(0.2,0.8)),
                RandomRotation([-15, 15])
            # transforms.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue=0.5)
            ])
            self.legend_data_transforms = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.CenterCrop((196, 196)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                
            ])
        else:
            self.data_transforms = Compose([
                Identity()])
            self.legend_data_transforms = transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.CenterCrop((196, 196)),
            ])
        self.final_transforms = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

        # else:
        #     self.data_transforms = transforms.Compose([
        #     transforms.Resize((resize_size, resize_size)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                             std=[0.229, 0.224, 0.225]),
        #     ])
        self.mask_transforms = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor()
        ])
        self.type = type

        self.pe = positionalencoding1d(4, 256)
        self.mapsh5i = setup_h5im(train = train)
        self.iterations = 0
        self.percent_valid = 1
        self.update_percent_valid_every = update_percent_valid_every
        if self.train:
            self.patches = generate_map_patches(self.mapsh5i, percent_valid = 1, train = self.train)
        else:
            print("warning set validation set percent_valid to 1.0")
            self.patches = generate_map_patches(self.mapsh5i, percent_valid = 1., train = self.train)
        if range is not None:
            self.patches = self.patches[range[0]:range[1]]
        print("debug: ", np.shape(self.patches))
        self.shape = [0,0]
        if hist:
            self.build_hist_map()
        self.hist = hist
        self.assign_color_index()
        self.fuse = fuse
        self.args = args
    # def build_hist_map(self):
    #     cdfs = {}

    #     map_names = list(self.mapsh5i.keys())

    #     with ThreadPoolExecutor() as executor:
    #         futures = {executor.submit(process_map, map_name, self.mapsh5i, self.train): map_name for map_name in map_names}
    #         for future in tqdm(as_completed(futures), total=len(map_names), desc="Building CDFs"):
    #             map_name, cdf_normalized = future.result()
    #             cdfs[map_name] = cdf_normalized

    #     return cdfs
    def build_hist_map(self):
        # check if cdfs are already cached
        filename = f"cache/{'train' if self.train else 'val'}_cdfs.pkl"
        if os.path.exists(filename):
            print("Loading cdfs from cache")
            self.cdfs = pickle.load(open(filename, "rb"))
            return

        cdfs = {}
        
        for map_name in tqdm(self.mapsh5i.keys(), desc="Building CDFs"): 

            try:
                cor = self.mapsh5i[map_name].get_map_corners(map_name) 
            except:
                print(f"error when getting map corners for {map_name}")
                try:
                    patches = self.mapsh5i[map_name].get_patches(map_name)
                    used_patches = set()
                    map_hist = np.zeros((256))
                    for index in range(len(patches)):
                        map_name, layer_name, row, column = patches[index]
                        # ignore the layer and unique on row and column
                        idx = row * 10000 + column
                        if idx in used_patches:
                            continue
                        used_patches.add(idx)
                        map_img = self.mapsh5i[map_name].get_patch(row, column, map_name)
                        Lab_img = cv2.cvtColor(map_img, cv2.COLOR_RGB2YCrCb)
                        L_hist, _ = np.histogram(Lab_img[:,:,0].flatten(), bins=256, range=(0, 256))
                        map_hist += L_hist
                    cdf, cdf_normalized = calculate_cdf(map_hist)
                    cdfs[map_name] = cdf_normalized
                    continue
                # except
                except:
                    print(f"error when getting patches for {map_name}")
                    continue

            map_hist = np.zeros((256))
            for row in range(cor[0][0], cor[1][0]):
                for column in range(cor[0][1], cor[1][1]):
                    map_img = self.mapsh5i[map_name].get_patch(row, column, map_name)
                    Lab_img = cv2.cvtColor(map_img, cv2.COLOR_RGB2YCrCb)
                    L_hist, _ = np.histogram(Lab_img[:,:,0].flatten(), bins=256, range=(0, 256))
                    map_hist += L_hist
            cdf, cdf_normalized = calculate_cdf(map_hist)
            cdfs[map_name] = cdf_normalized
        self.cdfs = cdfs
        # cache the cdfs
        os.makedirs("cache", exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(cdfs, f)
    
    def apply_hist(self, image, map_name):
        # cdf = self.cdfs[map_name]
        # Lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        # Lab_img[:,:,0] = apply_cdf(Lab_img[:,:,0], cdf)
        # return cv2.cvtColor(Lab_img, cv2.COLOR_LAB2RGB)
        if type(image) == list:
            res = []
            for img in image:
                cdf = self.cdfs[map_name]
                Lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
                Lab_img[:,:,0] = apply_cdf(Lab_img[:,:,0], cdf)
                res.append(cv2.cvtColor(Lab_img, cv2.COLOR_YCrCb2RGB))
            return res
        else:
            cdf = self.cdfs[map_name]
            # print(cdf.max())
            Lab_img = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            Lab_img[:,:,0] = apply_cdf(Lab_img[:,:,0], cdf)
            res = cv2.cvtColor(Lab_img, cv2.COLOR_YCrCb2RGB)
            return res
    

    def get_dominant_color(self, image, k=4):
        # 下采样图像以加快处理速度
        # center crop the image to (196, 196)
        
        small_image = cv2.resize(image, (256, 256))
        small_image = small_image[30:226, 30:226]
        # 将图片数据转换为二维数组
        small_image = small_image.reshape((small_image.shape[0] * small_image.shape[1], 3))
        
        # 使用KMeans聚类
        kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
        kmeans.fit(small_image)
        
        # 找到每个簇的中心颜色
        centers = kmeans.cluster_centers_
        # 统计每个簇中的点数
        counts = Counter(kmeans.labels_)
        
        # 找到出现次数最多的簇
        dominant_color = centers[counts.most_common(1)[0][0]]
        # 转换为整数类型
        dominant_color = dominant_color.astype(int)
        
        return tuple(dominant_color)

    def build_color_base(self,num_colors = 20):
        
        
        n = num_colors
        L = np.linspace(0, 100, n)  # Lightness from 0 to 100
        a = np.linspace(-128, 127, n)  # a from -128 to 127
        b = np.linspace(-128, 127, n)  # b from -128 to 127
        self.lab_colors = np.array(np.meshgrid(L, a, b)).T.reshape(-1, 3)

    def assign_color_index(self, thres=3, lb=3, ub=30):
        # filename = f"cache/color_index_{'train' if self.train else 'val'}.pkl"
        # if os.path.exists(filename):
        #     print("Loading color index from cache")
        #     self.map_legend_color = pickle.load(open(filename, "rb"))
        #     return
        self.build_color_base()

        filename = f"cache/color_{'train' if self.train else 'val'}.pkl"
        if os.path.exists(filename):
            print("Loading color index from cache")
            self.dominate_colors = pickle.load(open(filename, "rb"))
        else:
            self.dominate_colors = {}

            for map_name in tqdm(self.mapsh5i.keys(), desc=f"Assigning color index for {'train' if self.train else 'val'} set"):
                # print(map_name)
                
                self.dominate_colors[map_name] = {}
                try:
                    legend_names = self.mapsh5i[map_name].get_layers(map_name)
                except:
                    print(f"error when getting layers for {map_name}")
                    continue
                if "boundary_line" in legend_names:
                    legend_names.remove("boundary_line")
                for legend_name in legend_names:
                    legend_img = self.mapsh5i[map_name].get_legend(map_name, legend_name)
                    dominant_color = np.array(self.get_dominant_color(legend_img))[None,None,...]
                    self.dominate_colors[map_name][legend_name] = dominant_color
                    # save legend image and dominant color as png for debug
                    # H, W, C = legend_img.shape
                    # saved_img = np.zeros((H, W*2, C))
                    # saved_img[:, :W] = legend_img
                    # saved_img[:, W:] = np.array(dominant_color)
                    # cv2.imwrite(f"debug/{map_name}_{legend_name}.png", saved_img)
                    # print(color.rgb2lab(dominant_color/255.0))
            with open(filename, "wb") as f:
                pickle.dump(self.dominate_colors, f)
        self.map_legend_color = {}
        self.map_legend_color_bound = {}
        # for map_name in self.dominate_colors:
        #     self.map_legend_color[map_name] = {}
        #     for legend_name in self.dominate_colors[map_name]:
                
        #         dominant_color = self.dominate_colors[map_name][legend_name]
        #         color_index = np.argmin(np.linalg.norm(self.lab_colors - color.rgb2lab(dominant_color/255.0).squeeze(), axis=1, ord=2))
        #         self.map_legend_color[map_name][legend_name] = color_index
        for map_name in self.dominate_colors:
            self.map_legend_color[map_name] = {}
            self.map_legend_color_bound[map_name] = {}
            lab_colors = []
            rgb_colors = []
            if len(self.dominate_colors[map_name]) == 0:
                continue
            for legend_name in self.dominate_colors[map_name]:
                
                dominant_color = self.dominate_colors[map_name][legend_name]
                lab_colors.append(color.rgb2lab(dominant_color/255.0)) # Lab color
                rgb_colors.append(dominant_color)
            # print(np.array(lab_colors).shape)
            lab_colors = np.array(lab_colors).squeeze((1,2)) # N, 3
            rgb_colors = np.array(rgb_colors).squeeze((1,2)) # N, 3
            # print(lab_colors.shape)
            similarity = lab_colors[:, None,:] - lab_colors[None, :, :]
            similarity_rgb = rgb_colors[:, None,:] - rgb_colors[None, :, :]
            similarity_l2 = np.linalg.norm(similarity, axis = 2, ord=2) # N, N
            # print(similarity_l2.shape)
            # assign color index based on similarity, if the similarity is less than thres, they are the same color
            color_index = np.zeros(len(lab_colors))
            for i in range(len(lab_colors)):
                if color_index[i] == 0:
                    color_index[i] = i+1
                for j in range(i+1, len(lab_colors)):
                    if similarity_l2[i,j] < thres:
                        color_index[j] = color_index[i]
            for i in range(len(lab_colors)):
                self.map_legend_color[map_name][list(self.dominate_colors[map_name].keys())[i]] = color_index[i]
                sim = similarity_rgb[i].copy().astype(float)
                sim[color_index==color_index[i]] = 0
                pos_sim = sim.copy()
                neg_sim = sim.copy()
                
                pos_sim[pos_sim <= 0] = np.inf
                neg_sim[neg_sim >= 0] = -np.inf
                # print(sim.shape)
                # print(pos_sim)
                lower, upper = np.min(pos_sim, axis = 0)/2, -np.max(neg_sim, axis = 0)/2

                lower[lower < lb] = lb
                lower[lower > ub] = ub
                upper[upper < lb] = lb
                upper[upper > ub] = ub
                # print(lower, upper)
                # assert False
                self.map_legend_color_bound[map_name][list(self.dominate_colors[map_name].keys())[i]] = (lower, upper)
            


        
    def update_percent_valid(self, percent_valid = 0.0):
        assert percent_valid <= 1 and percent_valid >= 0
        assert self.train
        print("warning: currently update percent valid is not doing anything, just keep valid percent to 1")
        self.percent_valid = 1.0 #percent_valid
        self.patches = generate_map_patches(self.mapsh5i, percent_valid)


    def __getitem__(self, index):
        # map_img = Image.open(self.map_path[index])
        a_channel = torch.tensor([0.0])
        map_name, layer_name, row, column = self.patches[index]
        map_img = self.mapsh5i[map_name].get_patch(row, column, map_name)
        legend_img = self.mapsh5i[map_name].get_legend(map_name, layer_name)
        if self.args.paint:
            # ori_legend = legend_img.copy()
            legend_img[:, :, :] = self.dominate_colors[map_name][layer_name]
            # reshape to map size
            legend_img = cv2.resize(legend_img, (map_img.shape[1], map_img.shape[0]))
            # ori_legend = cv2.resize(ori_legend, (map_img.shape[1], map_img.shape[0]))
        self.shape = [max(self.shape[0],legend_img.shape[0]),max(self.shape[1],legend_img.shape[1])]    
        # plot map, ori_legend, legend_img and subtracted image, each have caption with it
        # plt.figure(figsize=(20,20))
        # plt.subplot(1,4,1)
        # plt.imshow(map_img)
        # plt.title("map")
        # plt.subplot(1,4,2)
        # plt.imshow(ori_legend)
        # plt.title("original legend")
        # plt.subplot(1,4,3)
        # plt.imshow(legend_img)
        # plt.title("extracted color")
        # plt.subplot(1,4,4)
        # dif = map_img.astype(np.float32) - legend_img.astype(np.float32)
        # l2 = np.abs(dif).sum(axis=2)
        # plt.imshow(l2, cmap='gray')
        # plt.title("subtracted image")
        # plt.savefig(f"dif_vis/{index}.png")
        # if len(os.listdir("dif_vis")) >= 100:
        #     assert False
        if self.args.encode_boundary:
            upper, lower = self.map_legend_color_bound[map_name][layer_name]
            upper_image = np.ones_like(legend_img)
            lower_image = np.ones_like(legend_img)
            upper_image[:, :, :] = self.dominate_colors[map_name][layer_name] + upper
            lower_image[:, :, :] = self.dominate_colors[map_name][layer_name] - lower
            upper_image[upper_image > 255] = 255
            lower_image[lower_image < 0] = 0
            legend_imgs = [legend_img, upper_image, lower_image]
        else:
            legend_imgs = [legend_img]

        if self.args.line:
            seg_img = self.mapsh5i[map_name].get_patch(row, column, map_name, "boundary_line")
        else:
            seg_img = self.mapsh5i[map_name].get_patch(row, column, map_name, layer_name)
        
        layer_names = self.mapsh5i[map_name].get_layers(map_name)
        # print(layer_names)

        color_seg = np.zeros_like(seg_img)
        for i in range(len(layer_names)):
            if self.map_legend_color[map_name][layer_names[i]] == self.map_legend_color[map_name][layer_name]:
                color_seg += self.mapsh5i[map_name].get_patch(row, column, map_name, layer_names[i])
                

        if self.hist:
            map_img = self.apply_hist(map_img, map_name)
            legend_imgs = self.apply_hist(legend_imgs, map_name)
            

        map_img = Image.fromarray(np.uint8(map_img))
        # legend_img = Image.fromarray(np.uint8(legend_img))
        legend_imgs = [Image.fromarray(np.uint8(legend_img)) for legend_img in legend_imgs]
        seg_img = Image.fromarray(np.uint8(seg_img))
        # legend_img = Image.open(self.legend_path[index])
        # seg_img = Image.open(self.seg_path[index])
        seg_img = torch.tensor(np.array(seg_img).astype(float))
        seg_img = F.interpolate(seg_img.unsqueeze(0).unsqueeze(0), size=(self.resize_size,self.resize_size), mode='nearest').squeeze(0)
        seg_img = seg_img > 0.5
        seg_img = seg_img.type(torch.int32)

        color_seg = torch.tensor(np.array(color_seg).astype(float))
        color_seg = F.interpolate(color_seg.unsqueeze(0).unsqueeze(0), size=(self.resize_size,self.resize_size), mode='nearest').squeeze(0)
        color_seg = color_seg > 0.5
        color_seg = color_seg.type(torch.int32)

        
        if self.color == "lab":
            map_img = map_img.convert('LAB')
            legend_imgs = [legend_img.convert('LAB') for legend_img in legend_imgs]
        
        
        imgs = self.color_transform([map_img]+legend_imgs)
        map_img, legend_imgs = imgs[0], imgs[1:]
        map_img, masks = self.data_transforms(map_img, [seg_img, color_seg])
        legend_imgs = [self.legend_data_transforms(legend_img) for legend_img in legend_imgs]
        
        seg_img, color_seg = masks
        
        map_img = self.final_transforms(map_img)
        legend_imgs = [self.final_transforms(legend_img) for legend_img in legend_imgs]
        # seg_img, color_seg = self.mask_transforms(seg_img), self.mask_transforms(color_seg)
        # map_img = sincolor(map_img,self.pe)
        # legend_img = sincolor(legend_img,self.pe)

        if self.fuse: 
            gt = color_seg[0]
        else:
            gt = seg_img[0]
        legend_img = legend_imgs[0]
        if len(legend_imgs) > 1:
            a_channel = torch.cat(legend_imgs[1:], dim = 0)
        # print(a_channel.size())
        # assert False
        if index == 0:
            print("debug in data loader")
            print("train:", self.train)

            print(map_img.size(),legend_img.size(),seg_img.size())
            # assert False
        self.iterations += 1
        if self.iterations % (len(self.patches) * self.update_percent_valid_every) == 0 and self.train and self.percent_valid > 0.1:
            percent_valid = self.percent_valid - 0.01
            print("update percent valid to: ", percent_valid)
            self.update_percent_valid(percent_valid)
        

        return {
            "map_img": map_img,    # 3 H W
            "legend_img": legend_img, # 1 3 H W - > 3 H W 
            "a_channel": a_channel,
            "GT": gt, #.type(torch.LongTensor),    # 1 H W
            "color_seg": color_seg[0], #.type(torch.LongTensor),    # 1 H W
            "seg": seg_img[0], #.type(torch.LongTensor),    # 1 H W
            # "sup_msk": legend_seg, # 1 H W
            # "cls": 1,
            # 'qry_names': ["datasetid"+str(index)],
            'qry_ori_size': torch.tensor((256,256)),
            'superpixel': map_img
        }
    
    def __len__(self):
        return len(self.patches)

    def sample_tasks(self):
        pass
    def reset_sampler(self):
        pass

class eval_MAPData(data.Dataset):
    '''
    return:
        map_img: map image (3,resize_size,resize_size)
        legend_img: legend image (3,resize_size,resize_size)
        seg_img: segmentation image (3,resize_size,resize_size)
    '''
    def __init__(self, data_path=None, type="poly", range = None, resize_size = 256, mapName = '', viz = False, edge = False, legend = None):
        self.edge_only = edge
        self.resize_size = resize_size
        self.train = False # train
        self.viz = viz
        self.data_transforms = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]),
            ])
        self.mask_transforms = transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ToTensor()
        ])
        self.root = data_path
        self.type = type 
        # read in map data:
        mapPath = os.path.join(data_path, mapName)
        jsonPath = os.path.join(data_path, mapName[0:-4]+'.json')
        # gtPath = data_path + "" + mapName[:-4] #_Xjgb_poly.tif
        gtPath = data_path + "/../validation_rasters/" + mapName[:-4] #_Xjgb_poly.tif
        print(gtPath)
        
        self.map = cv2.imread(mapPath)
        if self.viz:
            plt.figure(figsize=(20,20))
            plt.imshow(self.map)
            plt.savefig("eval_viz/"+mapName+"_map.png")
            plt.close()
        with open(jsonPath, 'r') as f:
            jsonData = json.load(f)
        
        # only keep polygon label
        # read in legend img
        legend_name = []
        legend_img = []
        legend_gt = []

        # print(jsonData)
        for label_dict in jsonData['shapes']:
            
            if not label_dict['label'].endswith('_poly'):
                continue
            if not (legend is None) and label_dict['label'] != legend:
                continue
            print(label_dict['label'],legend)
            
            legend_name.append(label_dict['label'])
            gt_file = gtPath + "_" + label_dict['label'] + ".tif"
            print(gt_file)
            gt_im = cv2.imread(gt_file)
            try:
                legend_gt.append(gt_im[:,:,0])
            except:
                print("No GT found, using all zero")
                legend_gt.append(np.zeros_like(self.map[:,:,0]))
            point_coord = label_dict['points']
            flatten_list = list(chain.from_iterable(point_coord))
            if point_coord[0][0] >= point_coord[1][0] or point_coord[0][1] >= point_coord[1][1] or (len(flatten_list)!=4):
                x_coord = [x[0] for x in point_coord]
                y_coord = [x[1] for x in point_coord]
                x_low, y_low, x_hi, y_hi = int(min(x_coord)), int(min(y_coord)), int(max(x_coord)), int(max(y_coord))

            else: x_low, y_low, x_hi, y_hi = [int(x) for x in flatten_list]
            legend_coor =  [(x_low, y_low), (x_hi, y_hi)]
            shift_pixel  = 4
            im_crop = self.map[y_low+shift_pixel:y_hi-shift_pixel, x_low+shift_pixel:x_hi-shift_pixel] # need to resize
            im_crop_resize = cv2.resize(im_crop, dsize=(256,256), interpolation=cv2.INTER_CUBIC)
            legend_img.append(im_crop_resize.transpose(0,1,2))

        self.legend_name = legend_name # [:2]
        self.legend_img = legend_img # [:2]
        self.legend_gt = legend_gt # [:2]
        self.map_patches, self.shift_coord, self.cut_shape,self.patch_shape = cut_map(self.map)
        self.map_mask_im = map_mask(self.map)
        # print("debug: ", np.shape(legend_gt),np.min(legend_gt),np.max(legend_gt))

        # self.w =
        self.pe = positionalencoding1d(4, 256)
        print("when init dataset:", np.shape(legend_name), np.shape(legend_img), np.shape(legend_gt),np.shape(self.map_patches))
        

    def __getitem__(self, index):
        # return {}
        # map_img = Image.open(self.map_path[index])
        # legend_img = Image.open(self.legend_path[index])
        # seg_img = Image.open(self.seg_path[index])
        # map_shape = np.shape(self.map)
        patch_idx = index %  len(self.map_patches)
        
        legend_idx = index // len(self.map_patches)
        
        map_img = self.map_patches[patch_idx]
        # deep copy map_img to avoid change the original map
        raw_img = copy.deepcopy(map_img)
        super_pixel_img = cv2.bilateralFilter(raw_img, 5, 75, 75)
        super_pixel_transform = transforms.Compose([
            flow_transforms.ArrayToTensor(),
            transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
            transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
        ])
        super_pixel_img = super_pixel_transform(super_pixel_img)

        legend_img = self.legend_img[legend_idx]
        seg_img = np.zeros((256,256)) #self.legend_gt[patch_idx,legend_idx]

        seg_img = torch.tensor(np.array(seg_img).astype(float))
        seg_img = F.interpolate(seg_img.unsqueeze(0).unsqueeze(0), size=(self.resize_size,self.resize_size), mode='bilinear', align_corners=False).squeeze(0)
        seg_img = seg_img > 0.5
        seg_img = seg_img.type(torch.int64)
        # seg_img = seg_img.type(torch.float)

        # seg_img = (seg_img > 0.0001).type(torch.float)
        # print("debug:", np.shape(map_img),np.shape(legend_img))
        if self.edge_only:
            im = np.array(map_img)
            # cv2.imwrite(str(index)+"_map_patches.png", im)
            # print("debug for edge:",im.shape) # 256 256 3
            # pad im with 0
            # im = np.pad(im, ((1, 1), (1, 1), (0, 0)), mode='constant')
            # edges = im[1:-1,1:-1] * 8  - im[0:-2, 1:-1] - im[2:, 1:-1] - im[1:-1, 0:-2] - im[1:-1, 2:] - im[0:-2, 0:-2] - im[2:, 2:] - im[0:-2, 2:] - im[2:, 0:-2]
            kernel = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]]) 
            im = np.sum(im, axis = 2)
            edges = np.zeros_like(im)
            edges = scipy.signal.convolve2d(im,kernel)
            edges = np.absolute(edges)
            
            # save edges to png for debug
            # cv2.imwrite(str(index)+"_edges.png", (edges/np.max(edges)*255).astype(np.uint8) )
            edges = torch.from_numpy(edges[1:-1,1:-1].astype(np.float32)).unsqueeze(0) #  1 256 1256
            # edges = edges
        else:
            edges = None

        map_img = self.data_transforms(Image.fromarray(map_img))
        legend_img = self.data_transforms(Image.fromarray(legend_img))# .unsqueeze(0)

        
        if self.train and random.random() < self.color_jitter_rate:
            hue_factor = float(torch.empty(1).uniform_(-0.5, 0.5))
            map_img = aug_f.adjust_hue(map_img, hue_factor)
            legend_img = aug_f.adjust_hue(legend_img, hue_factor)

        if self.edge_only:
            map_img = edges 

        legend_seg = seg_img.clone()
        legend_seg = legend_seg * 0 + 1

        # map_img = sincolor(map_img,self.pe)
        # legend_img = sincolor(legend_img,self.pe)

        # print("mean of msk",torch.mean(seg_img.type(torch.LongTensor).type(torch.float)))
        # print("debug:", map_img.size(),legend_img.size(),seg_img.size(),legend_seg.size())
        return {
            "map_img": map_img,    # 3 H W
            "legend_img": legend_img, # 1 3 H W -> 3 H W
            "GT": seg_img[0], #.type(torch.LongTensor),    # 1 H W
            # "sup_msk": legend_seg, # 1 H W
            # "cls": 1,
            'qry_names': ["datasetid"+str(index)],
            'ori_size': torch.tensor((256,256)),
            'superpixel': super_pixel_img,
        }

    def metrics(self,preds,thres = [0.5]):
        shape = np.shape(preds)
        preds = preds.reshape(len(self.legend_name),self.patch_shape[0],self.patch_shape[1],1,shape[-1],shape[-1],1)
        preds_max = np.array([np.max(preds, axis=0)]*len(self.legend_name)) - 0.00001
        preds =  preds > preds_max
        final_dice_thes = []
        print("done with pre")
        
        # from multiprocessing import Pool, Manager
        # manager = Manager()
        # shared_list = manager.list([preds, preds_max, self.legend_gt , self.legend_name, self.cut_shape, self.map_mask_im, self.shift_coord, False, thres]) 
        
        # print("launching pool")
        # with Pool(processes=1) as pool:
        #     p = pool.map(unpatch_get_dice,
        #                     [ (shared_list, i) for i in range(len(thres))] 
        #                 )
        # print(p)
        final_dice_thes = []
        for thre in thres:
            item = (preds, preds_max, self.legend_gt , self.legend_name, self.cut_shape, self.map_mask_im, self.shift_coord, self.viz, thre)
            curdice = unpatch_get_dice(item)
            final_dice_thes.append(curdice)
        print(final_dice_thes)
        return final_dice_thes

    def __len__(self):
        return len(self.map_patches) * len(self.legend_name)

    def sample_tasks(self):
        pass
    def reset_sampler(self):
        pass

def map_mask(imarray):
    gray = cv2.cvtColor(imarray, cv2.COLOR_BGR2GRAY)  # greyscale image
    # Detect Background Color
    pix_hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    background_pix_value = np.argmax(pix_hist, axis=None)

    # Flood fill borders
    height, width = gray.shape[:2]
    corners = [[0,0],[0,height-1],[width-1, 0],[width-1, height-1]]
    for c in corners:
        cv2.floodFill(gray, None, (c[0],c[1]), 255)

    # AdaptiveThreshold to remove noise
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    # Edge Detection
    thresh_blur = cv2.GaussianBlur(thresh, (11, 11), 0)
    canny = cv2.Canny(thresh_blur, 0, 200)
    canny_dilate = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # Keeping only the largest detected contour.
    contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]

    wid, hight = imarray.shape[0], imarray.shape[1]
    mask = np.zeros([wid, hight])
    mask = cv2.fillPoly(mask, pts=[contour], color=(1,1,1)).astype(int)
    
    
    return mask

