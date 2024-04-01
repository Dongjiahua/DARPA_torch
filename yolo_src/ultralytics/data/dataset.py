# Ultralytics YOLO 🚀, AGPL-3.0 license
import contextlib
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from h5image import H5Image
from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, colorstr, is_dir_writeable, remove_colorstr
import math
from .augment import Compose, Format, Instances, LetterBox, classify_albumentations, classify_transforms, v8_transforms, ori_v8_transforms
from .base import BaseDataset
from .utils import HELP_URL, LOGGER, get_hash, img2label_paths, img2legend_paths, verify_image, verify_image_label, h5setup, verify_h5image_label, verify_h5image_label_multiclass
from PIL import Image
from copy import deepcopy
import random
# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = '1.0.3'
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM

class YOLODataset(BaseDataset):
    """
    Dataset class for loading object detection and/or segmentation labels in YOLO format.

    Args:
        data (dict, optional): A dataset YAML dictionary. Defaults to None.
        use_segments (bool, optional): If True, segmentation masks are used as labels. Defaults to False.
        use_keypoints (bool, optional): If True, keypoints are used as labels. Defaults to False.

    Returns:
        (torch.utils.data.Dataset): A PyTorch dataset object that can be used for training an object detection model.
    """

    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs):
        """Initializes the YOLODataset with optional configurations for segments and keypoints."""
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.data = data
        assert not (self.use_segments and self.use_keypoints), 'Can not use both segments and keypoints.'
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path('./labels.cache')):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
        total = len(self.im_files)
        nkpt, ndim = self.data.get('kpt_shape', (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_label,
                                iterable=zip(self.im_files, self.label_files, repeat(self.prefix),
                                             repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
                                             repeat(ndim)))
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x['labels'].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format='xywh'))
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache['version'] == DATASET_CACHE_VERSION  # matches current version
            assert cache['hash'] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            TQDM(None, desc=self.prefix + d, total=n, initial=n)  # display results
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))  # display warnings

        # Read cache
        [cache.pop(k) for k in ('hash', 'version', 'msgs')]  # remove items
        labels = cache['labels']
        if not labels:
            LOGGER.warning(f'WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}')
        self.im_files = [lb['im_file'] for lb in labels]  # update im_files

        # Check if the dataset is all boxes or all segments
        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        if len_cls == 0:
            LOGGER.warning(f'WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}')
        return labels

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms

    def close_mosaic(self, hyp):
        """Sets mosaic, copy_paste and mixup options to 0.0 and builds transformations."""
        hyp.mosaic = 0.0  # set mosaic ratio=0.0
        hyp.copy_paste = 0.0  # keep the same behavior as previous v8 close-mosaic
        hyp.mixup = 0.0  # keep the same behavior as previous v8 close-mosaic
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        """Custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # We can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop('bboxes')
        segments = label.pop('segments')
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img' or k == 'legend_img':
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch

class OneShotDataset(YOLODataset):
    def __init__(self, *args, data=None, use_segments=False, use_keypoints=False, **kwargs):
        super().__init__(*args, data=data, use_segments=use_segments, use_keypoints=use_keypoints, **kwargs)
        self.get_legends()
        self.legend_t = LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)


    
    def get_legends(self):
        self.legend_files = img2legend_paths(self.im_files)
        for t, legend in enumerate(self.legend_files):
            if not legend.endswith("pt.png"):
                print(f"img_name:{self.im_files[t]}, legend_name:{legend.split('/')[-1]}")
        
    
    def get_legend_img(self,index, imshape):
        h, w = imshape
        f = self.legend_files[index]
        im = cv2.imread(f)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)

        im = torch.from_numpy(im).permute(2,0,1)
        label = dict()
        label['legend_img'] = im 
        return label
    
    def update_labels_info(self,label):
        label = super().update_labels_info(label)
        label.update(self.get_legend_img(label['batch_idx'], label['img'].shape[:2]))
  
        return label

class OneShotHDF5(torch.utils.data.Dataset):
    def __init__(self, img_path,
                 imgsz=640,
                 cache=False,
                 augment=True,
                 hyp=DEFAULT_CFG,
                 prefix='',
                 rect=False,
                 batch_size=16,
                 stride=32,
                 pad=0.5,
                 single_cls=False,
                 classes=None,
                 fraction=1.0, data=None, use_segments=False, use_keypoints=False, **kwargs):
        super().__init__()
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        self.data = data
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.h5_files = h5setup(img_path)
        self.files = self.build_files(self.h5_files)
        self.ni = len(self.files)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        self.labels = self.get_labels()
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

        # self.get_legends()
        self.legend_t = LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)
        
    def get_labels(self):
        """Returns dictionary of labels for YOLO training."""
        cache_path = Path(self.img_path).with_suffix('.cache')
        try:
            cache, exists = load_dataset_cache_file(cache_path), True  # attempt to load a *.cache file
            assert cache['version'] == DATASET_CACHE_VERSION  # matches current version
            assert cache['hash'] == get_hash(["_".join([str(x) for x in file]) for file in self.files])  # identical hash
            # assert False
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops
        labels = cache["labels"]
        return labels
    

    def cache_labels(self, path=Path('./labels.cache')):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
        total = len(self.files)
        nkpt, ndim = self.data.get('kpt_shape', (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_h5image_label,
                                iterable=zip(self.files, repeat(self.h5_files), 
                                             repeat(self.use_keypoints), repeat(len(self.data['names'])), repeat(nkpt),
                                             repeat(ndim)))
            pbar = TQDM(results, desc=desc, total=total)
            for lb, keypoint, shape, l_name in pbar:
                x['labels'].append(
                    dict(
                        im_file = l_name,
                        cls=lb[:, 0:1],  # n, 1
                        bboxes=lb[:, 1:],  # n, 4
                        shape = shape,
                        keypoints=keypoint,
                        normalized=True,
                        bbox_format='xywh'))

            pbar.close()
        files_strs = ["_".join([str(x) for x in file]) for file in self.files]
        x['hash'] = get_hash(files_strs)
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        map_name, legend, row, col = self.files[i]
        im = self.h5_files[map_name].get_patch(row, col,  map_name)
        
        h0, w0 = im.shape[:2]  # orig hw
        if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
            r = self.imgsz / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        # Add to buffer if training with augmentations
        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
            self.buffer.append(i)
            if len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

        return im, (h0, w0), im.shape[:2]


    
    def get_image_and_label(self, index):
        if index >= self.len_valids:
            index = random.randint(self.len_valids, len(self.files)-1)
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop('shape', None)  # shape is for rect, remove it
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
        label['batch_idx'] = index
        if self.rect:
            label['rect_shape'] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)
    
    def build_files(self, h5_files):
        
        maps = list(h5_files.keys())
        valids = []
        raw = []
        for map_name in maps:
            try:
                patches = h5_files[map_name].get_patches(map_name)
            except:
                print(f"Map {map_name} has no patches")
                continue
            try:
                for legend in [layer for layer in patches if layer.endswith('_pt') ]:
                    legend = h5_files[map_name].get_legend(map_name, legend)
            except:
                print(f"Map {map_name} has no legend: {legend}")
                continue
            try:
                cor = h5_files[map_name].get_map_corners(map_name)
                all_patches = [(map_name, layer, x, y) for layer in h5_files[map_name].get_layers(map_name) if layer.endswith('pt') for x in range(cor[0][0], cor[1][0]) for y in range(cor[0][1], cor[1][1])]
                raw.extend(all_patches)
            except:
                print(f"Map {map_name} has no corners")
           
            valid_patches = [(map_name, layer, x, y) for layer in patches if layer.endswith('_pt') for x, y in patches[layer]]
            valids.extend(valid_patches)
            self.len_valids = len(valids)
        
        return valids


        # if remove_colorstr(self.prefix)[:-2] == 'train':
        #     self.total_length = 1*self.len_valids
        #     return valids + raw
        # else:
        #     self.len_valids = len(raw)
        #     self.total_length = self.len_valids
        #     return raw


    
    def get_legend_img(self,index, imshape):
        map_name, legend, row, col = self.files[index]
        im = self.h5_files[map_name].get_legend(map_name, legend)
        h, w = imshape
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        # assert h==w
        # # create a image with 4 im 2x2, each rotate 90 degree
        # imo = np.zeros((h*2, w*2, 3), dtype=np.uint8)
        # imo[:h, :w] = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
        # imo[:h, w:] = cv2.rotate(im, cv2.ROTATE_180)
        # imo[h:, :w] = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # imo[h:, w:] = im
        # im = cv2.resize(imo, (h, w), interpolation=cv2.INTER_LINEAR)
        im = torch.from_numpy(im).permute(2,0,1)
        label = dict()
        label['legend_img'] = im 
        return label

    def super_labels_info(self, label):
        """Custom your label format here."""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # We can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop('bboxes')
        segments = []
        keypoints = label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img' or k == 'legend_img':
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch
    
    def update_labels_info(self,label):
        label = self.super_labels_info(label)
        label.update(self.get_legend_img(label['batch_idx'], label['img'].shape[:2]))
  
        return label

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms

    def __len__(self):
        """Returns the length of the labels list for the dataset."""
        return self.total_length


    def set_rectangle(self):
        """Sets the shape of bounding boxes for YOLO detections as rectangles."""
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop('shape') for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.files = [self.files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index):
        """Returns transformed label information for given index."""

        # print(self.transforms(self.get_image_and_label(index)).keys())
        # raise NotImplementedError
        return self.transforms(self.get_image_and_label(index))
    

class Multiclass_h5(OneShotHDF5):
    def __init__(self, img_path, imgsz=640, cache=False, augment=True, hyp=DEFAULT_CFG, prefix='', rect=False, batch_size=16, stride=32, pad=0.5, single_cls=False, classes=None, fraction=1, data=None, use_segments=False, use_keypoints=False, **kwargs):
        self.label_names_path = "/u/jiahuad2/codebase/yolo_inference/yolo_data/legend_name2class.json"
        import json
        label_names = json.load(open(self.label_names_path))
        self.label_to_class = {name: int(label_names[i]["id"]) for i in range(len(label_names)) for name in label_names[i]["names"]}
        super().__init__(img_path, imgsz, cache, augment, hyp, prefix, rect, batch_size, stride, pad, single_cls, classes, fraction, data, use_segments, use_keypoints, **kwargs)

    def __getitem__(self, index):
        """Returns transformed label information for given index."""

        # print(self.transforms(self.get_image_and_label(index)).keys())
        # raise NotImplementedError
        label =self.get_image_and_label(index)
        legend_img = label['legend_img']
        result_label = self.transforms(label)
        result_label['legend_img'] = legend_img
        return result_label
    
    def getitem(self, index):
        """Returns transformed label information for given index."""

        # print(self.transforms(self.get_image_and_label(index)).keys())
        # raise NotImplementedError
        return self.transforms(self.get_image_and_label(index))   
    
    def get_legend_img(self,imshape):
        im = torch.zeros(3, imshape[0], imshape[1])

        return im
    
    def update_labels_info(self,label):
        
        label = self.super_labels_info(label)
        label['legend_img'] = self.get_legend_img(label['img'].shape[:2])

        return label
    
    def load_image(self, i, rect_mode=True):
        """Loads 1 image from dataset index 'i', returns (im, resized hw)."""
        map_name,  row, col = self.files[i]
        im = self.h5_files[map_name].get_patch(row, col,  map_name)
        
        h0, w0 = im.shape[:2]  # orig hw
        if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
            r = self.imgsz / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
            im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)

        # Add to buffer if training with augmentations
        if self.augment:
            self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
            self.buffer.append(i)
            if len(self.buffer) >= self.max_buffer_length:
                j = self.buffer.pop(0)
                self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

        return im, (h0, w0), im.shape[:2]
    
    def build_files(self, h5_files):
        
        maps = list(h5_files.keys())
        valids = []
        raw = []
        for map_name in maps:
            try:
                patches = h5_files[map_name].get_patches(map_name)
            except:
                print(f"Map {map_name} has no patches")
                continue
            try:
                for legend in [layer for layer in patches if layer.endswith('_pt') ]:
                    legend = h5_files[map_name].get_legend(map_name, legend)
            except:
                print(f"Map {map_name} has no legend: {legend}")
                continue
            try:
                cor = h5_files[map_name].get_map_corners(map_name)
                all_patches = [(map_name, layer, x, y) for layer in h5_files[map_name].get_layers(map_name) if layer.endswith('pt') for x in range(cor[0][0], cor[1][0]) for y in range(cor[0][1], cor[1][1])]
                raw.extend(all_patches)
            except:
                print(f"Map {map_name} has no corners")
           
            valid_patches = [(map_name, layer, x, y) for layer in patches if layer.endswith('_pt') for x, y in patches[layer]]
            valids.extend(valid_patches)
            self.len_valids = len(valids)
        modified_list = [(map_name, x, y) for map_name, _, x, y in valids]
        unique_list = list(set(modified_list))
        self.total_length = len(unique_list)
        return sorted(unique_list)

    def cache_labels(self, path=Path('./labels.cache')):
        """
        Cache dataset labels, check images and read shapes.

        Args:
            path (Path): path where to save the cache file (default: Path('./labels.cache')).
        Returns:
            (dict): labels.
        """
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
        total = len(self.files)
        nkpt, ndim = self.data.get('kpt_shape', (0, 0))
        if self.use_keypoints and (nkpt <= 0 or ndim not in (2, 3)):
            raise ValueError("'kpt_shape' in data.yaml missing or incorrect. Should be a list with [number of "
                             "keypoints, number of dims (2 for x,y or 3 for x,y,visible)], i.e. 'kpt_shape: [17, 3]'")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_h5image_label_multiclass,
                                iterable=zip(self.files, repeat(self.h5_files), repeat(self.label_to_class),
                                             repeat(self.use_keypoints), repeat(len(self.label_to_class)), repeat(nkpt),
                                             repeat(ndim)))
            pbar = TQDM(results, desc=desc, total=total)
            for lb, keypoint, shape, l_name in pbar:
                x['labels'].append(
                    dict(
                        im_file = l_name,
                        cls=lb[:, 0:1],  # n, 1
                        bboxes=lb[:, 1:],  # n, 4
                        shape = shape,
                        keypoints=keypoint,
                        normalized=True,
                        bbox_format='xywh'))

            pbar.close()
        files_strs = ["_".join([str(x) for x in file]) for file in self.files]
        x['hash'] = get_hash(files_strs)
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def build_transforms(self, hyp=None):
        """Builds and appends transforms to the list."""
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            transforms = ori_v8_transforms(self, self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(bbox_format='xywh',
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True,
                   mask_ratio=hyp.mask_ratio,
                   mask_overlap=hyp.overlap_mask))
        return transforms

    @staticmethod
    def collate_fn(batch):
        """Collates data samples into batches."""
        new_batch = {}
        keys = batch[0].keys()
        # print(f"before collate_fn: {keys}")
        if "legend_img" not in keys:
            assert False
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img' or k == 'legend_img':
                value = torch.stack(value, 0)
            if k in ['masks', 'keypoints', 'bboxes', 'cls']:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i  # add target image index for build_targets()
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        # print(f"after collate_fn: {new_batch.keys()}")
        return new_batch

    def get_image_and_label(self, index):
        if index >= self.len_valids:
            index = random.randint(self.len_valids, len(self.files)-1)
        """Get and return label information from the dataset."""
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop('shape', None)  # shape is for rect, remove it
        label['img'], label['ori_shape'], label['resized_shape'] = self.load_image(index)
        label['ratio_pad'] = (label['resized_shape'][0] / label['ori_shape'][0],
                              label['resized_shape'][1] / label['ori_shape'][1])  # for evaluation
        label['batch_idx'] = index
        if self.rect:
            label['rect_shape'] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)
    
# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLO Classification Dataset.

    Args:
        root (str): Dataset path.

    Attributes:
        cache_ram (bool): True if images should be cached in RAM, False otherwise.
        cache_disk (bool): True if images should be cached on disk, False otherwise.
        samples (list): List of samples containing file, index, npy, and im.
        torch_transforms (callable): torchvision transforms applied to the dataset.
        album_transforms (callable, optional): Albumentations transforms applied to the dataset if augment is True.
    """

    def __init__(self, root, args, augment=False, cache=False, prefix=''):
        """
        Initialize YOLO object with root, image size, augmentations, and cache settings.

        Args:
            root (str): Dataset path.
            args (Namespace): Argument parser containing dataset related settings.
            augment (bool, optional): True if dataset should be augmented, False otherwise. Defaults to False.
            cache (bool | str | optional): Cache setting, can be True, False, 'ram' or 'disk'. Defaults to False.
        """
        super().__init__(root=root)
        if augment and args.fraction < 1.0:  # reduce training fraction
            self.samples = self.samples[:round(len(self.samples) * args.fraction)]
        self.prefix = colorstr(f'{prefix}: ') if prefix else ''
        self.sp = prefix
        self.cache_ram = cache is True or cache == 'ram'
        self.cache_disk = cache == 'disk'
        self.samples = self.verify_images()  # filter out bad images
        self.samples = [list(x) + [Path(x[0]).with_suffix('.npy'), None] for x in self.samples]  # file, index, npy, im
        self.torch_transforms = classify_transforms(args.imgsz, rect=args.rect)
        self.album_transforms = classify_albumentations(
            augment=augment,
            size=args.imgsz,
            scale=(1.0 - args.scale, 1.0),  # (0.08, 1.0)
            hflip=args.fliplr,
            vflip=args.flipud,
            hsv_h=args.hsv_h,  # HSV-Hue augmentation (fraction)
            hsv_s=args.hsv_s,  # HSV-Saturation augmentation (fraction)
            hsv_v=args.hsv_v,  # HSV-Value augmentation (fraction)
            mean=(0.0, 0.0, 0.0),  # IMAGENET_MEAN
            std=(1.0, 1.0, 1.0),  # IMAGENET_STD
            auto_aug=False) if augment else None

    def __getitem__(self, i):
        """Returns subset of data and targets corresponding to given indices."""
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f), allow_pickle=False)
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))['image']
        else:
            sample = self.torch_transforms(im)
        return {'img': sample, 'cls': j}

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.samples)

    def verify_images(self):
        """Verify all images in dataset."""
        desc = f'{self.prefix}Scanning {self.root}...'
        path = Path(self.root).with_suffix('.cache')  # *.cache file path

        with contextlib.suppress(FileNotFoundError, AssertionError, AttributeError):
            cache = load_dataset_cache_file(path)  # attempt to load a *.cache file
            assert cache['version'] == DATASET_CACHE_VERSION  # matches current version
            assert cache['hash'] == get_hash([x[0] for x in self.samples])  # identical hash
            nf, nc, n, samples = cache.pop('results')  # found, missing, empty, corrupt, total
            if LOCAL_RANK in (-1, 0):
                d = f'{desc} {nf} images, {nc} corrupt'
                TQDM(None, desc=d, total=n, initial=n)
                if cache['msgs']:
                    LOGGER.info('\n'.join(cache['msgs']))  # display warnings
            return samples

        # Run scan if *.cache retrieval failed
        nf, nc, msgs, samples, x = 0, 0, [], [], {}
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image, iterable=zip(self.samples, repeat(self.prefix)))
            pbar = TQDM(results, desc=desc, total=len(self.samples))
            for sample, nf_f, nc_f, msg in pbar:
                if nf_f:
                    samples.append(sample)
                if msg:
                    msgs.append(msg)
                nf += nf_f
                nc += nc_f
                pbar.desc = f'{desc} {nf} images, {nc} corrupt'
            pbar.close()
        if msgs:
            LOGGER.info('\n'.join(msgs))
        x['hash'] = get_hash([x[0] for x in self.samples])
        x['results'] = nf, nc, len(samples), samples
        x['msgs'] = msgs  # warnings
        save_dataset_cache_file(self.prefix, path, x)
        return samples


def load_dataset_cache_file(path):
    """Load an Ultralytics *.cache dictionary from path."""
    import gc
    gc.disable()  # reduce pickle load time https://github.com/ultralytics/ultralytics/pull/1585
    cache = np.load(str(path), allow_pickle=True).item()  # load dict
    gc.enable()
    return cache


def save_dataset_cache_file(prefix, path, x):
    """Save an Ultralytics dataset *.cache dictionary x to path."""
    x['version'] = DATASET_CACHE_VERSION  # add cache version
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()  # remove *.cache file if exists
        np.save(str(path), x)  # save cache for next time
        path.with_suffix('.cache.npy').rename(path)  # remove .npy suffix
        LOGGER.info(f'{prefix}New cache created: {path}')
    else:
        LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):
    """
    Semantic Segmentation Dataset.

    This class is responsible for handling datasets used for semantic segmentation tasks. It inherits functionalities
    from the BaseDataset class.

    Note:
        This class is currently a placeholder and needs to be populated with methods and attributes for supporting
        semantic segmentation tasks.
    """

    def __init__(self):
        """Initialize a SemanticDataset object."""
        super().__init__()

# daaset  = OneShotHDF5("/u/jiahuad2/codebase/yolo_inference/config/h5_15.yaml")
    