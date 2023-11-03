import json
import logging
import math
import os
import os.path
import cv2
import h5py
import numpy as np


class H5Image:
    """Class to read and write images to HDF5 file"""

    # initialize the class
    def __init__(self, h5file, mode='r', compression="lzf", patch_size=256, patch_border=3):
        """
        Create a new H5Image object.
        :param h5file: filename on disk
        :param mode: set to 'r' for read-only, 'w' for write, 'a' for append
        :param compression: compression type, None for no compression
        :param patch_size: size of patch, used to crop image and calculate good patches
        :param patch_border: border around patch, used to crop image and calculate good patches
        """
        self.h5file = h5file
        self.mode = mode
        self.compression = compression
        self.patch_size = patch_size
        self.patch_border = patch_border
        self.h5f = h5py.File(h5file, mode)

    # close the file
    def close(self):
        """
        Close the file
        """
        self.h5f.close()

    def __str__(self):
        """
        String representation of the object
        :return: string representation
        """
        return f"H5Image(filename={self.h5file}, mode={self.mode}, maps={len(self.get_maps())})"

    def _add_image(self, filename, name, group):
        """
        Helper function to add an image to the file
        :param filename: image on disk
        :param name: name of image in hdf5 file
        :param group: parent folder of image
        :return: dataset of image loaded (numpy array)
        """
        image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        dset = group.create_dataset(name=name, data=image, shape=image.shape, compression=self.compression)
        dset.attrs.create('CLASS', 'IMAGE', dtype='S6')
        dset.attrs.create('IMAGE_MINMAXRANGE', [0, 255], dtype=np.uint8)
        if len(image.shape) == 3 and image.shape[2] == 3:
            dset.attrs.create('IMAGE_SUBCLASS', 'IMAGE_TRUECOLOR', dtype='S16')
        elif len(image.shape) == 2 or image.shape[2] == 1:
            dset.attrs.create('IMAGE_SUBCLASS', 'IMAGE_GRAYSCALE', dtype='S15')
        else:
            raise Exception("Unknown image type")
        dset.attrs.create('IMAGE_VERSION', '1.2', dtype='S4')
        dset.attrs.create('INTERLACE_MODE', 'INTERLACE_PIXEL', dtype='S16')
        return dset

    # add an image to the file
    def add_image(self, filename, folder="", mapname=""):
        """
        Add a set of images to the file. The filenname is assumed to be json and is
        used to load all images with the same prefix. The map is assumed to have the
        same prefix as the json file (but ending with .tif). The json file lists all
        layers that are added as well. The json file is attached to the group.
        :param filename: the json file to load
        :param folder: directory where to find the json file
        :param mapname: the name of the map, if empty the name of the json file is used
        """
        # make sure file is writeable
        if self.mode == 'r':
            raise Exception("Cannot add image to read-only file")

        # check json file
        if not filename.endswith(".json"):
            raise Exception("Need to pass json file")
        jsonfile = os.path.join(folder, filename)
        if not os.path.exists(jsonfile):
            raise Exception("File not found")
        prefix = jsonfile.replace(".json", "")
        if mapname == "":
            mapname = os.path.basename(prefix)

        # check image file exists
        tiffile = f"{prefix}.tif"
        if not os.path.exists(tiffile):
            tiffile = f"{prefix}.tiff"
        if not os.path.exists(tiffile):
            raise Exception("Image file not found")

        # load json
        json_data = json.load(open(jsonfile))
        if 'shapes' not in json_data or len(json_data['shapes']) == 0:
            raise Exception("No shapes found")

        # create the group
        group = self.h5f.create_group(mapname)
        group.attrs.update({'json': json.dumps(json_data)})

        # load image
        dset = self._add_image(tiffile, "map", group)
        w = math.ceil(dset.shape[0] / 256)
        h = math.ceil(dset.shape[1] / 256)

        # loop through shapes
        all_patches = {}
        layers_patch = {}
        for shape in json_data['shapes']:
            label = shape['label']
            patches = []
            try:
                dset = self._add_image(f"{prefix}_{label}.tif", label, group)
                for x in range(w):
                    for y in range(h):
                        rgb = self._crop_image(dset,
                                               x * self.patch_size - self.patch_border,
                                               y * self.patch_size - self.patch_border,
                                               (x+1) * self.patch_size + self.patch_border,
                                               (y+1) * self.patch_size + self.patch_border)
                        if np.average(rgb, axis=(0, 1)) > 0:
                            patches.append((x, y))
                            layers_patch.setdefault(f"{x}_{y}", []).append(label)
                dset.attrs.update({'patches': json.dumps(patches)})
                all_patches[label] = patches
            except ValueError as e:
                logging.warning(f"Error loading {label} : {e}")
        valid_patches = [[int(k.split('_')[0]), int(k.split('_')[1])] for k in layers_patch.keys()]
        r1 = min(valid_patches, key=lambda value: int(value[0]))[0]
        r2 = max(valid_patches, key=lambda value: int(value[0]))[0]
        c1 = min(valid_patches, key=lambda value: int(value[1]))[1]
        c2 = max(valid_patches, key=lambda value: int(value[1]))[1]
        group.attrs.update({'patches': json.dumps(all_patches)})
        group.attrs.update({'layers_patch': json.dumps(layers_patch)})
        group.attrs.update({'valid_patches': json.dumps(valid_patches)})
        group.attrs.update({'corners': [[r1, c1], [r2, c2]]})

    # get list of all maps
    def get_maps(self):
        """
        Returns a list of all maps in the file.
        :return: list of map names
        """
        return list(self.h5f.keys())

    # get map by index
    def get_map(self, mapname):
        """
        Returns the map as a numpy array.
        :param mapname: the name of the map
        :return: image as numpy array
        """
        return self.h5f[mapname]['map']

    # return map size
    def get_map_size(self, mapname):
        """
        Returns the size of the map.
        :param mapname: the name of the map
        :return: size of the map
        """
        return self.h5f[mapname]['map'].shape

    def get_map_corners(self, mapname):
        """
        Returns the bounds of the map.
        :param mapname: the name of the map
        :return: bounds of the map
        """
        return list(self.h5f[mapname].attrs['corners'])

    # get list of all layers for map
    def get_layers(self, mapname):
        """
        Returns a list of all layers for a map.
        :param mapname: the name of the map
        :return: list of layer names
        """
        layers = list(self.h5f[mapname].keys())
        layers.remove('map')
        return layers

    def get_layer(self, mapname, layer):
        """
        Returns the layer as a numpy array.
        :param mapname: the name of the map
        :param layer: the name of the layer
        :return: image as numpy array
        """
        return self.h5f[mapname][layer]

    def get_patches(self, mapname, by_location=False):
        """
        Returns a list of all patches for a map. The patches are grouped by layer. If by_location is
        False it returns a dict of layers, each with a list of patches (as arrays). If by location
        the result will be a dict of patches (col-row) , each with a list of layers.
        patches for a map
        :param mapname: the name of the map
        :param by_location: if True, return a dictionary with locations as keys and layers as values
        :return: list of patches
        """
        if by_location:
            return json.loads(self.h5f[mapname].attrs['layers_patch'])
        else:
            return json.loads(self.h5f[mapname].attrs['patches'])

    def get_valid_patches(self, mapname):
        """
        Returns a list of all valid patches for a map. A valid patch is a patch that has
        at least one layer with a value > 0.
        :param mapname: the name of the map
        :return: list of valid patches
        """
        return json.loads(self.h5f[mapname].attrs['valid_patches'])

    def get_patches_for_layer(self, mapname, layer):
        """
        Returns a list of all patches for a layer.
        :param mapname: the name of the map
        :param layer: the name of the layer
        :return: list of patches
        """
        return json.loads(self.h5f[mapname][layer].attrs['patches'])

    def get_layers_for_patch(self, mapname, row, col):
        """
        Returns a list of all layers for a patch.
        :param mapname: the name of the map
        :param row: the row of the patch
        :param col: the column of the patch
        :return: list of layers
        """
        return json.loads(self.h5f[mapname].attrs['layers_patch']).get(f"{row}_{col}", [])

    # crop image, this assumes x1 < x2 and y1 < y2
    @staticmethod
    def _crop_image(dset, x1, y1, x2, y2):
        """
        Helper function to crop an image.
        :param dset: the hdf5 dataset to crop
        :param x1: upper left x coordinate
        :param y1: upper left y coordinate
        :param x2: lower right x coordinate
        :param y2: lower right y coordinate
        :return: cropped image as numpy array
        """
        if x1 < 0:
            x1 = 0
        w = abs(x2 - x1)
        if x2 > dset.shape[0]:
            w = w - (x2 - dset.shape[0])
            x2 = dset.shape[0]
        if y1 < 0:
            y1 = 0
        h = abs(y2 - y1)
        if x1 < 0:
            w = w - x1
            x1 = 0
        if y2 > dset.shape[1]:
            h = h - (y2 - dset.shape[1])
            y2 = dset.shape[1]
        if len(dset.shape) == 3 and dset.shape[2] == 3:
            rgb = np.zeros((w, h, 3), dtype=np.uint8)
        else:
            rgb = np.zeros((w, h), dtype=np.uint8)
        dset.read_direct(rgb, np.s_[x1:x2, y1:y2])
        return rgb

    # get legend from map
    def get_legend(self, mapname, layer):
        """
        Returns the cropped image of the legend in the map.
        :param mapname: the name of the map
        :param layer: the name of the layer
        :return: cropped image of legend
        """
        json_data = json.loads(self.h5f[mapname].attrs['json'])
        for shape in json_data['shapes']:
            if shape['label'] == layer:
                points = shape['points']
                w = abs(points[1][1] - points[0][1])
                h = abs(points[1][0] - points[0][0])
                x1 = min(points[0][1], points[1][1])
                y1 = min(points[0][0], points[1][0])
                x2 = x1 + w
                y2 = y1 + h
                # points in array are floats
                return self._crop_image(self.h5f[mapname]['map'], int(x1), int(y1), int(x2), int(y2))
        return None

    # get patch by index
    # row and col are 0 based
    def get_patch(self, row, col, mapname, layer="map"):
        """
        Returns the cropped image of the patch.
        :param row: the row of the patch
        :param col: the column of the patch
        :param mapname: the name of the map
        :param layer: the name of the layer
        :return: cropped image of patch as a numpy array
        """
        if row < 0 or col < 0:
            raise Exception("Invalid index")
        if row == 0:
            x1 = 0
            x2 = self.patch_size + self.patch_border
        else:
            x1 = (row * self.patch_size) - self.patch_border
            x2 = ((row + 1) * self.patch_size) + self.patch_border
        if col == 0:
            y1 = 0
            y2 = self.patch_size + self.patch_border
        else:
            y1 = (col * self.patch_size) - self.patch_border
            y2 = ((col + 1) * self.patch_size) + self.patch_border
        return self._crop_image(self.h5f[mapname][layer], x1, y1, x2, y2)
