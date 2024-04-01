import glob
import os
import numpy as np
from h5image import H5Image
from random import sample, choice
import cv2 
def setup():
    mapsh5i = {}
    exclude_map = set(['CO_SanchezRes','CA_NV_LasVegas','CO_VailE100K','CA_Cambria','ID_LakeWalcott','AZ_PioRico_Nogales'])
    for file in glob.glob("/projects/bbym/shared/data/commonPatchData/256/15/*.hdf5"):
        mapname = os.path.basename(file).replace(".hdf5", "")
        if mapname not in exclude_map:
            mapsh5i[mapname] = H5Image(file, "r")
    print("Setup complete. Number of maps loaded:", len(mapsh5i))
    return mapsh5i

def create_data_and_labels(map_patches, mapsh5i):
    data_array = []
    label_array = []

    for map_name, layer_name, row, column in map_patches:
        map_patch = mapsh5i[map_name].get_patch(row, column, map_name)
        try:
            legend_patch = mapsh5i[map_name].get_legend(map_name, layer_name)
        except:
            print(f"Error: {map_name}, {layer_name}, {row}, {column}")
            continue
        
        label_patch = mapsh5i[map_name].get_patch(row, column, map_name, layer_name)
        legend_patch = cv2.resize(legend_patch, (map_patch.shape[1], map_patch.shape[0]))
        label_array.append(label_patch)
        data_patch = np.concatenate([legend_patch, map_patch], axis=-1)
        data_array.append(data_patch)

    return data_array, label_array

def main(): 
    mapsh5i = setup()
    maps = list(mapsh5i.keys())
    map_patches = []
    for map_name in maps:
        patches = mapsh5i[map_name].get_patches(map_name)
        cor = mapsh5i[map_name].get_map_corners(map_name)

        valid_patches = [(map_name, layer, x, y) for layer in patches if layer.endswith('_pt') for x, y in patches[layer]]
        random_patches = [(map_name,choice(list(patches.keys())), 
                        sample(range(cor[0][0], cor[1][0]), 1)[0], 
                        sample(range(cor[0][1], cor[1][1]), 1)[0]) 
                        for _ in range(len(valid_patches))]

        n = len(valid_patches)
        percent_valid = 0.75
        # One-liner to get the samples
        map_patches = map_patches + sample(valid_patches, int(n * percent_valid)) + sample(random_patches, int(n * (1-percent_valid)))
    
    data_array, label_array = create_data_and_labels(map_patches, mapsh5i)

    return data_array, label_array

if __name__ == "__main__":
    main()