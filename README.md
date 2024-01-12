## Install
We suggest using conda for virtual environments.
Please run the following command to create the environment:
```
conda env create -f environment.yml
```

then, activate the environment and install the modified one-shot Yolo
```
conda activate yolo_det
pip install -e ./yolo_src/
```

## Run
To use the Map Inference Tool, run the script with the necessary arguments:

```
python inference.py --mapPath "/path/to/your/map.hdf5" --outputPath "/path/to/output/directory" --modelPath "/path/to/your/model.pt"
```

Make sure to install pytorch and lightning. Other packages are also needed, including einops, h5py, matplotlib, pillow, opencv-python.
