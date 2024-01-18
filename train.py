from ultralytics import YOLO
import torchvision 
from PIL import Image
# torchvision.ops.nms(1,1,1)
# Load a model
model = YOLO("oneshot-yolov8n.yaml", task = "oneshot-detect")  # build a new model from scratch
model.load('yolov8n.pt')
# model = YOLO("yolov8n.yaml")  # build a new model from scratch
# model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("/media/jiahua/FILE/uiuc/NCSA/DARPA_torch/SAM/runs/detect/train4/weights/best.pt")

# Use the model

model.train(data="config/pt.yaml", epochs=200)  # train the model
# model.val(data="pt.yaml")

# metrics = model.val()  # evaluate model performance on the validation set
# model.eval()
# image = Image.open("/media/jiahua/FILE/uiuc/NCSA/all_patched/all_patched_data/training/point/map_patches/CO_HandiesPeak_451002_1955_24000_geo_mosaic_3_pt_7_13.png")
# results = model.predict(image)  # predict on an image
# # results.show()
# print(results[0].boxes)
# path = model.export(format="onnx")  # export the model to ONNX format