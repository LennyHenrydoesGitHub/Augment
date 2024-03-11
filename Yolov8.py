# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:51:42 2024

@author: Neyma
"""

import os
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Load a model
model = YOLO("yolov8n.yaml")
#model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
model.train(data="C:/Users/Neyma/OneDrive/Documents/Bristol Uni/2023-24 Modules/MDM3/Phase 3/Augment/Car detection.v1i.yolov8/data.yaml", epochs=3)  # train the model
metrics = model.val()  # evaluate model performance on the validation set
results = model("C:/Users/Neyma/AppData/Local/Temp/best_electric_luxury_car_bmw_i7.jpg")  # predict on an image
path = model.export(format="onnx")  # export the model to ONNX format