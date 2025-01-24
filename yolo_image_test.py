import cv2
import numpy as np
from mathplotlib import pyplot as plt
from ultralytics import YOLO

# load image
img_path = 'data/images/bus.jpg'
image = cv2.imread(img_path)
image_rbg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# load model
model = YOLO("yolov8n.pt")
results = model(image_rbg)
results[0].show()