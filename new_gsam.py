import requests
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoModel
from segment_anything import  sam_model_registry, SamPredictor
import os
import cv2
import supervision as sv
from typing import List

# grounded dino---------------------------------------------------------------------
model_id = "IDEA-Research/grounding-dino-tiny" #lightweight version of grounding dino
device = "cuda"
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join("sam","weights", "groundingdino_swint_ogc.pth")
GROUNDING_DINO_CONFIG_PATH = os.path.join("sam","GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")

# model and processor
model = AutoModel.from_pretrained(GROUNDING_DINO_CHECKPOINT_PATH)
processor = AutoProcessor.from_pretrained(GROUNDING_DINO_CONFIG_PATH)


image = cv2.imread("/home/stretch/Documents/bree/sam/forestcat.jpg")
# Check for cats and stuff
text = "a cat. a remote control."

inputs = processor(images=image, text=text, return_tensors="pt").to(device) # transforms raw image into PyTorch tensor resize to input to dino model
with torch.no_grad(): # disabled gradient calculations (to save time re-enable if no time constriants)
    detections = model(**inputs)

# detect with text prompt with processor (preprocessing before input model then post processing for threshold matching)
# output: list of dictionaries 
# dictionaries inlcude per image/batch: scores, labels and bounding boxes
results = model.predict_with_classes(
    image=image,
    classes=text.split(" "),
    box_threshold=0.4,
    text_threshold=0.3
)

labels = []
i = "labels"
j = "scores"
for result in results:
    labels.append(f"{result[i]} {result[j]}")
print(labels)

box_drawer = sv.BoxAnnotator()
annotated_frame = box_drawer.annotate(scene=image.copy(), detections=results, labels=labels)

sv.plot_image(annotated_frame, (16, 16))
plt.show()

#sam----------------------------------------------------------------------------------------
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join("weights", "sam_vit_h_4b8939.pth")
sam_device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=sam_device)
sam_predictor = SamPredictor(sam)

# image and boxes must be a numpy array
def segment(image, boxes):
    sam_predictor.set_image(image)
    result_masks = []
    for box in boxes:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

masks = segment(
    sam_predictor=sam_predictor,
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
    boxes=detections.xyxy
)