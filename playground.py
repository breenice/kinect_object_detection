import requests
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
from segment_anything import  sam_model_registry, SamPredictor
import os
import cv2

# grounded dino---------------------------------------------------------------------
model_id = "IDEA-Research/grounding-dino-tiny" #lightweight version of grounding dino
device = "cuda"

gprocessor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image = cv2.imread("/home/stretch/Documents/bree/sam/forestcat.jpg")
# Check for cats and remote controls
text = "a cat. a tree."

inputs = gprocessor(images=image, text=text, return_tensors="pt").to(device) # transforms raw image into PyTorch tensor resize to input to dino model
with torch.no_grad(): # disabled gradient calculations (to save time re-enable if no time constriants)
    outputs = model(**inputs)

# output: list of dictionaries 
# dictionaries inlcude per image/batch: scores, labels and bounding boxes
results = gprocessor.post_process_grounded_object_detection(
    outputs, # predictions: bounding boxes and classifications (with scoring)
    inputs.input_ids, # tokenized text input ids with separator tokens
    box_threshold=0.4, # keeps bounding boxes with at least 40% confidence score
    text_threshold=0.3, # keep predictions (matched to text input) only if at least 30% confidence

    # note: tensor size if specified must be tuple or list
)

# sam ----------------------------------------------------------------------------------------
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join("weights", "sam_vit_h_4b8939.pth")
sam_device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=sam_device)
sam_predictor = SamPredictor(sam)

# image and boxes must be a numpy array
sam_predictor.set_image(image)
result_masks = []
for result in results:
    masks, scores, logits = sam_predictor.predict(
        box=result["boxes"].cpu().numpy()[0],
        multimask_output=True
    )
    index = np.argmax(scores)
    result_masks.append(masks[index])
resultmask=np.array(result_masks)

# show image ---------------------------------------------------------------------------------------------
import matplotlib.patches as patches
fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image)

for idx, result in enumerate(results):
    box = result["boxes"].cpu().numpy()[0]
    label = result["labels"][0]  
    h, w = image.shape[:2]
    box = box * np.array([w, h, w, h])
    x_min, y_min, x_max, y_max = box.astype(int)
    
    rect = patches.Rectangle(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        linewidth=2, edgecolor='r', facecolor='none'
    )
    ax.add_patch(rect)
    
    ax.text(x_min, y_min - 10, f"{label} {scores[idx]:0.2f}", color='white', fontsize=12)
    mask = resultmask[idx]
    ax.imshow(mask, alpha=0.5, cmap='jet') # lower mask opacity

plt.axis('off')  
plt.show()

