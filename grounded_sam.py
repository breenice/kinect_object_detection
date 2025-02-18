import requests
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor
from segment_anything import  sam_model_registry, SamPredictor
import os

# grounded dino---------------------------------------------------------------------
model_id = "IDEA-Research/grounding-dino-tiny" #lightweight version of grounding dino
device = "cuda"

gprocessor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image = Image.open("/home/stretch/Documents/bree/sam/forestcat.jpg")
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
    target_sizes=[image.size[::-1]] # refromats for viewing (previously flipped for model input)
)

# sam ----------------------------------------------------------------------------------------
SAM_CHECKPOINT_PATH = os.path.join("weights", "sam_vit_h_4b8939.pth")
SAM_ENCODER_VERSION = "vit_h"
CLASSES = ['car', 'dog', 'person', 'nose', 'chair', 'shoe', 'ear']
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

sam_device = "cuda" if torch.cuda.is_available() else "cpu" # run with cuda
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(sam_device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
sam_image = Image.open("/home/stretch/Documents/bree/sam/forestcat.jpg")
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

sam_inputs = sam_processor(sam_image, return_tensors="pt").to(sam_device) # tensor format with normalization, resize, and shape transformation
sam_inputs["input_points"] = []
sam_inputs["input_boxes"] = []
sam_inputs["input_labels"] = []

input_boxes = []
image_embeddings = sam_model.get_image_embeddings(sam_inputs["pixel_values"]) 

for result in results:
    scores = result['scores']
    labels = result['labels']
    boxes = result['boxes']

    for score, label, box in zip(scores, labels, boxes):
        x_min, y_min, x_max, y_max = box.tolist()
        input_boxes.append([x_min, y_min, x_max, y_max])
input_boxes = np.array(input_boxes)

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
sam_predictor = SamPredictor(sam)

masks, _, _ = sam_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes[None, :],
    multimask_output=False,
)

inputs = sam_processor(image, input_boxes=[input_boxes], return_tensors="pt").to(device)

inputs.pop("pixel_values", None) #no longer using raw pixel values, using embedded inputs
inputs.update({"image_embeddings": image_embeddings})

with torch.no_grad():
    sam_outputs = sam_model(**sam_inputs) # predictions (forward pass with no gradients)

# masks output as (number of detected objects, 1(singleton used for identifying batches), image height, image width)
# use masks[0][0] as first detected mask group without the extra singleton dimension (so can be viewed on 2D image)
masks = sam_processor.image_processor.post_process_masks(sam_outputs.pred_masks.cpu(), 
                                                         sam_inputs["original_sizes"].cpu(), 
                                                         sam_inputs["reshaped_input_sizes"].cpu()) # masks resized with input preferences


# show image ---------------------------------------------------------------------------------------------

# extract data from grounding dino results
for result in results:
    scores = result['scores']
    labels = result['labels']
    boxes = result['boxes']

nb_predictions = scores.shape[-1]
fig, axes = plt.subplots(1, nb_predictions, figsize=(15, 15))

for i, (mask, score) in enumerate(zip(masks[0][0], scores)):
    mask = mask.cpu().detach().numpy() 
    
    axes[i].imshow(np.array(sam_image)) # raw image
    axes[i].imshow(mask, cmap='grey', alpha=0.5) # apply masks as transparent grey

    x_min, y_min, x_max, y_max = boxes[i].tolist()
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                         linewidth=2, edgecolor='red', facecolor='none')
    axes[i].add_patch(rect) # draw bounding boxes
    axes[i].text(x_min, y_min - 10, f"Score: {score.item():.2f}", color='red', fontsize=12)
    axes[i].title.set_text(f"Mask {i+1}, Score: {score.item():.3f}")
    axes[i].axis("off") 

plt.show()
