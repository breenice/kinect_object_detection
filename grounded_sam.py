import requests

import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, SamModel, SamProcessor

# grounded dino---------------------------------------------------------------------
model_id = "IDEA-Research/grounding-dino-tiny" #lightweight version of grounding dino
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image = Image.open("/home/stretch/Documents/bree/sam/forestcat.jpg")
# Check for cats and remote controls
text = "a cat. a tree."

inputs = processor(images=image, text=text, return_tensors="pt").to(device) # transforms raw image into PyTorch tensor resize to input to dino model
with torch.no_grad(): # disabled gradient calculations (to save time re-enable if no time constriants)
    outputs = model(**inputs)

# output: list of dictionaries 
# dictionaries inlcude per image/batch: scores, labels and bounding boxes
results = processor.post_process_grounded_object_detection(
    outputs, # predictions: bounding boxes and classifications (with scoring)
    inputs.input_ids, # tokenized text input ids with separator tokens
    box_threshold=0.4, # keeps bounding boxes with at least 40% confidence score
    text_threshold=0.3, # keep predictions (matched to text input) only if at least 30% confidence

    # note: tensor size if specified must be tuple or list
    target_sizes=[image.size[::-1]] # refromats for viewing (previously flipped for model input)
)

# sam ----------------------------------------------------------------------------------------
sam_device = "cuda" if torch.cuda.is_available() else "cpu" # run with cuda
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(sam_device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
sam_image = Image.open("/home/stretch/Documents/bree/sam/forestcat.jpg")
inputs = sam_processor(image, return_tensors="pt").to(sam_device) # tensor format with normalization, resize, and shape transformation
image_embeddings = sam_model.get_image_embeddings(inputs["pixel_values"]) 
sam_masks = []
sam_scores = []
for result in results:
    boxes = result['boxes']
    for box in boxes: 
        x_min, y_min, x_max, y_max = box.tolist()
        # Instead of creating a list of lists, directly append the box coordinates
        center_x = (x_min+x_max) / 2
        center_y = (y_min+y_max) / 2
        input_points = [[[0,0]]]  # this is now a list of points
        input_boxes = boxes  # directly use the box here, not a list of lists
        print(input_boxes)

        # Assuming sam_processor and sam_device are properly defined
        inputs = sam_processor(sam_image, input_boxes=[input_boxes], input_points=[input_points], return_tensors="pt").to(sam_device)

        inputs.pop("pixel_values", None) #no longer using raw pixel values, using embedded inputs
        inputs.update({"image_embeddings": image_embeddings})
        with torch.no_grad():
            outputs = sam_model(**inputs) # predictions (forward pass with no gradients)

        sam_masks.append(processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())) # masks resized with input preferences
        sam_scores.append(outputs.iou_scores)

fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(sam_image)

for result in results:
    scores = result['scores']
    labels = result['labels']
    boxes = result['boxes']

    for score, label, box in zip(scores, labels, boxes):
        # draw bounding box
        x_min, y_min, x_max, y_max = box.tolist()
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                             linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min - 10, f"{label}: {score:.2f}", color='red', fontsize=12) # label

# show segmentation masks
for mask in sam_masks:
    for m in mask:
        ax.imshow(m[0], alpha=0.5) 

plt.axis('off') 
plt.show()