import requests

import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny" #lightweight version of grounding dino
device = "cuda"

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)
# Check for cats and remote controls
text = "a cat. a remote control."

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
print(results)

fig, ax = plt.subplots(1, figsize=(10, 10))
ax.imshow(image)

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
plt.axis('off') 
plt.show()