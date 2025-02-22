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

class GroundedSAM:
    def __init__(self):
        # initialize grounding dino processor and model from hugging face
        gdino_model_id = "IDEA-Research/grounding-dino-tiny" #lightweight version of grounding dino
        self.device = "cuda"
        self.gprocessor = AutoProcessor.from_pretrained(gdino_model_id)
        self.gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(gdino_model_id).to(self.device)

        # initialize sam processor and model from checkpoint
        SAM_ENCODER_VERSION = "vit_h"
        SAM_CHECKPOINT_PATH = os.path.join("weights", "sam_vit_h_4b8939.pth")
        sam_device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=sam_device)
        self.sam_predictor = SamPredictor(sam)

        self.image = None

    def load_image(self, image_path):
        self.image = cv2.imread(image_path)

    """
    get object detections with Grounding DINO
    2 methods: query or all objects
    """
    # query method
    def get_detections(self,text):
        inputs = self.gprocessor(images=self.image, text=text, return_tensors="pt").to(self.device) # transforms raw image into PyTorch tensor resize to input to dino model
        with torch.no_grad(): # disabled gradient calculations (to save time re-enable if no time constriants)
            outputs = self.gdino_model(**inputs)

        # output: list of dictionaries 
        # dictionaries inlcude per image/batch: scores, labels and bounding boxes
        results = self.gprocessor.post_process_grounded_object_detection(
            outputs, # predictions: bounding boxes and classifications (with scoring)
            inputs.input_ids, # tokenized text input ids with separator tokens
            box_threshold=0.4, # keeps bounding boxes with at least 40% confidence score
            text_threshold=0.3, # keep predictions (matched to text input) only if at least 30% confidence
            target_sizes = [self.image.shape[:2]]

            # note: tensor size if specified must be tuple or list
        )
        print(results)
        return results
    
    """
    segment image per detected object with segment anything model
    SAM outputs array of booleans
    """
    def segment_with_boxes(self, results):
        result_masks = []
        self.sam_predictor.set_image(self.image)
        boxes = results[0]["boxes"].cpu().numpy() # extract bounding boxes from grounding dino detections

        # iterate through bounding boxes and save mask of best confidence score per box
        for b in boxes:
            masks, s_scores, _ = self.sam_predictor.predict(
                box=b,
                multimask_output=True
            )
            index = np.argmax(s_scores)
            result_masks.append(masks[index])

        result_masks=np.array(result_masks)
        return result_masks

    """
    show annotated image with bounding box and segmentation mask
    """
    def show_seg_box(self, results, masks):
        classes = (results[0])["labels"]
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(image_rgb)


        scores = results[0]['scores']
        labels = results[0]['labels']
        boxes = results[0]['boxes']
        colors = []
        for i in range(len(scores)):
            colors.append(np.random.randint(100, 256, size=3).tolist()) # rand color

        for i, (score, label, box) in enumerate(zip(scores, labels, boxes)):
            # draw bounding box
            color = [x / 256 for x in colors[i]]
            x_min, y_min, x_max, y_max = box.tolist()
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x_min, y_min - 10, f"{label}: {score:.2f}", color=color, fontsize=12) # label

        overlay = image_rgb.copy() 
        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask = mask.astype(np.uint8)
            mask_colored = np.zeros_like(image_rgb, dtype=np.uint8)
            mask_colored[mask==1] = colors[i]

            overlay = cv2.addWeighted(overlay, 1.0, mask_colored, 0.5, 0) # iteratively adds mask (imshow overwrites last saved mask)
        ax.imshow(overlay)

        plt.axis('off') 
        plt.show()

if __name__ == "__main__":
    gsam = GroundedSAM()
    gsam.load_image("/home/stretch/Documents/bree/sam/data/forestcat.jpg")
    query = "a cat. a tree."
    detections = gsam.get_detections(query)
    masks = gsam.segment_with_boxes(detections)
    gsam.show_seg_box(detections, masks)

