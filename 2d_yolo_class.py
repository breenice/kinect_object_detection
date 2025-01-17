# from ultralytics import YOLO

# class YOLO_Model:

#     def __init__(self, model_id=0):
#         self.model_id = model_id

#     def train_models():
#         # load you only look once models
#         model = YOLO()
#         model = YOLO("yolo11n.yaml")  # new model from YAML
#         model = YOLO("yolo11n.pt")  #retrained model
#         model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build + transfer weights

#         # train model
#         results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

#     def validation(self):
#         # Load a model
#         model = YOLO("yolo11n.pt")  # load an official model

#         # Validate the model
#         metrics = model.val()  # no arguments needed, dataset and settings remembered
#         metrics.box.map  # map50-95
#         metrics.box.map50  # map50
#         metrics.box.map75  # map75
#         metrics.box.maps  # a list contains map50-95 of each category

# if __name__ == "__main__":
#     model1 = YOLO_Model(model_id=0)
#     model1.validation()

from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load an official model

# Validate the model
metrics = model.val(data="coco.yaml")  # no arguments needed, dataset and settings remembered
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category