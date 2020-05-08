# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# Import necessary libraries
import numpy as np
import cv2
import random

# Import detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

im = cv2.imread("input.jpg")
cv2.imshow("Input", im)

cfg = get_cfg()

# Add project specifig confi
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Set threshold

# Run model on CPU
cfg.MODEL.DEVICE = 'cpu'

# Find a model from the mode zoo
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# Look at outputs
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

# Draw the predictions on the image
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale = 1.2)
v = v.draw_instance_predictions(outputs["instances"]).to("cpu")

cv2.imshow("Prediction", v.get_image()[:, :, ::-1])