import torch
from torch import nn
from torchvision import models
import os
from loguru import logger
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import config

os.path.exists("./model") or os.makedirs("./model")

def load_fasterRcnn(weight=True,static_dict_path=None):
    if weight:
        fasterRcnn=models.detection.fasterrcnn_resnet50_fpn(weights=models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    else:
        if static_dict_path is None:
            raise ValueError(f"未设置static_dict_path的值!")
        fasterRcnn=models.detection.fasterrcnn_resnet50_fpn(weight=None)
        best_fasterRcnn_params=torch.load(static_dict_path,weights_only=True)
        fasterRcnn.load_state_dict(best_fasterRcnn_params)
    return fasterRcnn

def easy_fasterRcnn():
    return models.detection.fasterrcnn_resnet50_fpn(weights=None)

def modify_fasterRcnn(fasterRcnn:models.detection.FasterRCNN,num_classes=config.NUM_CLASSES):
    in_features=fasterRcnn.roi_heads.box_predictor.cls_score.in_features
    fasterRcnn.roi_heads.box_predictor=FastRCNNPredictor(in_channels=in_features,num_classes=num_classes)
    return fasterRcnn

if __name__=="__main__":
    fasterRcnn=load_fasterRcnn(weight=False,static_dict_path=config.DEFAULT_FASTERRCNN_STATIC_DICT)
    fasterRcnn=modify_fasterRcnn(fasterRcnn=fasterRcnn)