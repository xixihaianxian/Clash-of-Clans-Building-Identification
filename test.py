import torch
from torch import nn
from torchvision import models
import tools
from loguru import logger
import model
import config
from tqdm import tqdm
import random
import matplotlib.pyplot as plt

def test(model:models.detection.FasterRCNN,testIter):
    model=model.to(device=torch.device(tools.environment()))
    model.eval()
    testRecord=tools.Record()
    with torch.no_grad():
        for image,target in tqdm(testIter):
            image=[image.to(device=torch.device(tools.environment()))]
            target=[{key:value.to(device=torch.device(tools.environment())) for key,value in target.items()}]
            loss_dict=tools.compute_loss(model,image,target)
            model.eval()
            loss=sum(value for value in loss_dict.values())
            testRecord.save(loss)
    logger.info(f"test loss is {testRecord.value}")
    testRecord.reset()
    
def compare(model:models.detection.FasterRCNN,testIter):
    model=model.to(device=torch.device("cpu"))
    model=model.eval()
    item=random.choice(range(len(testIter)))
    image,target=testIter[item]
    image=[image]
    target=[target]
    #TODO 绘制真实情况
    tools.draw_anchor_box(image=image[0],anchors=target[0]["boxes"],categorie_ids=target[0]["labels"])
    
    #TODO 绘制预测情况
    output=model(image)
    tools.draw_anchor_box(image=image[0],anchors=output[0]["boxes"],categorie_ids=output[0]["labels"])
    
if __name__=="__main__":
    fasterRcnn=model.easy_fasterRcnn()
    fasterRcnn=model.modify_fasterRcnn(fasterRcnn)
    params=torch.load(config.BEST_FASTERRCNN_STATIC_DICT,weights_only=True)
    fasterRcnn.load_state_dict(params)
    testIter=tools.COCDataset(data_type="test",transform=tools.transform("test"))
    # test(model=fasterRcnn,testIter=testIter)
    compare(model=fasterRcnn,testIter=testIter)