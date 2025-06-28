import torch
from torch import nn
from torch import optim
import config
import tools
from loguru import logger
import math
import model
from tqdm import tqdm
from collections import defaultdict

def train(model:nn.Module,optimizer:optim.Optimizer,trainIter,validIter,num_epochs=config.NUM_EPOCHS):
    
    model=model.to(device=torch.device(tools.environment()))
    trainRecord=tools.Record()
    validRecord=tools.Record()
    minloss=math.inf
    train_loss_list=list()
    valid_loss_list=list()
    
    for epoch in range(num_epochs):
        model.train()
        trainRecord.reset()
        for images,targets in tqdm(trainIter):
            images=[image.to(device=torch.device(tools.environment())) for image in images]
            targets=[{key:value.to(device=torch.device(tools.environment())) for key,value in target.items() } for target in targets]
            loss_dict=model(images,targets)
            loss=sum(value for value in loss_dict.values())
            trainRecord.save(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f"train epoch:{epoch+1} loss:{trainRecord.value}")
        train_loss_list.append(trainRecord.value)
            
        with torch.no_grad():
            
            validRecord.reset()
            for val_image,val_target in tqdm(validIter):
                val_image=[val_image.to(device=torch.device(tools.environment()))]
                val_target=[{key:value.to(device=torch.device(tools.environment())) for key,value in val_target.items()}]
                val_loss_dict=model(val_image,val_target)
                val_loss_value=sum(value for value in val_loss_dict.values())
                validRecord.save(val_loss_value)
            
            logger.info(f"valid loss {validRecord.value}")
            valid_loss_list.append(validRecord.value)
            
            if minloss>validRecord.value:
                minloss=validRecord.value
                torch.save(obj=model.state_dict(),f=config.BEST_FASTERRCNN_STATIC_DICT)
                logger.info(f"save best fasterRcnn model!")
                
    return train_loss_list,valid_loss_list

if __name__=="__main__":
    
    fasterRcnn=model.load_fasterRcnn(weight=False,static_dict_path=config.DEFAULT_FASTERRCNN_STATIC_DICT)
    fasterRcnn=model.modify_fasterRcnn(fasterRcnn=fasterRcnn)
    optimizer=tools.build_optimizer(fasterRcnn=fasterRcnn)
    trainDataset=tools.COCDataset(data_type="train",transform=tools.transform(type="train"))
    validDataset=tools.COCDataset(data_type="valid",transform=tools.transform(type="valid"))
    trainIter=tools.data_iter(dataset=trainDataset,batch_size=config.BATCH_SIZE,shuffle=True,self_collate_fn=True)
    
    train_loss_list,valid_loss_list=train(model=fasterRcnn,optimizer=optimizer,trainIter=trainIter,validIter=validDataset)
    
    loss_kind=defaultdict(list)
    
    loss_kind["train loss"]=train_loss_list
    loss_kind["valid loss"]=valid_loss_list
    
    tools.draw_loss_line(loss_kind=loss_kind)