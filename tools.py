import torch
from torch import nn
from torch.utils import data
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import config
from matplotlib.patches import Rectangle
import matplotlib
from typing import List
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch import optim
from torchvision import models

def builds_category(coco_file,categories_path):
    
    categories_all=defaultdict(str)
    
    with open(coco_file,"r",encoding="utf-8") as file:
        contens=file.read()
        
    coco_json=json.loads(contens)
    
    categories=coco_json.get("categories")
    
    for categorie in categories:
        categories_all[categorie.get("id")]=categorie.get("name")
        
    with open(categories_path,"w",encoding="utf-8") as file:
        file.write(json.dumps(categories_all,indent=2,ensure_ascii=False))
        

def load_categories(categories_path):
    with open(categories_path,"r",encoding="utf-8") as file:
        categories=json.load(file)
    return categories

def anchor_transform(anchors):
    r"""
    将x1,y1,x2,y2转化为x1,y1,w,h
    """
    if isinstance(anchors,torch.Tensor):
        anchors=anchors.detach().cpu().numpy()
    x1,y1,x2,y2=anchors[:,0],anchors[:,1],anchors[:,2],anchors[:,3]
    w=x2-x1
    h=y2-y1
    return np.stack([x1,y1,w,h],axis=1)

def add_anchor(fill:bool,color:str,anchor:torch.Tensor):
    x1,y1,w,h=anchor
    ragb_color=matplotlib.colors.to_rgb(color)
    return Rectangle(
        xy=(x1,y1),
        width=w,
        height=h,
        fill=fill,
        facecolor=(*ragb_color,0.5) if fill else (*ragb_color,1),
        edgecolor=(*ragb_color,1),
        linewidth=1.5,
    )

def add_text(axes,text_color,content,anchor):
    x1,y1,w,h=anchor
    axes.text(
        x=x1,
        y=y1,
        s=content,
        va="center",
        ha="center",
        fontsize=6,
        color=text_color,
    )

def draw_anchor_box(image:torch.Tensor,anchors,categorie_ids):
    id2category=load_categories("./categories.json")
    anchors=anchor_transform(anchors)
    if isinstance(image,torch.Tensor):
        image=image.permute(1,2,0)
    elif isinstance(image,np.ndarray):
        image=image.transpose((1,2,0))
    plt.figure()
    fig=plt.imshow(image)
    for anchor,categorie_id in zip(anchors,categorie_ids):
        if isinstance(categorie_id,torch.Tensor):
            categorie_id=categorie_id.item()
        fig.axes.add_patch(p=add_anchor(fill=False,color=config.COLORS[categorie_id],anchor=anchor))
        add_text(axes=fig.axes,text_color=config.COLORS[categorie_id],content=id2category.get(str(categorie_id)),anchor=anchor)
    plt.show()
    
class COCDataset(data.Dataset):
    
    def __init__(self,data_type,transform=None):
        super().__init__()
        data_file=os.path.join(data_type,f"{data_type}.csv")
        self.data=pd.read_csv(data_file)
        self.transform=transform
        self.image_dir=os.path.join(data_type)
        self.image_names=self.data["file_name"].unique()
        
    def __getitem__(self,item):
        target=defaultdict(list)
        
        image_name=self.image_names[item]
        records=self.data[self.data["file_name"]==image_name]
        anchors=records[["x1","y1","x2","y2"]].values
        labels=records["category_id"].values
        labels=np.array(labels,dtype=np.int64)
        iscrowds=torch.as_tensor(records["iscrowd"].values,dtype=torch.int64)
        areas=((anchors[:,2]-anchors[:,0])*(anchors[:,3]-anchors[:,1]))
        areas=torch.as_tensor(data=areas,dtype=torch.float32)
        
        image_file=os.path.join(self.image_dir,image_name)
        image=cv2.imread(image_file,cv2.IMREAD_COLOR)
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)
        image=image/255.0
        
        target["boxes"]=anchors
        target["labels"]=labels
        target["areas"]=areas
        target["iscrowds"]=iscrowds
        
        if self.transform is not None:
            smaple={
                "image":image,
                "bboxes":anchors,
                "labels":labels,
            }
            #TODO smaple里面的三个件不要变化
            smaple=self.transform(**smaple)
            image=smaple["image"]
            
            target["boxes"]=torch.stack(tuple(map(lambda x:torch.tensor(x),zip(*smaple["bboxes"])))).permute(1,0)
            
        else:
            target["boxes"]=torch.as_tensor(target["boxes"],dtype=torch.float32)
            
        target["labels"]=torch.from_numpy(target["labels"]).to(dtype=torch.int64)
            
        return image,target
    
    def __len__(self):
        return len(self.image_names)
        
def transform(type):
    if type=="train":
        return A.Compose(transforms=[
            A.HorizontalFlip(p=0.5),
            ToTensorV2(p=1),
        ],bbox_params={"format":"pascal_voc","label_fields":["labels"]})
        # format': 'pascal_voc'：
        # 指定边界框的格式为PASCAL VOC格式，即(x_min, y_min, x_max, y_max)，表示左上角和右下角的坐标。
        # 这是最常用的边界框表示格式之一。
        # 'label_fields': ['labels']：
        # 指定与每个边界框相关联的标签存储在名为'labels'的字段中。
        # 这告诉Albumentations在进行图像变换（如水平翻转）时，不仅要变换边界框坐标，还要保持边界框和标签的对应关系。
    else:
        return A.Compose(transforms=[
            ToTensorV2(p=1),
        ],bbox_params={"format":"pascal_voc","label_fields":["labels"]})
        
        
def collate_fn(batch):
    return list(zip(*batch))
        
def data_iter(dataset:data.Dataset,batch_size:int,shuffle:bool,self_collate_fn:bool):
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn if self_collate_fn else None,
    )
    
def environment():
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_optimizer(fasterRcnn:models.detection.FasterRCNN):
    modify_params=[param for name,param in fasterRcnn.named_parameters() if "roi_heads.box_predictor" in name]
    original_params=[param for name,param in fasterRcnn.named_parameters() if "roi_heads.box_predictor" not in name]
    return optim.SGD(
        params=[
            {"params":modify_params,"lr":config.MODIFY_LEARNING},
            {"params":original_params,"lr":config.COMMON_LEARNING},
        ],
        momentum=0.9,
        weight_decay=5e-4,
    )
    
def compute_loss(model, images, targets):
    r'''
    测试集上检查损失，时使用
    '''
    model.train()
    with torch.no_grad():
        loss_dict = model(images, targets)
    return loss_dict
    
class Record:
    def __init__(self):
        self.loss_total=0.0
        self.iter_num=0
        
    def save(self,value):
        self.loss_total+=value
        self.iter_num+=1
    
    @property
    def value(self):
        return self.loss_total/self.iter_num
    
    def reset(self):
        self.loss_total=0.0
        self.iter_num=0
        
def draw_loss_line(loss_kind,epochs=config.NUM_EPOCHS):
    os.path.exists("result") or os.makedirs("result")
    plt.figure()
    for name,loss in loss_kind.items():
        loss=[item.detach().cpu().numpy() for item in loss]
        plt.plot(range(epochs),loss,label=f"{name}")
    plt.legend()
    plt.xticks(np.linspace(1,epochs,10,dtype=np.int64))
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.savefig("./result/result.png")
    plt.show()

if __name__=="__main__":
    cocodataset=COCDataset(data_type="train",transform=transform("train"))
    image=cocodataset[0][0]
    anchors=cocodataset[0][1]["anchors"]
    labels=cocodataset[0][1]["labels"]
    draw_anchor_box(image,anchors,labels)
    plt.show()