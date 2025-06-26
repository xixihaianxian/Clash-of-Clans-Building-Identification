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
        image=image.permute(2,1,0)
    elif isinstance(image,np.ndarray):
        image=image.transpose((2,1,0))
    plt.figure()
    fig=plt.imshow(image)
    for anchor,categorie_id in zip(anchors,categorie_ids):
        fig.axes.add_patch(p=add_anchor(fill=False,color=config.COLORS[categorie_id],anchor=anchor))
        add_text(axes=fig.axes,text_color=config.COLORS[categorie_id],content=id2category.get(str(categorie_id)),anchor=anchor)
    plt.show()
    
class COCDataset(data.Dataset):
    def __init__(self):
        super().__init__()
        

if __name__=="__main__":
    builds_category(config.COCO_FILE,config.CATEGORIES_JSON)
    csv=pd.read_csv("./train/train.csv",sep=",")
    image_name=csv["file_name"].unique()[0]
    records=csv[csv["file_name"]==image_name]
    anchors=records[["x1","y1","x2","y2"]].values
    categoris=records["category_id"].values
    image_path=os.path.join("train",image_name)
    image=cv2.imread(image_path,cv2.IMREAD_COLOR)
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float64)
    image/=255.0
    image=image.transpose((2,1,0))
    draw_anchor_box(image=image,anchors=anchors,categorie_ids=categoris)