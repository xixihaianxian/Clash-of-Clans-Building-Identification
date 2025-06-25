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

def add_anchor()

def draw_anchor_box(image:torch.Tensor,anchors,categorie_ids):
    image=image.permute(1,2,0)
    plt.figure()
    fig=plt.imshow(image)
    for anchor,categorie_id in zip(anchors,categorie_ids):
        pass

if __name__=="__main__":
    builds_category(config.COCO_FILE,config.CATEGORIES_JSON)
    categories=load_categories(config.CATEGORIES_JSON)
    print(categories)