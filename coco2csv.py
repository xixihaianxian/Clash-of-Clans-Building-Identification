import numpy as np
import pandas as pd
import os
from pycocotools.coco import COCO

def get_coco_file_path(data_dir):
    coco_file = os.path.join(data_dir,'_annotations.coco.json')
    return coco_file

def coco2csv(coco_file, csv_file):

    coco = COCO(coco_file)

    image_ids = coco.getImgIds()
    images = coco.loadImgs(image_ids)
    print(f"寻找到图片{len(images)}张！")

    all_annotations = list() #TODO 所有annotations
    
    for image in images:
        ann_ids = coco.getAnnIds(imgIds=image['id'], iscrowd=None)
        annotations = coco.loadAnns(ann_ids)
        if annotations:
            all_annotations.extend(annotations)
 
    annotations = pd.DataFrame(data = all_annotations)
    print(f"寻找到annotations{len(annotations)}个！")

    annotations[['x1', 'y1', 'x2', 'y2']] = pd.DataFrame(annotations['bbox'].values.tolist(), index=annotations.index).astype(int)
    annotations = annotations.assign(x2 = annotations['x1'] + annotations['x2'], y2 = annotations['y1'] + annotations['y2']) 
    #TODO 将x1,y1,height,width转化x1,y1x2,y2

    images = pd.DataFrame(images)
    images.rename(columns={'id':'image_id'}, inplace=True)
    images.set_index('image_id')

    annotations_csv = pd.merge(annotations, images, on=['image_id'], how='right')
    annotations_csv = annotations_csv.replace(np.nan, '', regex=True)

    colnames = ["file_name","height","width","x1","y1","x2","y2","category_id"]
    #TODO 获取"file_name","height","width","x1","y1","x2","y2","category_id"等数据
    annotations_csv = annotations_csv[colnames]

    annotations_csv.to_csv(path_or_buf=csv_file, index=False, header=True, columns=colnames)
    
if __name__=="__main__":
    
    coco_dirs=["train","test","valid"]
    
    for coco_dir in coco_dirs:
        coco_dir=coco_dir
        csv_file=os.path.join(coco_dir,f"{coco_dir}.csv")
        coco_file=get_coco_file_path(data_dir=coco_dir)
        coco2csv(coco_file=coco_file,csv_file=csv_file)