import os

COCO_FILE=os.path.join("train","_annotations.coco.json")
CATEGORIES_JSON=os.path.join("categories.json")
NUM_EPOCHS=40
BEST_FASTERRCNN_STATIC_DICT="./model/bestfasterRcnn.pth"
DEFAULT_FASTERRCNN_STATIC_DICT="./model/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
NUM_CLASSES=17
BATCH_SIZE=12

#TODO 学习率
MODIFY_LEARNING=5e-3
COMMON_LEARNING=1e-3

COLORS = [
    "#8470FF",  # 原色1: 蓝紫色
    "#87CEFA",  # 原色2: 浅天蓝
    "#7FFFD4",  # 原色3: 碧绿色
    "#FFFFE0",  # 原色4: 浅黄
    "#F08080",  # 原色5: 亮珊瑚
    "#FFC0CB",  # 原色6: 粉红
    "#AEEEEE",  # 原色7: 淡青色
    "#00FFFF",  # 原色8: 青色
    "#32C5FF",  # 原色9: 亮蓝
    "#D8BFD8",  # 原色10: 紫罗兰
    "#FFA07A",  # 新增11: 浅橙红
    "#98FB98",  # 新增12: 薄荷绿
    "#DDA0DD",  # 新增13: 梅红色
    "#FFD700",  # 新增14: 金色
    "#9370DB",  # 新增15: 中紫色
    "#3CB371",  # 新增16: 海洋绿
    "#FF6347",  # 新增17: 番茄红
    "#40E0D0",  # 新增18: 绿松石
    "#FF8C00",  # 新增19: 深橙
    "#BA55D3",  # 新增20: 中兰花紫
]