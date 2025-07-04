import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # 强制使用 CPU

yolo_repo = "../models/yolo/yolov5"
yolo_weights = "../models/yolo/best.pt"

print(f"Loading model from: {yolo_repo}")
print(f"Weights file: {yolo_weights}")

model = torch.hub.load(yolo_repo, 'custom', path=yolo_weights, source='local')
print("模型加载成功！")

# import torch
# print(torch.cuda.is_available())
# print(torch.__version__)