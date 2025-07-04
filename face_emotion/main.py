import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import numpy as np
import io
import requests
import time

# 配置
CONF_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 0.7))
BACKEND_URL = os.getenv("BACKEND_URL", "")

# 模型路径设置为容器内的位置
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "./models/yolo/best.pt")
EMOTION_WEIGHTS = os.getenv("EMOTION_WEIGHTS", "./models/emotion/rafdb_mobilenetv2.pth")

# 加载 YOLOv5n 自定义权重
print("正在加载 YOLO 模型...")
yolo_model = torch.hub.load(os.getenv("YOLO_MODEL", "./models/yolo/yolov5"), 'custom', path=YOLO_WEIGHTS, source='local')
print("YOLO 模型加载完成")
yolo_model.eval()

# 构建 MobileNetV2 并加载自定义权重
num_emotions = 7
emotion_model = torchvision.models.mobilenet_v2(weights=None)
in_features = emotion_model.classifier[1].in_features
emotion_model.classifier[1] = nn.Linear(in_features, num_emotions)
print("正在加载情绪识别模型...")
emotion_model.load_state_dict(torch.load(EMOTION_WEIGHTS, map_location='cpu'))
print("情绪识别模型加载完成")
emotion_model.eval()

# 图像预处理
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 情绪类别映射
emotion_labels = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

app = FastAPI()

# 处理图像的核心函数，提取出来方便测试脚本使用
def process_image(image):
    timing_info = {}
    faces = []
    
    # 总时间计时开始
    start_total = time.time()
    print("开始计时")
    
    # 人脸检测计时
    start_face_detection = time.time()
    results = yolo_model([np.array(image)])
    detections = results.xyxy[0]
    timing_info["face_detection_time"] = time.time() - start_face_detection
    print(f"人脸检测时间：{timing_info['face_detection_time']:.2f}秒")
    
    # 情绪识别时间总计
    emotion_recognition_total = 0
    
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        face_img = image.crop((x1, y1, x2, y2))

        # 情绪识别计时
        start_emotion = time.time()
        input_tensor = preprocess(face_img).unsqueeze(0)
        with torch.no_grad():
            output = emotion_model(input_tensor)
        probs = torch.softmax(output, dim=1)
        score, idx = torch.max(probs, 1)
        emotion = emotion_labels[idx.item()]
        emotion_time = time.time() - start_emotion
        emotion_recognition_total += emotion_time

        if score.item() >= CONF_THRESHOLD:
            faces.append({
                "bbox": [x1, y1, x2, y2],
                "emotion": emotion,
                "confidence": float(score.item())
            })
    
    # 记录情绪识别平均时间和总时间
    if detections.shape[0] > 0:  # 如果检测到了人脸
        timing_info["emotion_recognition_avg_time"] = emotion_recognition_total / detections.shape[0]
    else:
        timing_info["emotion_recognition_avg_time"] = 0
    
    timing_info["emotion_recognition_total_time"] = emotion_recognition_total
    print(f"情绪识别平均时间：{timing_info['emotion_recognition_avg_time']:.2f}秒")
    
    # 总时间计算
    timing_info["total_processing_time"] = time.time() - start_total
    print(f"总处理时间：{timing_info['total_processing_time']:.2f}秒")

    return faces, timing_info

@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    start_time = time.time()
    
    # 读取图像计时
    start_read = time.time()
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    read_time = time.time() - start_read
    
    # 处理图像
    faces, timing_info = process_image(image)
    
    # 添加图像读取时间
    timing_info["image_read_time"] = read_time
    
    # 后端传输计时
    backend_time = 0
    if BACKEND_URL and faces:
        start_backend = time.time()
        try:
            requests.post(BACKEND_URL, json={"results": faces})
        except Exception as e:
            print(f"Error sending to backend: {e}")
        backend_time = time.time() - start_backend
    
    timing_info["backend_communication_time"] = backend_time
    
    # API总响应时间
    timing_info["total_api_time"] = time.time() - start_time
    
    # 返回结果，包含时间信息
    return JSONResponse({
        "faces": faces,
        "timing": timing_info
    })