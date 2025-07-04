FROM python:3.9-slim

WORKDIR /face_emotion

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# 安装yolov5依赖
COPY yolov5 /face_emotion/models/yolo/yolov5
WORKDIR /face_emotion/models/yolo/yolov5
RUN pip install -r requirements.txt

# 创建模型目录
RUN mkdir -p /face_emotion/models/yolo
RUN mkdir -p /face_emotion/models/emotion

# 复制应用代码和配置文件
COPY main.py .
COPY .env .

# 暴露服务端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]