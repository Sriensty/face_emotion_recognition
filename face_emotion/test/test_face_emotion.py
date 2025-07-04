import argparse
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import requests
import io
import json

def test_local(image_path):
    """本地直接测试情绪识别"""
    from main import process_image
    
    # 加载图像
    print(f"正在处理图像: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    # 处理图像
    results = process_image(image)
    
    # 显示结果
    draw = ImageDraw.Draw(image)
    
    # 尝试加载字体，如果失败则使用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
    
    # 绘制结果
    for face in results:
        x1, y1, x2, y2 = face["bbox"]
        emotion = face["emotion"]
        confidence = face["confidence"]
        
        # 绘制边界框
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
        
        # 绘制标签
        label = f"{emotion}: {confidence:.2f}"
        draw.text((x1, y1 - 15), label, fill="red", font=font)
    
    # 显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    # 输出识别结果
    print("识别结果:")
    for i, face in enumerate(results):
        print(f"人脸 {i+1}:")
        print(f"  情绪: {face['emotion']}")
        print(f"  置信度: {face['confidence']:.2f}")
        print(f"  位置: {face['bbox']}")
    
    return results

def test_api(image_path, api_url="http://localhost:8000/detect"):
    """通过API测试情绪识别"""
    # 读取图像
    with open(image_path, "rb") as f:
        files = {"file": f}
        
        # 发送请求
        response = requests.post(api_url, files=files)
        
    # 检查请求是否成功
    if response.status_code == 200:
        results = response.json()
        print("API 返回结果:")
        print(json.dumps(results, indent=2))
        return results
    else:
        print(f"请求失败，状态码: {response.status_code}")
        print(response.text)
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试人脸情绪识别")
    parser.add_argument("image_path", help="图像文件路径")
    parser.add_argument("--api", action="store_true", help="使用API测试而不是本地测试")
    parser.add_argument("--api_url", default="http://localhost:8000/detect", help="API的URL")
    
    args = parser.parse_args()
    
    if args.api:
        test_api(args.image_path, args.api_url)
    else:
        test_local(args.image_path)