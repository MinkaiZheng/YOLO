import os
import cv2
import json
import time
import base64
import argparse
import numpy as np
from datetime import datetime
import requests
from detect import run, postprocess
from fastapi import FastAPI
from pydantic import BaseModel
from ultralytics import YOLO
import gc
import torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=r'D:\Data\AI+随手拍\code\model\tower-v1.pt', help='model.pt path')
    parser.add_argument('--source', type=str, help='file path, Base64 string, or URL')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, nargs='+', default=[640,640], help='train, val image size (pixels)')
    parser.add_argument('--conf', type=float, default=0.60, help='confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--device', type=str, default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    return parser.parse_args()

class ImageRequest(BaseModel):
    type: str = "0"
    image: str

# Base64 解析
def base64_to_cv2_image(base64_str):
    """
    将Base64字符串解码为 OpenCV 图像对象
    :param base64_str: 图片的Base64字符串
    :return: cv2 的图像对象
    """
    # 如果有头部信息，去掉
    if base64_str.startswith("data:image"):
        header, base64_str = base64_str.split(",", 1)
    # 解码 Base64
    image_data = base64.b64decode(base64_str)
    # 转换成 numpy 数组
    np_arr = np.frombuffer(image_data, np.uint8)
    # 解码为 OpenCV 图像对象
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    print(f"[INFO] Image decoded to OpenCV format, shape: {img.shape}")
    
    # # 将图像调整为固定尺寸 640x640 以确保推理一致性
    img_resized = cv2.resize(img, (640, 640))
    return img_resized

def url_to_cv2_image(url):
    resp = requests.get(url, timeout=10)
    if resp.status_code != 200:
        raise ValueError(f"URL请求失败，状态码: {resp.status_code}")
    img_array = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("URL图片内容解码失败")
    img_resized = cv2.resize(img, (640, 640))
    return img_resized

def warmup_model():
    print("[INFO] Warming up model with actual inference size...")
    # 创建与实际推理相同尺寸的图像(640x640)
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(5):  # 增加预热次数到5次
        opt.source = dummy_image
        run(opt, model)
        # 强制进行内存回收
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # 等待GPU操作完成
            torch.cuda.empty_cache()  # 清空GPU缓存
        gc.collect()  # 强制进行Python垃圾回收
    print("[INFO] Model warmed up successfully")

opt = parse_args()
# 全局加载模型
model = YOLO(opt.weights)
print("[INFO] YOLO Model Loaded Successfully!")
print(f"[INFO] Using device: {opt.device}")

# 固定随机种子以提高一致性
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    # 设置GPU为高性能模式
    torch.set_float32_matmul_precision('high')

app = FastAPI()
# 在启动服务前预热模型
warmup_model()

@app.get("/heartbeat")
async def heartbeat():
    return {
        "code": 200,
        "msg": "success",
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data": [
            {
                "result": "{\"msg\":\"success\",\"state\":0,\"result\":{}}",
                "algorithmCode": "XDL-001-AISSP",
                "algorithmName": "AI随手拍"
            }
        ]
    }

@app.post("/detect_tower")
async def detect(request: ImageRequest):
    try:
        # 在每次请求前清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        start_time = time.time()
        # opt.source = base64_to_cv2_image(request.image)
        if request.type == "0":  # base64
            opt.source = base64_to_cv2_image(request.image)
        elif request.type == "1":  # url
            opt.source = url_to_cv2_image(request.image)
        detections = run(opt, model)
        output = postprocess(detections)
        inference_time = (time.time() - start_time) * 1000
        print(f"[INFO] Total inference time: {inference_time:.1f}ms")
        
        # 将推理时间添加到返回结果中
        # output["data"]["inference_time"] = round(inference_time, 2)
        return output
    except Exception as e:
        print(f"[ERROR] Inference failed: {str(e)}")
        return {
            "code": -1,
            "msg": "fail",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)