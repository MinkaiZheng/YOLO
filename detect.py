#!/usr/bin/env python
# YOLOv11 inference script for object detection
import cv2
import json
import torch
import argparse
import numpy as np
from ultralytics import YOLO
import time

# 启动 CUDNN 加速
torch.backends.cudnn.benchmark = True
# 使用确定性算法提高一致性 - 设为True会降低速度但增加一致性
torch.backends.cudnn.deterministic = True
# 设置GPU优先级
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # 使用主GPU
    # 可选：设置较大的GPU缓存，避免频繁内存分配
    torch.cuda.empty_cache()

def run(opt, model):
    # 在每次推理前同步GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # 记录模型推理起始时间
    inference_start = time.time()
    
    # Run inference
    results = model.predict(
        source=opt.source,
        imgsz=opt.imgsz,
        conf=opt.conf,
        iou=opt.iou,
        device=opt.device,
        stream=False,
        verbose=False  # 减少不必要的日志输出
    )
    
    # 同步GPU完成推理
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    # 记录模型推理实际用时
    inference_time = (time.time() - inference_start) * 1000
    print(f"[INFO] Pure model inference time: {inference_time:.1f}ms")

    # Process and return detection results
    detections = []
    for i, result in enumerate(results):
        # 可视化结果
        # image = result.plot()
        # cv2.imshow(f"Detection Result - {i}", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Get detection info
        boxes = result.boxes  # Boxes object for bbox outputs

        # Convert boxes to list of detections
        for j, box in enumerate(boxes):
            cls_id = int(box.cls.item())
            cls_name = model.names[cls_id]
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            # Save detection info
            detection = {
                'class_id': cls_id,
                'class_name': cls_name,
                'confidence': conf,
            }
            detections.append(detection)
    return detections

def postprocess(detections):
    # 初始化状态
    has_weed = False
    has_warning_sign = False
    has_tower_door = False
    tower_door_status = None
    # 遍历检测结果
    for detection in detections:
        class_name = detection['class_name']
        if class_name == '杂草等易燃物':
            has_weed = True
        if class_name == '警示牌':
            has_warning_sign = True
        if class_name == '塔门打开':
            has_tower_door = True
            tower_door_status = 'open'
        elif class_name == '塔门关闭':
            has_tower_door = True
            tower_door_status = 'closed'

    # 构建结果列表
    # results = []
    # # 塔门状态判断
    # if tower_door_status == 'open':
    #     results.append("塔门打开")
    # # 杂草判断
    # if has_weed:
    #     results.append("有杂草等易燃物")
    # # 警示牌判断
    # if not has_warning_sign:
    #     results.append("无警示牌")
    
    # 计算状态码和消息
    code = 0  # 默认成功
    msg = "success"
    
    # 返回标准格式的结果
    output = {
        "code": code,
        "msg": msg,
        "data": {
            "has_weed": has_weed,
            "has_warning_sign": has_warning_sign,
            "has_tower_door": has_tower_door,
            "tower_door_status": tower_door_status
            # "abnormal_items": results,
            # "result_text": "，".join(results)
        }
    }
    return output
