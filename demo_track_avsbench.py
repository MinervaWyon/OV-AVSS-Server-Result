#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import torch
import numpy as np

# ---------- 添加 YOLOv5 路径 ----------
yolov5_path = '/root/autodl-tmp/yolov5'
if yolov5_path not in sys.path:
    sys.path.insert(0, yolov5_path)

# ---------- 导入 YOLOv5 模块 ----------
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression, scale_boxes
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox

# ---------- 导入 DeepSORT ----------
from deep_sort_realtime.deepsort_tracker import DeepSort

# ---------- 配置 ----------
DATA_ROOT = '/root/autodl-tmp/datasets/AVSBench-openvoc/test/JPEGImages/'
WEIGHTS = '/root/autodl-tmp/yolov5/yolov5s.pt'
OUTPUT_DIR = '/root/autodl-tmp/'
DEVICE = select_device('0')
CONF_THRESH = 0.25
IOU_THRESH = 0.45

tracker = DeepSort(
    max_age=30,
    n_init=3,
    nms_max_overlap=1.0,
    max_iou_distance=0.7,
    max_cosine_distance=0.2,
    embedder='mobilenet',
    half=False,
    bgr=True,
    embedder_gpu=True
)
# -------------------------

def main():
    if not os.path.isdir(DATA_ROOT):
        print(f"错误：数据集目录不存在 {DATA_ROOT}")
        return
    if not os.path.exists(WEIGHTS):
        print(f"错误：YOLOv5 权重文件不存在 {WEIGHTS}")
        return

    print("加载 YOLOv5 模型...")
    model = DetectMultiBackend(WEIGHTS, device=DEVICE, dnn=False)
    print(f"模型加载完成，设备：{DEVICE}")

    try:
        video_dirs = [d for d in os.listdir(DATA_ROOT) 
                      if os.path.isdir(os.path.join(DATA_ROOT, d))]
    except Exception as e:
        print(f"读取数据集目录失败：{e}")
        return

    if not video_dirs:
        print(f"在 {DATA_ROOT} 下未找到任何视频子目录。")
        return

    print(f"找到 {len(video_dirs)} 个视频序列，开始处理...")

    for idx, video_name in enumerate(video_dirs):
        video_path = os.path.join(DATA_ROOT, video_name)
        print(f"[{idx+1}/{len(video_dirs)}] 正在处理: {video_name}")

        try:
            frame_files = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
        except Exception as e:
            print(f"  读取帧列表失败：{e}，跳过")
            continue

        if not frame_files:
            print(f"  跳过：文件夹 {video_name} 中没有 .jpg 文件")
            continue

        first_frame = cv2.imread(os.path.join(video_path, frame_files[0]))
        if first_frame is None:
            print(f"  警告：无法读取第一帧，跳过")
            continue
        h, w = first_frame.shape[:2]

        output_file = os.path.join(OUTPUT_DIR, f"output_tracked_{video_name}.mp4")
        out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 10, (w, h))

        for frame_file in frame_files:
            frame_path = os.path.join(video_path, frame_file)
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            orig_h, orig_w = frame.shape[:2]

            # 使用 letterbox 缩放为 32 的倍数
            img_resized = letterbox(frame, new_shape=(640, 640), stride=32)[0]
            img_tensor = torch.from_numpy(img_resized).to(DEVICE).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

            pred = model(img_tensor)
            pred = non_max_suppression(pred, CONF_THRESH, IOU_THRESH, classes=None, agnostic=False)

            detections = []
            if pred[0] is not None and len(pred[0]):
                # 将检测框坐标映射回原始图像尺寸
                pred[0][:, :4] = scale_boxes(img_tensor.shape[2:], pred[0][:, :4], frame.shape).round()
                for *xyxy, conf, cls in pred[0]:
                    x1, y1, x2, y2 = map(int, xyxy)
                    # 边界裁剪
                    x1 = max(0, x1); y1 = max(0, y1); x2 = min(orig_w, x2); y2 = min(orig_h, y2)
                    # 转换为 [x, y, w, h] 格式
                    w_ = x2 - x1
                    h_ = y2 - y1
                    detections.append(([x1, y1, w_, h_], float(conf), int(cls)))

            # 更新 DeepSORT 跟踪器
            if detections:
                # 直接传入格式为 ([x, y, w, h], conf, cls) 的列表
                tracks = tracker.update_tracks(detections, frame=frame)
            else:
                tracks = []

            outputs = []
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                # 获取边界框 [x1, y1, x2, y2]
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                cls_id = track.det_class if hasattr(track, 'det_class') else -1
                outputs.append([x1, y1, x2, y2, track_id, cls_id])

            # 可视化
            for output in outputs:
                x1, y1, x2, y2, track_id, cls_id = output[:6]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID:{track_id}', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            out.write(frame)

        out.release()
        print(f"  完成，输出文件：{output_file}")

    print("所有视频处理完毕！")

if __name__ == "__main__":
    main()