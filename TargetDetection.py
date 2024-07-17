import torch
import sys
import cv2
import numpy as np
from pathlib import Path

# link to yolov5
yolov5_path = r'./yolov5'
sys.path.append(yolov5_path)

from models.common import DetectMultiBackend
from utils.torch_utils import select_device
from utils.general import (check_img_size, non_max_suppression, scale_coords, xyxy2xywh)
from utils.augmentations import letterbox
from utils.plots import Annotator, colors

np.random.seed(0)

# Load model
weights = Path(r'./yolov5/weights/s_best.pt')
data = Path(r'./yolov5/data/myVisDrone.yaml')
imgsz = (1280, 1280)
# device = select_device('gpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DetectMultiBackend(weights, device=device, dnn=False, data=data)
# stride = 32，names是各类别的名称
stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
imgsz = check_img_size(imgsz, s=stride)  # check image size

def RemoveOverlaps(All, xy):
    x = 0.5*(xy[0] + xy[2])
    y = 0.5 * (xy[1] + xy[3])
    for D in All:
        if abs(0.5*(D[2] + D[4]) - x) < 30 and abs(0.5*(D[3] + D[5]) - y) < 30:
            return 0
    return 1

def detection_target(im0, type):

    # 返回识别目标的坐标信息
    targetinformation = []
    num = 0
    # Padded resize
    im = letterbox(im0, imgsz, stride=stride, auto=True)[0]

    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=False)  # warmup # NCHW
    im = torch.from_numpy(im).to(device)
    im = im.float()
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im, augment=False, visualize=False)
    # NMS
    pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.45, max_det=1000)  # a list of tensor
    # Process predictions
    det = pred[0]
    annotator = Annotator(im0, line_width=5, example=str(names))
    if len(det):
        # Rescale boxes from img_size to im0 size
        # im.shape现在是NCHW，所以取了后两位
        det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

        # Write results
        # 预测结果的tensor，形状n * 6，分别是xyxy, conf, cls
        # reversed 函数返回一个反转的迭代器。
        # 这个reversed，是从最后一个检测结果开始倒着处理
        for *xyxy, conf, cls in reversed(det):
            # Add bbox to image
            c = int(cls)  # integer class
            label = f'{num} {names[c]} {conf:.2f}'
            ab = names[c]
            # print(xyxy)
            if ab in type:
                # print(xyxy)
                if RemoveOverlaps(targetinformation, xyxy):
                    targetinformation.append([num, ab, xyxy[0], xyxy[1], xyxy[2], xyxy[3]])
                    # color = np.random.randint(0, 255, (300, 3))
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    num = num + 1
    im0 = annotator.result()
    return im0, targetinformation




