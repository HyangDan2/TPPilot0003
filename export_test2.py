# export_onnx_face_detect_and_preview.py
import torch
import torch.nn as nn
from models.experimental import attempt_load

weights = './weights/yolov5s-face.pt'
imgsz   = (640, 640)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

try:
    core = attempt_load(weights, device=device)       # new sig
except TypeError:
    core = attempt_load(weights, map_location=device) # old sig
core.to(device).eval()

class DetectPlusPreview(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
    def forward(self, x):
        y = self.inner(x)
        # YOLOv5 계열은 보통 (pred, ...) 형태이므로 첫 요소를 예측으로 사용
        pred = y[0] if isinstance(y, (list, tuple)) else y
        img_nhwc = x.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC
        # 1번 출력: Detect (기존 뷰어 호환)
        # 2번 출력: 미리보기용 이미지
        return pred, img_nhwc

model = DetectPlusPreview(core).to(device).eval()

dummy = torch.zeros(1, 3, *imgsz, device=device)

torch.onnx.export(
    model, dummy, 'yolov5s-face_detect+preview.onnx',
    input_names=['images'],
    output_names=['output', 'images_nhwc'],   # 첫 출력 이름을 'output'으로 유지 (호환!)
    opset_version=12,
    do_constant_folding=True,
    dynamic_axes={
        'images':      {0: 'batch', 2: 'height', 3: 'width'},
        'output':      {0: 'batch'},           # 감지 텐서는 구현마다 축이 다르므로 batch만 동적
        'images_nhwc': {0: 'batch', 1: 'height', 2: 'width'}
    }
)

print('OK -> yolov5s-face_detect+preview.onnx  (outputs: Detect first, NHWC image second)')
