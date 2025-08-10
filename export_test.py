# export_onnx_face.py
import torch
from models.experimental import attempt_load

weights = './weights/yolov5s-face.pt'   # 경로 맞춰주세요
imgsz   = (640, 640)

# 'cuda:0' 가능하면 사용, 아니면 cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# attempt_load 시그니처 호환 처리 (구버전 map_location, 신버전 device)
try:
    model = attempt_load(weights, device=device)            # 신버전
except TypeError:
    model = attempt_load(weights, map_location=device)      # 구버전

model.to(device).eval()

# 더미 입력 (NCHW)
dummy = torch.zeros(1, 3, *imgsz, device=device)

# ONNX 내보내기
torch.onnx.export(
    model, dummy, 'yolov5s-face.onnx',
    input_names=['images'],
    output_names=['output'],
    opset_version=12,
    do_constant_folding=True,
    dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},
                  'output': {0: 'batch'}}   # 런타임에서 크기 바뀌는 경우 안전
)

print('OK -> yolov5s-face.onnx')