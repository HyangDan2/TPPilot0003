YOLOv5s for personal use repository

Use pt files in

https://github.com/miladsoltany/Face-Detection

Use YOLOv5s files in 

https://github.com/ultralytics/yolov5

Save pt file in weights/ folder

then 

python export_test.py 
reference in --> [1,25200,6] format

by using result face.onnx file

then use

python main.py --model onnx --sources "path/to/your/file.jpg"

yolov5s-face is created by using above pt files and YOLOv5s