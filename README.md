# TensorFlow C++ and Python Image Recognition Demo

## Description

This demo uses a YOLOV5 onnx model to classify video files that are passed
in on the command line.

## To run

```bash
$ python3 infer_video.py --weights yolov5s.onnx  --imgsz 480 --conf-thres 0.5 --source data/video --visualize 
```

And get result similar to this:
```
infer_video: weights=['yolov5s.onnx'], source=data/video, imgsz=[480, 480], conf_thres=0.5, view_img=False, visualize=True, nosave=False
YOLOv5  d3323d5 Python-3.9.13 torch-2.0.1+cpu CPU

Loading yolov5s.onnx for ONNX Runtime inference...
Speed: 0.3ms pre-process, 48.8ms inference, 1.7ms NMS per image at shape (1, 3, 480, 480)
Results saved to runs\inference\exp
```

