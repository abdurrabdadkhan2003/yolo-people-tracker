import sys

print("Python version:", sys.version)

try:
    import torch
    print("Torch version:", torch.__version__)
except Exception as e:
    print("Torch import error:", e)

try:
    from ultralytics import YOLO
    print("Ultralytics YOLO import successful")
except Exception as e:
    print("Ultralytics import error:", e)

try:
    import cv2
    print("OpenCV version:", cv2.__version__)
except Exception as e:
    print("OpenCV import error:", e)
