"""
Standalone PCB defect detector inference.
On the target machine: pip install ultralytics onnxruntime
(ultralytics loads either best.pt or best.onnx and handles
pre/post-processing + NMS for you - no training code needed.)

Runs on CPU by default so it works on any machine regardless of
GPU/CUDA setup. If the target machine has a matching CUDA + cuDNN
install, change DEVICE below to 0 for GPU inference.

Usage: python predict.py path/to/image.jpg
"""
import sys
from ultralytics import YOLO

MODEL_PATH = "best.onnx"  # or "best.pt"
DEVICE = "cpu"


def main(image_path):
    model = YOLO(MODEL_PATH, task="detect")
    results = model(image_path, conf=0.25, device=DEVICE)
    for r in results:
        r.save(filename="prediction.jpg")
        for box in r.boxes:
            cls_name = model.names[int(box.cls)]
            conf = float(box.conf)
            print(f"{cls_name}: {conf:.2f}")
    print("Annotated image saved to prediction.jpg")


if __name__ == "__main__":
    main(sys.argv[1])
