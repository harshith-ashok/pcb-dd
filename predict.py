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
