PCB Defect Detector - deploy bundle
====================================

Files:
  best.pt      - native ultralytics/PyTorch weights
  best.onnx    - portable ONNX export (use this unless you need .pt specifically)
  classes.txt  - class names, in label-index order
  predict.py   - minimal standalone inference script (CPU by default)

Setup on the new machine:
  pip install ultralytics onnxruntime
  python predict.py your_image.jpg

Classes: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper
