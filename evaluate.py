"""Evaluate best.pt: confusion matrix, per-class P/R/F1, PR/F1 curves, training curves.

Usage: .venv/bin/python evaluate.py
Outputs go to ./eval_results/
"""
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATASET = Path.home() / ".cache/kagglehub/datasets/akhatova/pcb-defects/versions/1/PCB_DATASET"
WORK = ROOT / "working" / "eval_data"
OUT = ROOT / "eval_results"
CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]
SIZE = 768
DEVICE = "cpu"


def build_dataset():
    """Same preprocessing as train.ipynb: letterbox to 768 at top-left, gray 114 pad."""
    img_out, lbl_out = WORK / "images" / "val", WORK / "labels" / "val"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    n = 0
    for xml in sorted((DATASET / "Annotations").rglob("*.xml")):
        root = ET.parse(xml).getroot()
        fname = root.find("filename").text
        src = DATASET / "images" / xml.parent.name / fname
        w, h = int(root.find("size/width").text), int(root.find("size/height").text)
        scale = min(SIZE / w, SIZE / h)
        img = cv2.imread(str(src))
        nw, nh = int(w * scale), int(h * scale)
        canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(str(img_out / fname), canvas)
        lines = []
        for obj in root.findall("object"):
            c = CLASSES.index(obj.find("name").text)
            b = [int(obj.find(f"bndbox/{k}").text) * scale for k in ("xmin", "ymin", "xmax", "ymax")]
            b = [int(v) for v in b]
            lines.append(f"{c} {(b[0]+b[2])/(2*SIZE)} {(b[1]+b[3])/(2*SIZE)} "
                         f"{(b[2]-b[0])/SIZE} {(b[3]-b[1])/SIZE}")
        (lbl_out / f"{Path(fname).stem}.txt").write_text("\n".join(lines) + "\n")
        n += 1
    yaml = WORK / "data.yaml"
    yaml.write_text(f"path: {WORK.as_posix()}\ntrain: images/val\nval: images/val\n"
                    f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print(f"built {n} images")
    return yaml


def f1(p, r):
    return 2 * p * r / (p + r) if (p + r) else 0.0


def evaluate(yaml):
    model = YOLO(str(ROOT / "best.pt"))
    m = model.val(data=str(yaml), imgsz=SIZE, batch=8, device=DEVICE, conf=0.001, iou=0.7,
                  plots=True, project=str(OUT), name="ultralytics", exist_ok=True)
    # Confusion matrix at the deployment threshold (conf=0.25, same as predict.py).
    # At conf=0.001 low-score boxes steal GT matches and distort the matrix.
    model.val(data=str(yaml), imgsz=SIZE, batch=8, device=DEVICE, conf=0.25, iou=0.7,
              plots=True, project=str(OUT), name="confusion_conf0.25", exist_ok=True)
    rows = []
    for i, ci in enumerate(m.box.ap_class_index):
        p, r, ap50, ap = m.box.class_result(i)
        rows.append(dict(class_name=CLASSES[ci], precision=p, recall=r, f1=f1(p, r),
                         mAP50=ap50, mAP50_95=ap))
    df = pd.DataFrame(rows)
    # instance support
    sup = np.zeros(len(CLASSES), int)
    for f in (WORK / "labels/val").glob("*.txt"):
        for l in f.read_text().split("\n"):
            if l.strip():
                sup[int(l.split()[0])] += 1
    df["support"] = df["class_name"].map(dict(zip(CLASSES, sup)))
    # macro avg and ultralytics "all" row (P/R at max-F1 confidence)
    allrow = dict(class_name="ALL (macro)", precision=df.precision.mean(), recall=df.recall.mean(),
                  f1=df.f1.mean(), mAP50=m.box.map50, mAP50_95=m.box.map, support=int(sup.sum()))
    df = pd.concat([df, pd.DataFrame([allrow])], ignore_index=True)
    df.to_csv(OUT / "per_class_metrics.csv", index=False)
    return m, df


def plot_metric_bars(df):
    d = df[df.class_name != "ALL (macro)"]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, (col, lab) in enumerate([("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")]):
        bars = ax.bar(x + (k - 1) * 0.27, d[col], 0.27, label=lab)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f"{b.get_height():.3f}",
                    ha="center", fontsize=7)
    ax.set_xticks(x, d.class_name, rotation=20)
    ax.set_ylim(0.8, 1.02)
    ax.set_ylabel("score")
    ax.set_title("Per-class precision / recall / F1")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "per_class_prf1.png", dpi=150)
    plt.close(fig)


def parse_log():
    """Per-epoch held-out metrics for each fold from log.txt, plus final per-class table."""
    text = (ROOT / "log.txt").read_text(errors="ignore")
    folds = re.split(r"TRAINING FOLD (\d)/3", text)
    epochs, perclass = [], []
    for i in range(1, len(folds), 2):
        k, body = int(folds[i]), folds[i + 1]
        for m in re.finditer(r"^\s+all\s+\d+\s+\d+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$",
                             body, re.M):
            epochs.append(dict(fold=k, P=float(m[1]), R=float(m[2]), mAP50=float(m[3]), mAP50_95=float(m[4])))
        # final per-class block (after the last "Validating")
        tail = body[body.rfind("Validating"):]
        for m in re.finditer(r"^\s+(" + "|".join(CLASSES) + r")\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+"
                             r"([\d.]+)\s+([\d.]+)", tail, re.M):
            p, r = float(m[4]), float(m[5])
            perclass.append(dict(fold=k, class_name=m[1], images=int(m[2]), instances=int(m[3]),
                                 precision=p, recall=r, f1=f1(p, r), mAP50=float(m[6]), mAP50_95=float(m[7])))
    ep = pd.DataFrame(epochs)
    ep["epoch"] = ep.groupby("fold").cumcount() + 1
    ep["F1"] = [f1(p, r) for p, r in zip(ep.P, ep.R)]
    ep.to_csv(OUT / "training_curves_by_fold.csv", index=False)
    pc = pd.DataFrame(perclass)
    pc.to_csv(OUT / "heldout_per_class_by_fold.csv", index=False)
    return ep, pc


def plot_training(ep):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, col in zip(axes, ["P", "R", "F1", "mAP50_95"]):
        for k, g in ep.groupby("fold"):
            ax.plot(g.epoch, g[col], label=f"fold {k}")
        ax.set_title(col)
        ax.set_xlabel("epoch")
        ax.set_ylim(0.5 if col != "mAP50_95" else 0.2, 1.0 if col != "mAP50_95" else 0.6)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle("Held-out validation metrics per epoch (from log.txt)")
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    yaml = WORK / "data.yaml"
    if not yaml.exists():
        yaml = build_dataset()
    m, df = evaluate(yaml)
    plot_metric_bars(df)
    ep, pc = parse_log()
    plot_training(ep)
    pd.set_option("display.width", 200)
    print("\n=== best.pt on all 693 images (see caveat: not a clean held-out set) ===")
    print(df.round(4).to_string(index=False))
    print("\n=== Held-out per-class, by fold (from log.txt, final validation) ===")
    print(pc.round(4).to_string(index=False))
    print("\nMean across folds:")
    print(pc.groupby("class_name")[["precision", "recall", "f1", "mAP50", "mAP50_95"]].mean().round(4))
    print(pc.groupby("fold")[["precision", "recall", "f1", "mAP50", "mAP50_95"]].mean().round(4))


if __name__ == "__main__":
    main()
