#!/usr/bin/env python3
"""
ODT-1 — Download all required model weights.
Run this script once before using any detector.

    python download_weights.py
"""

import os
import urllib.request

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

FILES = [
    # YOLOv3 config + weights (3 sizes)
    ("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
     "yolov3.cfg"),
    ("https://pjreddie.com/media/files/yolov3.weights",
     "yolov3-416.weights"),   # same file — copy for 320 & 608 too

    # Tiny-YOLOv3
    ("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
     "yolov3-tiny.cfg"),
    ("https://pjreddie.com/media/files/yolov3-tiny.weights",
     "yolov3-tiny.weights"),

    # COCO class names
    ("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
     "coco.names"),
]

SSD_NOTICE = """
──────────────────────────────────────────────────────────────────────
SSD-MobileNetV3 weights must be downloaded manually from TensorFlow Hub:

  Model page:
    https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v3

  Required files (place in the weights/ folder):
    • frozen_inference_graph.pb
    • ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt

  Quick download via terminal:
    wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz
    tar -xzf ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz
    cp ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb weights/
    # pbtxt file available at:
    # https://gist.github.com/dkurt/54a8e8b51beb3bd3f770b79e56927bd7
──────────────────────────────────────────────────────────────────────
"""


def download(url: str, dest: str):
    if os.path.isfile(dest):
        print(f"  [skip] {os.path.basename(dest)} already exists")
        return
    print(f"  Downloading {os.path.basename(dest)} ...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / 1e6
        print(f"done  ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"FAILED: {e}")


if __name__ == "__main__":
    print("\nODT-1 — Downloading model weights\n")
    for url, fname in FILES:
        download(url, os.path.join(WEIGHTS_DIR, fname))

    # YOLOv3 weights file is the same binary for all three sizes
    src = os.path.join(WEIGHTS_DIR, "yolov3-416.weights")
    for alias in ("yolov3-320.weights", "yolov3-608.weights"):
        dst = os.path.join(WEIGHTS_DIR, alias)
        if os.path.isfile(src) and not os.path.isfile(dst):
            import shutil
            shutil.copy(src, dst)
            print(f"  [copy] {alias}")

    print(SSD_NOTICE)
    print("Done.  You can now run the detectors.\n")
