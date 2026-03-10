"""
ODT-1 | Object Detection and Classification with Computer Vision
SSD + MobileNetV3 Object Detection Model

Authors : Abdul WAHAB (1447523), Muhammad Faraz Abbasi (1566440)
Course  : Autonomous Intelligent Systems — WS 25/26
Guided by: Prof. Peter Nauth

Architecture : SSD head + MobileNetV3-Large backbone (Google, 2020)
Config file  : ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
Weights file : frozen_inference_graph.pb
Pre-trained on: MS-COCO (80 classes)
"""

import cv2
import numpy as np
import time
import os

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
COCO_NAMES  = os.path.join(WEIGHTS_DIR, "coco.names")
SSD_PBTXT   = os.path.join(WEIGHTS_DIR, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
SSD_PB      = os.path.join(WEIGHTS_DIR, "frozen_inference_graph.pb")

# Hyperparameters
CONFIDENCE_THRESHOLD = 0.5
INPUT_WIDTH          = 320
INPUT_HEIGHT         = 320
SCALE                = 1 / 127.5
MEAN                 = (127.5, 127.5, 127.5)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_class_names(path: str) -> list:
    """Load COCO class labels."""
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_ssd(pbtxt: str, pb: str):
    """Load SSD-MobileNetV3 via OpenCV DNN."""
    net = cv2.dnn_DetectionModel(pb, pbtxt)
    net.setInputSize(INPUT_WIDTH, INPUT_HEIGHT)
    net.setInputScale(SCALE)
    net.setInputMean(MEAN)
    net.setInputSwapRB(True)
    return net


def run_inference(net, frame: np.ndarray):
    """
    Run SSD-MobileNetV3 inference on a single frame.
    Returns (class_ids, confidences, boxes, cx_list, cy_list, inference_time).
    """
    t0 = time.time()
    class_ids, confidences, boxes = net.detect(
        frame, confThreshold=CONFIDENCE_THRESHOLD
    )
    inference_time = time.time() - t0

    cx_list, cy_list = [], []
    clean_ids, clean_conf, clean_boxes = [], [], []

    if len(class_ids) > 0:
        for cid, conf, box in zip(class_ids.flatten(),
                                   confidences.flatten(),
                                   boxes):
            x, y, bw, bh = box
            cx = x + bw // 2
            cy = y + bh // 2
            cx_list.append(cx)
            cy_list.append(cy)
            clean_ids.append(int(cid) - 1)   # COCO labels are 1-indexed
            clean_conf.append(float(conf))
            clean_boxes.append([x, y, bw, bh])

    return clean_ids, clean_conf, clean_boxes, cx_list, cy_list, inference_time


def draw_detections(frame, class_ids, confidences, boxes, class_names,
                    cx_list, cy_list):
    """Draw bounding boxes, labels, and center coordinates."""
    np.random.seed(99)
    colors = np.random.randint(0, 255, size=(len(class_names) + 1, 3), dtype="uint8")

    for i, box in enumerate(boxes):
        x, y, bw, bh = box
        cid   = class_ids[i]
        color = [int(c) for c in colors[cid % len(colors)]]
        name  = class_names[cid] if cid < len(class_names) else "unknown"
        label = f"{name}: {confidences[i]*100:.1f}%"
        coord = f"cx={cx_list[i]}, cy={cy_list[i]}"

        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(frame, label, (x, max(y - 10, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, coord, (x, y + bh + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return frame


# ─────────────────────────────────────────────
# Main — Real-Time Detection
# ─────────────────────────────────────────────
def main(source: int = 0):
    for path in (SSD_PBTXT, SSD_PB, COCO_NAMES):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Please download model weights — see README.md for instructions."
            )

    class_names = load_class_names(COCO_NAMES)
    net         = load_ssd(SSD_PBTXT, SSD_PB)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    print("[SSD-MobileNetV3] Running — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ids, confs, boxes, cx, cy, t = run_inference(net, frame)
        frame = draw_detections(frame, ids, confs, boxes, class_names, cx, cy)

        fps_label = f"FPS: {1/t:.1f}" if t > 0 else "FPS: --"
        cv2.putText(frame, fps_label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("SSD-MobileNetV3 | ODT-1", frame)

        # Output for robot arm controller
        for i in range(len(boxes)):
            name = class_names[ids[i]] if ids[i] < len(class_names) else "unknown"
            print(f"  [{name}]  conf={confs[i]*100:.1f}%  "
                  f"cx={cx[i]}  cy={cy[i]}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SSD-MobileNetV3 Real-Time Detector — ODT-1")
    p.add_argument("--source", default=0,
                   help="Camera index or video file path (default: 0)")
    args = p.parse_args()
    src = args.source if isinstance(args.source, str) and not args.source.isdigit() \
          else int(args.source)
    main(source=src)
