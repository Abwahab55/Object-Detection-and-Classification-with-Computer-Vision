"""
ODT-1 | Object Detection and Classification with Computer Vision
Tiny-YOLOv3 + DarkNet19 Object Detection Model

Authors : Abdul WAHAB (1447523), Muhammad Faraz Abbasi (1566440)
Course  : Autonomous Intelligent Systems — WS 25/26
Guided by: Prof. Peter Nauth

Architecture : 13 convolutional layers + 8 max-pooling layers
Output scales : 13×13  and  26×26
Pre-trained on: MS-COCO (80 classes)
"""

import cv2
import numpy as np
import time
import os

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
WEIGHTS_DIR  = os.path.join(os.path.dirname(__file__), "..", "weights")
COCO_NAMES   = os.path.join(WEIGHTS_DIR, "coco.names")
TINY_CFG     = os.path.join(WEIGHTS_DIR, "yolov3-tiny.cfg")
TINY_WEIGHTS = os.path.join(WEIGHTS_DIR, "yolov3-tiny.weights")

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD        = 0.3


# ─────────────────────────────────────────────
# Helpers  (shared logic with YOLOv3)
# ─────────────────────────────────────────────
def load_class_names(path: str) -> list:
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def get_output_layers(net):
    layer_names  = net.getLayerNames()
    unconnected  = net.getUnconnectedOutLayers()
    if isinstance(unconnected[0], (list, np.ndarray)):
        return [layer_names[i[0] - 1] for i in unconnected]
    return [layer_names[i - 1] for i in unconnected]


def run_inference(net, frame: np.ndarray, input_size: int = 416):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (input_size, input_size),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    t0             = time.time()
    layer_outputs  = net.forward(get_output_layers(net))
    inference_time = time.time() - t0

    boxes, confidences, class_ids = [], [], []
    cx_list, cy_list = [], []

    for output in layer_outputs:
        for detection in output:
            scores     = detection[5:]
            class_id   = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            cx = int(detection[0] * w)
            cy = int(detection[1] * h)
            bw = int(detection[2] * w)
            bh = int(detection[3] * h)
            x  = cx - bw // 2
            y  = cy - bh // 2

            boxes.append([x, y, bw, bh])
            confidences.append(confidence)
            class_ids.append(class_id)
            cx_list.append(cx)
            cy_list.append(cy)

    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD
    )

    final_boxes, final_conf, final_ids = [], [], []
    final_cx,    final_cy              = [], []

    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(boxes[i])
            final_conf.append(confidences[i])
            final_ids.append(class_ids[i])
            final_cx.append(cx_list[i])
            final_cy.append(cy_list[i])

    return final_boxes, final_conf, final_ids, final_cx, final_cy, inference_time


def draw_detections(frame, boxes, confidences, class_ids, class_names, cx_list, cy_list):
    np.random.seed(7)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype="uint8")

    for i, box in enumerate(boxes):
        x, y, bw, bh = box
        color = [int(c) for c in colors[class_ids[i]]]
        label = f"{class_names[class_ids[i]]}: {confidences[i]*100:.1f}%"
        coord = f"cx={cx_list[i]}, cy={cy_list[i]}"

        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, coord, (x, y + bh + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return frame


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(source: int = 0):
    for path in (TINY_CFG, TINY_WEIGHTS, COCO_NAMES):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                "Please download model weights — see README.md for instructions."
            )

    class_names = load_class_names(COCO_NAMES)
    net = cv2.dnn.readNetFromDarknet(TINY_CFG, TINY_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    print("[Tiny-YOLOv3] Running — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, confs, ids, cx, cy, t = run_inference(net, frame)
        frame = draw_detections(frame, boxes, confs, ids, class_names, cx, cy)

        fps_label = f"FPS: {1/t:.1f}" if t > 0 else "FPS: --"
        cv2.putText(frame, fps_label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Tiny-YOLOv3 | ODT-1", frame)

        for i in range(len(boxes)):
            print(f"  [{class_names[ids[i]]}]  conf={confs[i]*100:.1f}%  "
                  f"cx={cx[i]}  cy={cy[i]}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Tiny-YOLOv3 Real-Time Detector — ODT-1")
    p.add_argument("--source", default=0,
                   help="Camera index or video file path (default: 0)")
    args = p.parse_args()
    src = args.source if isinstance(args.source, str) and not args.source.isdigit() \
          else int(args.source)
    main(source=src)
