"""
ODT-1 | Object Detection and Classification with Computer Vision
YOLOv3 + DarkNet53 Object Detection Model

Authors : Abdul WAHAB (1447523), Muhammad Faraz Abbasi (1566440)
Course  : Autonomous Intelligent Systems — WS 25/26
Guided by: Prof. Peter Nauth
"""

import cv2
import numpy as np
import time
import os

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
WEIGHTS_DIR   = os.path.join(os.path.dirname(__file__), "..", "weights")
COCO_NAMES    = os.path.join(WEIGHTS_DIR, "coco.names")

YOLOV3_CFGS = {
    320: (os.path.join(WEIGHTS_DIR, "yolov3.cfg"),
          os.path.join(WEIGHTS_DIR, "yolov3-320.weights")),
    416: (os.path.join(WEIGHTS_DIR, "yolov3.cfg"),
          os.path.join(WEIGHTS_DIR, "yolov3-416.weights")),
    608: (os.path.join(WEIGHTS_DIR, "yolov3.cfg"),
          os.path.join(WEIGHTS_DIR, "yolov3-608.weights")),
}

CONFIDENCE_THRESHOLD = 0.5   # 50 % — as used in the project
NMS_THRESHOLD        = 0.3   # Non-Maximum Suppression threshold


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def load_class_names(path: str) -> list:
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def load_yolov3(cfg_path: str, weights_path: str):
    """Load YOLOv3 network via OpenCV DNN."""
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def get_output_layers(net):
    layer_names = net.getLayerNames()
    # OpenCV ≥4.x returns a flat array of ints
    unconnected = net.getUnconnectedOutLayers()
    if isinstance(unconnected[0], (list, np.ndarray)):
        return [layer_names[i[0] - 1] for i in unconnected]
    return [layer_names[i - 1] for i in unconnected]


def run_inference(net, frame: np.ndarray, input_size: int):
    """
    Forward pass through YOLOv3.
    Returns (boxes, confidences, class_ids, center_x_list, center_y_list).
    """
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (input_size, input_size),
        swapRB=True, crop=False
    )
    net.setInput(blob)
    t0 = time.time()
    layer_outputs = net.forward(get_output_layers(net))
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

            # Scale bounding box back to original frame size
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

    # Apply NMS
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
    """Draw bounding boxes and labels on frame."""
    np.random.seed(42)
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
# Main — Real-Time Detection
# ─────────────────────────────────────────────
def main(input_size: int = 416, source: int = 0, verbose: bool = False):
    """
    Run YOLOv3 object detection on a live webcam feed.

    Parameters
    ----------
    input_size : int
        Network input resolution (320 | 416 | 608).
    source : int or str
        Camera index (0 = default webcam) or video file path.
    verbose : bool
        When True, print detection results to console each frame.
    """
    if input_size not in YOLOV3_CFGS:
        raise ValueError(f"Unsupported input size {input_size}. Choose 320, 416, or 608.")

    cfg_path, weights_path = YOLOV3_CFGS[input_size]

    for path in (cfg_path, weights_path, COCO_NAMES):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Required file not found: {path}\n"
                f"Place the file at: {os.path.abspath(path)}\n"
                "Download model weights — see README.md for instructions."
            )

    class_names = load_class_names(COCO_NAMES)
    net         = load_yolov3(cfg_path, weights_path)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    print(f"[YOLOv3-{input_size}] Running — press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        boxes, confs, ids, cx, cy, t = run_inference(net, frame, input_size)
        frame = draw_detections(frame, boxes, confs, ids, class_names, cx, cy)

        fps_label = f"FPS: {1/t:.1f}" if t > 0 else "FPS: --"
        cv2.putText(frame, fps_label, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(f"YOLOv3-{input_size} | ODT-1", frame)

        # Print detections to console (for robot arm controller)
        if verbose:
            for i in range(len(boxes)):
                print(f"  [{class_names[ids[i]]}]  conf={confs[i]*100:.1f}%  "
                      f"cx={cx[i]}  cy={cy[i]}")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="YOLOv3 Real-Time Object Detector — ODT-1")
    p.add_argument("--size",    type=int, default=416, choices=[320, 416, 608],
                   help="Network input size (default: 416)")
    p.add_argument("--source",  default=0,
                   help="Camera index or video file path (default: 0)")
    p.add_argument("--verbose", action="store_true",
                   help="Print detection results to console each frame")
    args = p.parse_args()

    src = args.source if isinstance(args.source, str) and not args.source.isdigit() \
          else int(args.source)
    main(input_size=args.size, source=src, verbose=args.verbose)
