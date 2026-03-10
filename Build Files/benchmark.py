"""
ODT-1 | Object Detection and Classification with Computer Vision
Model Benchmark — Compares FPS and inference time for all three models

Authors : Abdul WAHAB (1447523), Muhammad Faraz Abbasi (1566440)
Course  : Autonomous Intelligent Systems — WS 24/25

Usage:
    python benchmark.py --source 0 --frames 100
"""

import cv2
import numpy as np
import time
import os
import argparse

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")
COCO_NAMES  = os.path.join(WEIGHTS_DIR, "coco.names")


def _output_layers(net):
    names = net.getLayerNames()
    out   = net.getUnconnectedOutLayers()
    if isinstance(out[0], (list, np.ndarray)):
        return [names[i[0] - 1] for i in out]
    return [names[i - 1] for i in out]


def benchmark_model(name, net, frame, input_size=416, use_dnn_detect=False, n=50):
    times = []
    for _ in range(n):
        if use_dnn_detect:
            t0 = time.time()
            net.detect(frame, confThreshold=0.5)
            times.append(time.time() - t0)
        else:
            blob = cv2.dnn.blobFromImage(
                frame, 1/255.0, (input_size, input_size), swapRB=True, crop=False
            )
            net.setInput(blob)
            t0 = time.time()
            net.forward(_output_layers(net))
            times.append(time.time() - t0)

    avg_ms  = sum(times) / len(times) * 1000
    avg_fps = 1.0 / (sum(times) / len(times))
    print(f"  {name:<30} avg: {avg_ms:6.1f} ms   FPS: {avg_fps:6.1f}")
    return avg_ms, avg_fps


def main(source=0, n_frames=50):
    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("[!] Could not read frame from source.")
        return

    print(f"\n{'='*60}")
    print(f"  ODT-1 — Model Benchmark  ({n_frames} frames each)")
    print(f"{'='*60}")

    results = {}

    # ── YOLOv3-320 ──────────────────────────────────────────────
    cfg     = os.path.join(WEIGHTS_DIR, "yolov3.cfg")
    weights = os.path.join(WEIGHTS_DIR, "yolov3-320.weights")
    if os.path.isfile(cfg) and os.path.isfile(weights):
        net = cv2.dnn.readNetFromDarknet(cfg, weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        ms, fps = benchmark_model("YOLOv3-320", net, frame, 320, n=n_frames)
        results["YOLOv3-320"] = {"avg_ms": ms, "fps": fps}
    else:
        print("  YOLOv3-320              weights not found — skipped")

    # ── YOLOv3-416 ──────────────────────────────────────────────
    weights = os.path.join(WEIGHTS_DIR, "yolov3-416.weights")
    if os.path.isfile(cfg) and os.path.isfile(weights):
        net = cv2.dnn.readNetFromDarknet(cfg, weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        ms, fps = benchmark_model("YOLOv3-416", net, frame, 416, n=n_frames)
        results["YOLOv3-416"] = {"avg_ms": ms, "fps": fps}
    else:
        print("  YOLOv3-416              weights not found — skipped")

    # ── YOLOv3-608 ──────────────────────────────────────────────
    weights = os.path.join(WEIGHTS_DIR, "yolov3-608.weights")
    if os.path.isfile(cfg) and os.path.isfile(weights):
        net = cv2.dnn.readNetFromDarknet(cfg, weights)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        ms, fps = benchmark_model("YOLOv3-608", net, frame, 608, n=n_frames)
        results["YOLOv3-608"] = {"avg_ms": ms, "fps": fps}
    else:
        print("  YOLOv3-608              weights not found — skipped")

    # ── Tiny-YOLOv3 ─────────────────────────────────────────────
    tiny_cfg = os.path.join(WEIGHTS_DIR, "yolov3-tiny.cfg")
    tiny_w   = os.path.join(WEIGHTS_DIR, "yolov3-tiny.weights")
    if os.path.isfile(tiny_cfg) and os.path.isfile(tiny_w):
        net = cv2.dnn.readNetFromDarknet(tiny_cfg, tiny_w)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        ms, fps = benchmark_model("Tiny-YOLOv3", net, frame, 416, n=n_frames)
        results["Tiny-YOLOv3"] = {"avg_ms": ms, "fps": fps}
    else:
        print("  Tiny-YOLOv3             weights not found — skipped")

    # ── SSD-MobileNetV3 ─────────────────────────────────────────
    ssd_pbtxt = os.path.join(WEIGHTS_DIR, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
    ssd_pb    = os.path.join(WEIGHTS_DIR, "frozen_inference_graph.pb")
    if os.path.isfile(ssd_pbtxt) and os.path.isfile(ssd_pb):
        net = cv2.dnn_DetectionModel(ssd_pb, ssd_pbtxt)
        net.setInputSize(320, 320)
        net.setInputScale(1 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        ms, fps = benchmark_model("SSD-MobileNetV3", net, frame,
                                   use_dnn_detect=True, n=n_frames)
        results["SSD-MobileNetV3"] = {"avg_ms": ms, "fps": fps}
    else:
        print("  SSD-MobileNetV3         weights not found — skipped")

    print(f"{'='*60}\n")

    if results:
        fastest = min(results, key=lambda k: results[k]["avg_ms"])
        print(f"  Fastest model: {fastest}  ({results[fastest]['fps']:.1f} FPS)")

    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="ODT-1 Model Benchmark")
    p.add_argument("--source", default=0, help="Camera index or video file")
    p.add_argument("--frames", type=int, default=50, help="Frames to average over")
    args = p.parse_args()
    src = int(args.source) if str(args.source).isdigit() else args.source
    main(source=src, n_frames=args.frames)
