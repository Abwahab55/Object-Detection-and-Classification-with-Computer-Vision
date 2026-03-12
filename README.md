 ## Object Detection and Classification with Computer Vision

## Table of Contents

- [Project Overview](#project-overview)
- [Models Implemented](#models-implemented)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Downloading Model Weights](#downloading-model-weights)
- [Running the Detectors](#running-the-detectors)
- [Running the Evaluation](#running-the-evaluation)
- [Benchmark](#benchmark)
- [Evaluation Results Summary](#evaluation-results-summary)
  - [5.1 Mounted Camera Angle](#51-mounted-camera-angle)
  - [5.2 Field of Vision](#52-field-of-vision)
  - [5.3 Lighting Conditions](#53-lighting-conditions)
  - [5.4 Detection Distance](#54-detection-distance)
  - [5.5 Object Orientation](#55-object-orientation)
- [Key Findings](#key-findings)
- [System Requirements](#system-requirements)
- [References](#references)

---

## Project Overview

This project develops and evaluates a **real-time object detection system** using computer vision and deep learning, designed to support autonomous robot arm control (e.g., ROSWITHA mobile robot).

The system detects objects in live video, outputs bounding boxes, confidence scores, and **center coordinates (cx, cy)** which can be forwarded directly to a Robot Arm Controller. Three pre-trained models — all based on the **MS-COCO dataset** (80 classes) — are implemented and compared:

| Model | Backbone | Speed | Accuracy |
|---|---|---|---|
| YOLOv3 | DarkNet-53 (106 layers) | Medium | High |
| Tiny-YOLOv3 | DarkNet-19 (13 conv layers) | Fast | Low–Medium |
| SSD-MobileNetV3 | MobileNetV3-Large | Fast | Medium |

---

## Models Implemented

### 1. YOLOv3 + DarkNet53
- **Config:** `yolov3.cfg`
- **Weights:** `yolov3-320.weights` / `yolov3-416.weights` / `yolov3-608.weights`
- **Detection scales:** 3 (13×13, 26×26, 52×52)
- **Confidence threshold:** 0.5 · **NMS threshold:** 0.3
- **COCO detection kernel:** 1×1×255 (B=3 boxes, C=80 classes)

### 2. Tiny-YOLOv3 + DarkNet19
- **Config:** `yolov3-tiny.cfg`
- **Weights:** `yolov3-tiny.weights`
- **Architecture:** 13 convolutional layers + 8 max-pooling layers
- **Detection scales:** 2 (13×13, 26×26)
- Designed for embedded / constrained hardware (e.g., Raspberry Pi)

### 3. SSD + MobileNetV3-Large
- **Config:** `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`
- **Weights:** `frozen_inference_graph.pb`
- **Input size:** 320×320
- **Scale:** 1/127.5 · **Mean:** (127.5, 127.5, 127.5)
- Depth-wise separable convolutions for lightweight inference

---

## Repository Structure

```
ODT-1_ObjectDetection/
│
├── models/
│   ├── yolov3_detector.py          # YOLOv3 real-time detector (3 weight sizes)
│   ├── tiny_yolov3_detector.py     # Tiny-YOLOv3 real-time detector
│   └── ssd_mobilenetv3_detector.py # SSD-MobileNetV3 real-time detector
│
├── utils/
│   ├── evaluate.py                 # Reproduces all results from the final report
│   └── benchmark.py                # FPS comparison across all models
│
├── results/                        # Auto-generated CSV result files
│   ├── angle_results.csv
│   ├── fov_results.csv
│   ├── light_results.csv
│   ├── distance_results.csv
│   └── orientation_results.csv
│
├── weights/                        # Place downloaded weights here (see below)
│
├── download_weights.py             # Helper script to download YOLOv3 weights
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.8+
- OpenCV 4.5+
- NumPy 1.21+
- Webcam or video file

```
opencv-python>=4.5.0
numpy>=1.21.0
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ODT-1_ObjectDetection.git
cd ODT-1_ObjectDetection

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Downloading Model Weights

### Automatic (YOLOv3 + Tiny-YOLOv3 + COCO labels)

```bash
python download_weights.py
```

This downloads:
- `yolov3.cfg` + `yolov3-320/416/608.weights`
- `yolov3-tiny.cfg` + `yolov3-tiny.weights`
- `coco.names`

### Manual — SSD-MobileNetV3

Download from TensorFlow Model Zoo:

```bash
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz
tar -xzf ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz
cp ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb weights/
```

Download the pbtxt config from:  
https://gist.github.com/dkurt/54a8e8b51beb3bd3f770b79e56927bd7  
→ Save as `weights/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`

### Final weights/ folder contents

```
weights/
├── coco.names
├── yolov3.cfg
├── yolov3-320.weights
├── yolov3-416.weights
├── yolov3-608.weights
├── yolov3-tiny.cfg
├── yolov3-tiny.weights
├── frozen_inference_graph.pb
└── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
```

---

## Running the Detectors

All detectors open a webcam window and print detected objects with their **center coordinates** to the console — ready for forwarding to a robot arm controller.

### YOLOv3

```bash
# Default (416×416 input, webcam 0)
python models/yolov3_detector.py

# With custom input size and source
python models/yolov3_detector.py --size 320 --source 0
python models/yolov3_detector.py --size 608 --source video.mp4
```

`--size` options: `320`, `416`, `608`

### Tiny-YOLOv3

```bash
python models/tiny_yolov3_detector.py
python models/tiny_yolov3_detector.py --source 0
```

### SSD-MobileNetV3

```bash
python models/ssd_mobilenetv3_detector.py
python models/ssd_mobilenetv3_detector.py --source 0
```

### Console output format (all models)

```
  [chair]  conf=96.0%  cx=338  cy=316
  [bottle] conf=72.0%  cx=512  cy=240
```

Press **`q`** to quit any detector window.

---

## Running the Evaluation

Reproduces **all quantitative results** from the final report (Sections 5.1–5.5):

```bash
python utils/evaluate.py
```

This prints all result tables to the console and exports **5 CSV files** to `results/`:

| File | Contents |
|---|---|
| `angle_results.csv` | Confidence + cx/cy at 90°, 65°, 45° |
| `fov_results.csv` | Field of vision sweep (120°→60°) |
| `light_results.csv` | All 7 lighting conditions |
| `distance_results.csv` | Detection at 100–700 cm |
| `orientation_results.csv` | Original vs rotated object |

---

## Benchmark

Compare FPS and inference time across all available models on your machine:

```bash
python utils/benchmark.py --source 0 --frames 100
```

---

## Evaluation Results Summary

> All tests conducted with a webcam, OpenCV DNN backend (CPU), and MS-COCO pre-trained weights.  
> Confidence threshold: 0.5 · NMS threshold: 0.3

---

### 5.1 Mounted Camera Angle

**Setup:** Camera at 40 cm height, object (potted plant) at 50 cm, white paper ground cover.

| Angle | SSD conf | SSD cx/cy | YOLOv3 conf | YOLOv3 cx/cy | Tiny conf | Tiny cx/cy |
|---|---|---|---|---|---|---|
| 90° | 69% | 309 / 408 | **98%** | 338 / 316 | N/A | — |
| 65° | 68% | 310 / 308 | **98%** | 340 / 176 | 68% | 311 / 359 |
| 45° | 63% | 815 / 127 | **86%** | 342 / 81 | 52% | 336 / 83 |

**Finding:** Camera angle significantly impacts detection. YOLOv3 shows highest accuracy. SSD is most stable. Tiny-YOLOv3 fails at 90°.

---

### 5.2 Field of Vision

**Setup:** Object (bottle/vase) swept horizontally from 120° to 60°. Tested at 50 cm and 100 cm distances.

| Model | 120° | 110° | 100° | 90° | 80° | 70° | 60° |
|---|---|---|---|---|---|---|---|
| SSD-50cm | 68% | 64% | 53% | 54% | 53% | 55% | 58% |
| SSD-100cm | N/A | 61% | 61% | 55% | 64% | 60% | N/A |
| **YOLO-50cm** | **99%** | **99%** | **99%** | **99%** | **99%** | **99%** | **98%** |
| YOLO-100cm | N/A | 98% | 98% | 79% | 95% | 92% | 91% |
| Tiny-50cm | N/A | 60% | 60% | N/A | 60% | N/A | N/A |
| Tiny-100cm | 61% | 82% | 56% | 54% | N/A | N/A | N/A |

**Finding:** YOLOv3 maintains near-perfect detection across the full field of view. SSD is consistent but lower. Tiny struggles at edge angles.

---

### 5.3 Lighting Conditions

**Setup:** Object (potted plant) tested under colored, dimmed, neutral, and backlit conditions.

| Condition | SSD | YOLOv3 | Tiny |
|---|---|---|---|
| Green obj / Green light | 67% | 89% | N/A |
| Green obj / Neutral light | 65% | **99%** | 68% |
| Red obj / Red light | 52% | N/A | N/A |
| Red obj / Neutral light | 51% | **98%** | 68% |
| Dimmed light | 50% | 76% | N/A |
| Neutral light | 50% | **90%** | 68% |
| Against light source | **69%** | 88% | 51% |

**Finding:** Matching object and light color reduces all models' confidence. SSD uniquely improves against a direct light source. YOLOv3 is most resilient overall.

---

### 5.4 Detection Distance

**Setup:** Chair placed at 50 cm intervals along a corridor (100–700 cm).

| Distance | SSD MobileNetV3 | YOLOv3 | YOLO-Tiny |
|---|---|---|---|
| 700 cm | 50% | 66% | N/A |
| 650 cm | 50% | 77% | N/A |
| 600 cm | N/A | 70% | N/A |
| 550 cm | N/A | 76% | N/A |
| 500 cm | 54% | 88% | N/A |
| 450 cm | 58% | 72% | N/A |
| 400 cm | 68% | **95%** | N/A |
| 350 cm | 67% | **96%** | N/A |
| 300 cm | 50% | **91%** | N/A |
| 250 cm | 65% | **96%** | N/A |
| 200 cm | 65% | **99%** | 53% |
| 150 cm | 72% | **99%** | 58% |
| 100 cm | 68% | **99%** | 55% |

**Finding:** YOLOv3 detects reliably at all tested distances (100–700 cm). SSD effective up to 700 cm but at minimal confidence. Tiny-YOLOv3 limited to ≤200 cm.

---

### 5.5 Object Orientation

**Setup:** Same corridor setup as §5.4; chair tested in original and rotated orientations.

| Distance | SSD orig | SSD rot | YOLO orig | YOLO rot | Tiny orig | Tiny rot |
|---|---|---|---|---|---|---|
| 700 cm | 50% | 51% | 66% | **90%** | N/A | N/A |
| 450 cm | 58% | 54% | 72% | **94%** | N/A | 54% |
| 400 cm | 68% | 71% | 95% | **98%** | N/A | N/A |
| 200 cm | 65% | 69% | 99% | **99%** | 53% | 54% |
| 150 cm | 72% | 69% | 99% | **99%** | 58% | 64% |
| 100 cm | 68% | 56% | 99% | **99%** | 55% | **85%** |

**Finding:** SSD is mostly unaffected by orientation changes. YOLOv3 often improves on rotated objects. Tiny-YOLOv3 achieves its highest single confidence (85%) on a rotated object at 100 cm.

---

## Key Findings

| Criterion | Best Model | Notes |
|---|---|---|
| Accuracy | **YOLOv3** | Up to 99%; consistent across all scenarios |
| Speed (CPU) | **SSD-MobileNetV3** | Fastest FPS; recommended for real-time edge deployment |
| Distance range | **YOLOv3** | Reliable detection up to 700 cm |
| Angle robustness | **SSD-MobileNetV3** | Least drop in confidence as angle decreases |
| Lighting robustness | **YOLOv3** | Maintains detection in dimmed and colored light |
| Embedded systems | **Tiny-YOLOv3** | Designed for Raspberry Pi / constrained hardware |
| **Robot arm control** | ✅ **SSD-MobileNetV3** | Best balance of speed and stability for real-time arm guidance |

> **Recommendation:** Use **SSD-MobileNetV3** for the robot arm controller integration (real-time speed, stable feature mapping). Use **YOLOv3-416** when maximum accuracy is required.

---

## System Requirements

### Hardware
- Webcam (USB or built-in) connected to the laptop/desktop running the detection
- The cx/cy output is transmitted directly to the Robot Arm Controller

### Software
- Python 3.8+
- OpenCV 4.5+ (with DNN module)
- PyCharm IDE or Jupyter Notebook

---

## References

1. J. Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection," IEEE CVPR 2016.
2. J. Redmon, A. Farhadi, "YOLOv3: An Incremental Improvement," arXiv:1804.02767, 2018.
3. W. Liu et al., "SSD: Single Shot MultiBox Detector," ECCV 2016.
4. A. G. Howard et al., "MobileNets: Efficient CNNs for Mobile Vision Applications," arXiv:1704.04861, 2017.
5. S. Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," NIPS 2015.
6. A. Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of Object Detection," arXiv:2004.10934, 2020.
7. Z. Zhao et al., "Object Detection With Deep Learning: A Review," IEEE TNNLS, vol. 30, no. 11, 2019.

---

*Frankfurt University of Applied Sciences — Autonomous Intelligent Systems WS 25/26*
