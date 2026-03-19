# Object Detection and Classification with Computer Vision

> Real-time object detection system for autonomous robot arm control — comparative evaluation of YOLOv3, Tiny-YOLOv3, and SSD-MobileNetV3 using OpenCV and Python.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green?style=flat-square&logo=opencv)
![PyTorch](https://img.shields.io/badge/Framework-OpenCV%20DNN-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)


---

## Overview

This project develops and evaluates a real-time object detection system using computer vision and deep learning, designed to support autonomous robot arm control (e.g., the ROSWITHA mobile robot platform).

The system processes live video streams, outputting **bounding boxes**, **confidence scores**, and **center coordinates (cx, cy)** that can be forwarded directly to a Robot Arm Controller. Three pre-trained models — all based on the **MS-COCO dataset (80 classes)** — are implemented and rigorously compared across five experimental conditions.

---

## Models at a Glance

| Model | Backbone | Speed | Accuracy | Best For |
|---|---|---|---|---|
| YOLOv3 | DarkNet-53 (106 layers) | Medium | High | Maximum accuracy |
| Tiny-YOLOv3 | DarkNet-19 (13 conv layers) | Fast | Low–Medium | Embedded / Raspberry Pi |
| SSD-MobileNetV3 | MobileNetV3-Large | Fast | Medium | Real-time robot arm control |

**Recommendation: SSD-MobileNetV3** for robot arm integration — best balance of speed and detection stability. Use **YOLOv3-416** when maximum accuracy is the priority.

---

## Table of Contents

1. [Repository Structure](#repository-structure)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Downloading Model Weights](#downloading-model-weights)
5. [Running the Detectors](#running-the-detectors)
6. [Running the Evaluation](#running-the-evaluation)
7. [Benchmark](#benchmark)
8. [Evaluation Results](#evaluation-results)
9. [Key Findings](#key-findings)
10. [System Requirements](#system-requirements)
11. [Acknowledgment](#acknowledgment)
12. [References](#references)

---

## Repository Structure

```
ODT-1_ObjectDetection/
│
├── Detectors/
│   ├── yolov3_detector.py           # YOLOv3 real-time detector (3 weight sizes: 320, 416, 608)
│   ├── tiny_yolov3_detector.py      # Tiny-YOLOv3 real-time detector
│   └── ssd_mobilenetv3_detector.py  # SSD-MobileNetV3 real-time detector
│
├── Evaluate/
│   ├── evaluate.py                  # Reproduces all results from the final report (Sections 5.1–5.5)
│   └── benchmark.py                 # FPS comparison across all models
│
├── results/                         # Auto-generated CSV result files
│   ├── angle_results.csv            # Camera angle experiment
│   ├── fov_results.csv              # Field of vision experiment
│   ├── light_results.csv            # Lighting conditions experiment
│   ├── distance_results.csv         # Detection distance experiment
│   └── orientation_results.csv      # Object orientation experiment
│
├── weights/                         # Place all downloaded model weights here
├── download_weights.py              # Helper script to download YOLOv3 weights automatically
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.8+
- OpenCV 4.5+ (with DNN module)
- NumPy 1.21+
- Webcam or video file input

```
opencv-python>=4.5.0
numpy>=1.21.0
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Abwahab55/ODT-1_ObjectDetection.git
cd ODT-1_ObjectDetection

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Downloading Model Weights

### Automatic — YOLOv3, Tiny-YOLOv3 & COCO labels

```bash
python download_weights.py
```

This downloads and places the following into `weights/` automatically:

- `yolov3.cfg` + `yolov3-320.weights`, `yolov3-416.weights`, `yolov3-608.weights`
- `yolov3-tiny.cfg` + `yolov3-tiny.weights`
- `coco.names`

### Manual — SSD-MobileNetV3

```bash
# Download and extract from TensorFlow Model Zoo
wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz
tar -xzf ssd_mobilenet_v3_large_coco_2020_01_14.tar.gz
cp ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb weights/
```

Download the `.pbtxt` config from [this gist](https://gist.github.com/dkurt/54a8e8b51beb3bd3f770b79e56927bd7) and save as:
`weights/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`

### Expected `weights/` folder after setup

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

All detectors open a live webcam window. Add `--verbose` to also print detected objects with their center coordinates to the console — ready for forwarding to a Robot Arm Controller.

Press **`q`** to quit any detector window.

### YOLOv3

```bash
# Default (416×416 input, webcam 0)
python Detectors/yolov3_detector.py

# Custom input size and source
python Detectors/yolov3_detector.py --size 320 --source 0
python Detectors/yolov3_detector.py --size 608 --source video.mp4

# Enable console output of detections
python Detectors/yolov3_detector.py --size 416 --source 0 --verbose
```

`--size` options: `320`, `416`, `608`

### Tiny-YOLOv3

```bash
python Detectors/tiny_yolov3_detector.py
python Detectors/tiny_yolov3_detector.py --source 0 --verbose
```

### SSD-MobileNetV3

```bash
python Detectors/ssd_mobilenetv3_detector.py
python Detectors/ssd_mobilenetv3_detector.py --source 0 --verbose
```

### Console output format (all models)

```
[chair]  conf=96.0%  cx=338  cy=316
[bottle] conf=72.0%  cx=512  cy=240
```

---

## Running the Evaluation

Reproduces all quantitative results from the final report (Sections 5.1–5.5):

```bash
python Evaluate/evaluate.py
```

Results are printed to the console and exported as CSV files to `results/`:

| File | Contents |
|---|---|
| `angle_results.csv` | Confidence + cx/cy at 90°, 65°, 45° |
| `fov_results.csv` | Field of vision sweep (120°→60°) |
| `light_results.csv` | All 7 lighting conditions |
| `distance_results.csv` | Detection at 100–700 cm |
| `orientation_results.csv` | Original vs. rotated object |

---

## Benchmark

Compare FPS and inference time across all models on your machine:

```bash
python Evaluate/benchmark.py --source 0 --frames 100
```

---

## Evaluation Results

> All tests conducted with a USB webcam, OpenCV DNN backend (CPU), and MS-COCO pre-trained weights.
> Confidence threshold: **0.5** · NMS threshold: **0.3**

### 5.1 Mounted Camera Angle

**Setup:** Camera at 40 cm height, potted plant at 50 cm, white paper ground cover. Camera rotated physically at 90°, 65°, and 45°.

| Angle | SSD conf | SSD cx/cy | YOLOv3 conf | YOLOv3 cx/cy | Tiny conf | Tiny cx/cy |
|---|---|---|---|---|---|---|
| 90° | 69% | 309 / 408 | 98% | 338 / 316 | N/A | — |
| 65° | 68% | 310 / 308 | 98% | 340 / 176 | 68% | 311 / 359 |
| 45° | 63% | 815 / 127 | 86% | 342 / 81 | 52% | 336 / 83 |

**Finding:** Camera angle significantly impacts detection performance. YOLOv3 achieves the highest accuracy. SSD-MobileNetV3 is the most stable across angles. Tiny-YOLOv3 fails to detect at 90°.

---

### 5.2 Field of Vision

**Setup:** Object (bottle/vase) swept horizontally from 120° to 60°, tested at 50 cm and 100 cm distances.

| Model | 120° | 110° | 100° | 90° | 80° | 70° | 60° |
|---|---|---|---|---|---|---|---|
| SSD @ 50 cm | 68% | 64% | 53% | 54% | 53% | 55% | 58% |
| SSD @ 100 cm | N/A | 61% | 61% | 55% | 64% | 60% | N/A |
| YOLOv3 @ 50 cm | 99% | 99% | 99% | 99% | 99% | 99% | 98% |
| YOLOv3 @ 100 cm | N/A | 98% | 98% | 79% | 95% | 92% | 91% |
| Tiny @ 50 cm | N/A | 60% | 60% | N/A | 60% | N/A | N/A |
| Tiny @ 100 cm | 61% | 82% | 56% | 54% | N/A | N/A | N/A |

**Finding:** YOLOv3 maintains near-perfect detection across the full field of view. SSD is consistent but lower confidence. Tiny-YOLOv3 struggles at edge angles.

---

### 5.3 Lighting Conditions

**Setup:** Potted plant tested under colored illumination, dimmed conditions, neutral light, and against a direct light source.

| Condition | SSD | YOLOv3 | Tiny |
|---|---|---|---|
| Green object / Green light | 67% | 89% | N/A |
| Green object / Neutral light | 65% | 99% | 68% |
| Red object / Red light | 52% | N/A | N/A |
| Red object / Neutral light | 51% | 98% | 68% |
| Dimmed light | 50% | 76% | N/A |
| Neutral light | 50% | 90% | 68% |
| Against light source | 69% | 88% | 51% |

**Finding:** Matching object and light color reduces confidence across all models. SSD-MobileNetV3 uniquely improves when detecting against a direct light source. YOLOv3 is the most resilient overall.

---

### 5.4 Detection Distance

**Setup:** Chair placed at 50 cm intervals along a corridor, from 100 cm to 700 cm.

| Distance | SSD-MobileNetV3 | YOLOv3 | Tiny-YOLOv3 |
|---|---|---|---|
| 700 cm | 50% | 66% | N/A |
| 650 cm | 50% | 77% | N/A |
| 600 cm | N/A | 70% | N/A |
| 550 cm | N/A | 76% | N/A |
| 500 cm | 54% | 88% | N/A |
| 450 cm | 58% | 72% | N/A |
| 400 cm | 68% | 95% | N/A |
| 350 cm | 67% | 96% | N/A |
| 300 cm | 50% | 91% | N/A |
| 250 cm | 65% | 96% | N/A |
| 200 cm | 65% | 99% | 53% |
| 150 cm | 72% | 99% | 58% |
| 100 cm | 68% | 99% | 55% |

**Finding:** YOLOv3 detects reliably at all tested distances (100–700 cm). SSD effective up to 700 cm at minimal confidence. Tiny-YOLOv3 limited to ≤200 cm.

---

### 5.5 Object Orientation

**Setup:** Same corridor as §5.4; chair tested in original and rotated orientations at each distance.

| Distance | SSD orig | SSD rot | YOLO orig | YOLO rot | Tiny orig | Tiny rot |
|---|---|---|---|---|---|---|
| 700 cm | 50% | 51% | 66% | 90% | N/A | N/A |
| 450 cm | 58% | 54% | 72% | 94% | N/A | 54% |
| 400 cm | 68% | 71% | 95% | 98% | N/A | N/A |
| 200 cm | 65% | 69% | 99% | 99% | 53% | 54% |
| 150 cm | 72% | 69% | 99% | 99% | 58% | 64% |
| 100 cm | 68% | 56% | 99% | 99% | 55% | 85% |

**Finding:** SSD is largely unaffected by orientation changes. YOLOv3 often improves on rotated objects. Tiny-YOLOv3 achieves its highest single confidence (85%) on a rotated object at 100 cm.

---

## Key Findings

| Criterion | Best Model | Notes |
|---|---|---|
| Accuracy | YOLOv3 | Up to 99%; consistent across all scenarios |
| Speed (CPU) | SSD-MobileNetV3 | Fastest FPS; ideal for real-time edge deployment |
| Detection range | YOLOv3 | Reliable detection up to 700 cm |
| Angle robustness | SSD-MobileNetV3 | Least confidence drop as angle decreases |
| Lighting robustness | YOLOv3 | Maintains detection under dimmed and colored light |
| Embedded systems | Tiny-YOLOv3 | Designed for Raspberry Pi / constrained hardware |
| **Robot arm control** | **SSD-MobileNetV3** | Best balance of speed and stability for real-time arm guidance |

---

## System Requirements

**Hardware**
- Webcam (USB or built-in) connected to the host laptop/desktop
- The `cx/cy` output is transmitted directly to the Robot Arm Controller

**Software**
- Python 3.8+
- OpenCV 4.5+ with DNN module
- PyCharm IDE or Jupyter Notebook (alternative)

---

## Acknowledgment

Abdul Wahab and Muhammad Faraz Abbasi express their sincere gratitude to Prof. Peter Nauth for his expert guidance, constructive feedback, and unwavering encouragement throughout the Autonomous Intelligent Systems course (WS 25/26) at Frankfurt University of Applied Sciences. His deep expertise in intelligent systems and computer vision provided the intellectual foundation upon which this work was built. The authors further acknowledge the open-source contributions of the OpenCV and TensorFlow communities, and the pioneering researchers whose foundational work on object detection architectures — including YOLO, SSD, and MobileNet — served as the theoretical and practical cornerstones of this implementation.

---

## References

1.	D. G. Lowe, "Distinctive image features from scale-invariant keypoints," Int. J. of Comput. Vision, vol. 60, no. 2, pp. 91–110, 2004.
2.	Z. Zhao, P. Zheng, S. Xu and X. Wu, "Object detection with deep learning: A review," IEEE Transactions on Neural Networks and Learning Systems, vol. 30, no. 11, pp. 3212–3232, Nov. 2019, doi: 10.1109/TNNLS.2018.2876865.
3.	N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), 2005.
4.	R. Lienhart and J. Maydt, "An extended set of Haar-like features for rapid object detection," in Proc. IEEE Int. Conf. Image Processing (ICIP), 2002.
5.	J. Loy, Neural Network Projects with Python. Birmingham, UK: Packt Publishing Limited, Feb. 2019, ISBN 9781789138900.
6.	D. Cao, Z. Chen and L. Gao, "An improved object detection algorithm based on multi-scaled and deformable convolutional neural networks," Human-centric Computing and Information Sciences, vol. 10, no. 14, Apr. 2020.
7.	H. Schneiderman and T. Kanade, "A statistical model for 3D object detection applied to faces and cars," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), pp. 746–751, 2000.
8.	F. Fleuret and D. Geman, "Stationary features and cat detection," Journal of Machine Learning Research (JMLR), vol. 9, pp. 2549–2578, 2008.
9.	S. Romdhani, P. Torr, B. Schölkopf and A. Blake, "Computationally efficient face detection," in Proc. Eighth IEEE Int. Conf. Computer Vision (ICCV), vol. 2, pp. 695–700, 2001, doi: 10.1109/ICCV.2001.937694.
10.	G. Alpaydin, "An adaptive deep neural network for detection and recognition of objects with long range auto surveillance," in Proc. IEEE Int. Conf. Semantic Computing (ICSC), pp. 316–317, 2018.
11.	R. Girshick, J. Donahue, T. Darrell and J. Malik, "Rich feature hierarchies for accurate object detection and semantic segmentation," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), Columbus, OH, pp. 580–587, Jun. 2014.
12.	S. Ren, K. He, R. Girshick and J. Sun, "Faster R-CNN: Towards real-time object detection with region proposal networks," in Advances in Neural Information Processing Systems (NIPS), pp. 91–99, 2015.
13.	P. Viola and M. Jones, "Robust real-time object detection," Int. Journal of Computer Vision (IJCV), vol. 57, Jul. 2001.
14.	Z. Cai, Q. Fan, R. S. Feris and N. Vasconcelos, "A unified multi-scale deep convolutional neural network for fast object detection," in Proc. European Conf. Computer Vision (ECCV). Springer, pp. 354–370, 2016.
15.	J. Redmon, S. Divvala, R. Girshick and A. Farhadi, "You only look once: Unified, real-time object detection," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), pp. 779–788, Jun. 2016.
16.	J. Redmon and A. Farhadi, "YOLOv3: An incremental improvement," arXiv preprint arXiv:1804.02767, 2018.
17.	J. Redmon and A. Farhadi, "YOLO9000: Better, faster, stronger," in Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR), Honolulu, HI, pp. 7263–7271, 2017.
18.	A. Bochkovskiy, C. Y. Wang and H. Y. M. Liao, "YOLOv4: Optimal speed and accuracy of object detection," arXiv preprint arXiv:2004.10934, 2020.
19.	W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C. Y. Fu and A. C. Berg, "SSD: Single shot multibox detector," in Proc. 14th European Conf. Computer Vision (ECCV), Amsterdam, The Netherlands, pp. 21–37, Oct. 2016.
20.	K. M. He, X. Y. Zhang, S. Q. Ren and J. Sun, "Spatial pyramid pooling in deep convolutional networks for visual recognition," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 37, pp. 1904–1916, 2015.
21.	J. Deng, X. Xuan, W. Wang, Z. Li, H. Yao and Z. Wang, "A review of research on object detection based on deep learning," Journal of Physics: Conference Series, vol. 1684, 012028, 2020.
22.	A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W. Wang, T. Weyand, M. Andreetto and H. Adam, "MobileNets: Efficient convolutional neural networks for mobile vision applications," arXiv preprint arXiv:1704.04861, 2017.
23.	S. K. Chadalawada, "Real time object detection and recognition using deep learning methods," M.S. thesis, Dept. Computer Science and Engineering, Blekinge Institute of Technology, Karlskrona, Sweden, Feb. 2020.
24.	M. R. Sri Sai, S. Rella and S. Veeravalli, "Object detection and identification," Project Report, Nov. 2019. [Online]. Available: https://www.researchgate.net/publication/337464355
25.	P. F. Felzenszwalb, R. B. Girshick, D. McAllester and D. Ramanan, "Object detection with discriminatively trained part-based models," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 32, no. 9, p. 1627, 2010.
26.	K. K. Sung and T. Poggio, "Example-based learning for view-based human face detection," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 20, no. 1, pp. 39–51, 2002.
27.	C. Wojek, P. Dollár, B. Schiele and P. Perona, "Pedestrian detection: An evaluation of the state of the art," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, no. 4, p. 743, 2012.
28.	H. Kobatake and Y. Yoshinaga, "Detection of spicules on mammogram based on skeleton analysis," IEEE Transactions on Medical Imaging, vol. 15, no. 3, pp. 235–245, 1996.
29.	J. Hui, "SSD object detection: Single shot multibox detector for real-time processing," Medium.com, Mar. 2018. [Online]. Available: https://jonathan-hui.medium.com/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06
30.	V. Jain, "Everything you need to know about MobileNetV3," Towards Data Science, Nov. 2019. [Online]. Available: https://towardsdatascience.com.

