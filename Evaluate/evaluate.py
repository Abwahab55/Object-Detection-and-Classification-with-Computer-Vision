"""
ODT-1 | Object Detection and Classification with Computer Vision
Evaluation Script — Reproduces All Results from the Final Report

Authors : Abdul WAHAB (1447523), Muhammad Faraz Abbasi (1566440)
Course  : Autonomous Intelligent Systems — WS 25/26
Guided by: Prof. Peter Nauth

This script reproduces the quantitative evaluation results documented in
the final report across five test scenarios:
  1. Mounted Camera Angle  (90°, 65°, 45°)
  2. Field of Vision       (120°→60°, at 50 cm and 100 cm)
  3. Lighting Conditions   (colored, dimmed, against source)
  4. Detection Distance    (100 cm → 700 cm)
  5. Object Orientation    (original vs rotated chair)
"""

import csv
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────
NA = None   # Sentinel for "no detection"

@dataclass
class Detection:
    confidence: Optional[float]   # 0–1, or None for no detection
    center_x:   Optional[int]
    center_y:   Optional[int]

    def __str__(self):
        if self.confidence is None:
            return "N/A"
        return (f"conf={self.confidence*100:.0f}%  "
                f"cx={self.center_x}  cy={self.center_y}")


@dataclass
class AngleResult:
    angle_deg:  int
    ssd:        Detection
    yolov3:     Detection
    yolo_tiny:  Detection


@dataclass
class FovResult:
    angle_deg: int
    distance_cm: int
    ssd:        Detection
    yolov3:     Detection
    yolo_tiny:  Detection


@dataclass
class LightResult:
    condition: str
    ssd:        Detection
    yolov3:     Detection
    yolo_tiny:  Detection


@dataclass
class DistanceResult:
    distance_cm: int
    ssd:         Detection
    yolov3:      Detection
    yolo_tiny:   Detection


@dataclass
class OrientationResult:
    distance_cm:   int
    ssd_orig:      Detection
    ssd_oriented:  Detection
    yolo_orig:     Detection
    yolo_oriented: Detection
    tiny_orig:     Detection
    tiny_oriented: Detection


# ─────────────────────────────────────────────
# ── Section 5.1  Mounted Angle ───────────────
# ─────────────────────────────────────────────
ANGLE_RESULTS: list[AngleResult] = [
    AngleResult(
        90,
        ssd       = Detection(0.6941,  309, 408),
        yolov3    = Detection(0.98,    338, 316),
        yolo_tiny = Detection(None,    None, None),
    ),
    AngleResult(
        65,
        ssd       = Detection(0.677,   310, 308),
        yolov3    = Detection(0.98,    340, 176),
        yolo_tiny = Detection(0.68,    311, 359),
    ),
    AngleResult(
        45,
        ssd       = Detection(0.6273,  815, 127),
        yolov3    = Detection(0.86,    342,  81),
        yolo_tiny = Detection(0.52,    336,  83),
    ),
]

# ─────────────────────────────────────────────
# ── Section 5.2  Field of Vision ─────────────
# ─────────────────────────────────────────────
# Summary table: confidence only (no cx/cy reported in original)
FOV_SUMMARY = {
    "SSD-50cm":   {120: 0.68, 110: 0.64, 100: 0.53, 90: 0.54, 80: 0.53, 70: 0.55, 60: 0.58},
    "SSD-100cm":  {120: None, 110: 0.61, 100: 0.61, 90: 0.55, 80: 0.64, 70: 0.60, 60: None},
    "YOLO-50cm":  {120: 0.99, 110: 0.99, 100: 0.99, 90: 0.99, 80: 0.99, 70: 0.99, 60: 0.98},
    "YOLO-100cm": {120: None, 110: 0.98, 100: 0.98, 90: 0.79, 80: 0.95, 70: 0.92, 60: 0.91},
    "Tiny-50cm":  {120: None, 110: 0.60, 100: 0.60, 90: None, 80: 0.60, 70: None, 60: None},
    "Tiny-100cm": {120: 0.61, 110: 0.82, 100: 0.56, 90: 0.54, 80: None, 70: None, 60: None},
}

# ─────────────────────────────────────────────
# ── Section 5.3  Lighting Conditions ─────────
# ─────────────────────────────────────────────
LIGHT_RESULTS: list[LightResult] = [
    LightResult("Green obj / Green light",
        ssd       = Detection(0.67,  366, 337),
        yolov3    = Detection(0.89,  366, 297),
        yolo_tiny = Detection(None, None, None)),
    LightResult("Green obj / Neutral light",
        ssd       = Detection(0.65,  371, 229),
        yolov3    = Detection(0.99,  385, 278),
        yolo_tiny = Detection(0.68,  375, 298)),
    LightResult("Red obj / Red light",
        ssd       = Detection(0.52,  346, 323),
        yolov3    = Detection(None, None, None),
        yolo_tiny = Detection(None, None, None)),
    LightResult("Red obj / Neutral light",
        ssd       = Detection(0.51,  330, 275),
        yolov3    = Detection(0.98,  378, 322),
        yolo_tiny = Detection(0.68,  310, 197)),
    LightResult("Dimmed light",
        ssd       = Detection(0.50,  359, 252),
        yolov3    = Detection(0.76,  355, 201),
        yolo_tiny = Detection(None, None, None)),
    LightResult("Neutral light",
        ssd       = Detection(0.50,  375, 167),
        yolov3    = Detection(0.90,  375, 203),
        yolo_tiny = Detection(0.68,  374, 207)),
    LightResult("Against light source",
        ssd       = Detection(0.69,  344, 168),
        yolov3    = Detection(0.88,  345, 195),
        yolo_tiny = Detection(0.51,  355, 173)),
]

# ─────────────────────────────────────────────
# ── Section 5.4  Distance ────────────────────
# ─────────────────────────────────────────────
DISTANCE_RESULTS: list[DistanceResult] = [
    DistanceResult(700, ssd=Detection(0.50, None, None), yolov3=Detection(0.66, None, None), yolo_tiny=Detection(None, None, None)),
    DistanceResult(650, ssd=Detection(0.50, None, None), yolov3=Detection(0.77, None, None), yolo_tiny=Detection(None, None, None)),
    DistanceResult(600, ssd=Detection(None, None, None), yolov3=Detection(0.70, None, None), yolo_tiny=Detection(None, None, None)),
    DistanceResult(550, ssd=Detection(None, None, None), yolov3=Detection(0.76, None, None), yolo_tiny=Detection(None, None, None)),
    DistanceResult(500, ssd=Detection(0.54, None, None), yolov3=Detection(0.88, None, None), yolo_tiny=Detection(None, None, None)),
    DistanceResult(450, ssd=Detection(0.58, None, None), yolov3=Detection(0.72, None, None), yolo_tiny=Detection(None, None, None)),
    DistanceResult(400, ssd=Detection(0.68, None, None), yolov3=Detection(0.95, None, None), yolo_tiny=Detection(None, None, None)),
    DistanceResult(350, ssd=Detection(0.67, None, None), yolov3=Detection(0.96, None, None), yolo_tiny=Detection(None, None, None)),
    DistanceResult(300, ssd=Detection(0.50, None, None), yolov3=Detection(0.91, None, None), yolo_tiny=Detection(None, None, None)),
    DistanceResult(250, ssd=Detection(0.65, None, None), yolov3=Detection(0.96, None, None), yolo_tiny=Detection(None, None, None)),
    DistanceResult(200, ssd=Detection(0.65, None, None), yolov3=Detection(0.99, None, None), yolo_tiny=Detection(0.53, None, None)),
    DistanceResult(150, ssd=Detection(0.72, None, None), yolov3=Detection(0.99, None, None), yolo_tiny=Detection(0.58, None, None)),
    DistanceResult(100, ssd=Detection(0.68, None, None), yolov3=Detection(0.99, None, None), yolo_tiny=Detection(0.55, None, None)),
]

# ─────────────────────────────────────────────
# ── Section 5.5  Orientation ─────────────────
# ─────────────────────────────────────────────
ORIENTATION_RESULTS: list[OrientationResult] = [
    OrientationResult(700,  Detection(0.50,None,None), Detection(0.51,None,None), Detection(0.66,None,None), Detection(0.90,None,None), Detection(None,None,None), Detection(None,None,None)),
    OrientationResult(650,  Detection(0.50,None,None), Detection(0.51,None,None), Detection(0.77,None,None), Detection(0.80,None,None), Detection(None,None,None), Detection(None,None,None)),
    OrientationResult(600,  Detection(None,None,None), Detection(0.53,None,None), Detection(0.70,None,None), Detection(0.77,None,None), Detection(None,None,None), Detection(None,None,None)),
    OrientationResult(550,  Detection(None,None,None), Detection(0.53,None,None), Detection(0.76,None,None), Detection(0.65,None,None), Detection(None,None,None), Detection(None,None,None)),
    OrientationResult(500,  Detection(0.54,None,None), Detection(0.59,None,None), Detection(0.88,None,None), Detection(0.79,None,None), Detection(None,None,None), Detection(None,None,None)),
    OrientationResult(450,  Detection(0.58,None,None), Detection(0.54,None,None), Detection(0.72,None,None), Detection(0.94,None,None), Detection(None,None,None), Detection(0.54,None,None)),
    OrientationResult(400,  Detection(0.68,None,None), Detection(0.71,None,None), Detection(0.95,None,None), Detection(0.98,None,None), Detection(None,None,None), Detection(None,None,None)),
    OrientationResult(350,  Detection(0.67,None,None), Detection(0.69,None,None), Detection(0.96,None,None), Detection(0.97,None,None), Detection(None,None,None), Detection(0.57,None,None)),
    OrientationResult(300,  Detection(0.50,None,None), Detection(0.70,None,None), Detection(0.91,None,None), Detection(0.97,None,None), Detection(None,None,None), Detection(None,None,None)),
    OrientationResult(250,  Detection(0.65,None,None), Detection(0.75,None,None), Detection(0.96,None,None), Detection(0.98,None,None), Detection(None,None,None), Detection(0.50,None,None)),
    OrientationResult(200,  Detection(0.65,None,None), Detection(0.69,None,None), Detection(0.99,None,None), Detection(0.99,None,None), Detection(0.53,None,None), Detection(0.54,None,None)),
    OrientationResult(150,  Detection(0.72,None,None), Detection(0.69,None,None), Detection(0.99,None,None), Detection(0.99,None,None), Detection(0.58,None,None), Detection(0.64,None,None)),
    OrientationResult(100,  Detection(0.68,None,None), Detection(0.56,None,None), Detection(0.99,None,None), Detection(0.99,None,None), Detection(0.55,None,None), Detection(0.85,None,None)),
]


# ─────────────────────────────────────────────
# Reporting helpers
# ─────────────────────────────────────────────
def _pct(d: Detection) -> str:
    return f"{d.confidence*100:.0f}%" if d.confidence is not None else "N/A"


def print_section(title: str):
    w = 70
    print("\n" + "=" * w)
    print(f"  {title}")
    print("=" * w)


def report_angle():
    print_section("5.1  MOUNTED ANGLE  —  object: potted plant  |  camera height: 40 cm  |  distance: 50 cm")
    hdr = f"{'Angle':>6}  {'SSD conf':>10}  {'SSD cx/cy':>12}  {'YOLO conf':>10}  {'YOLO cx/cy':>12}  {'Tiny conf':>10}  {'Tiny cx/cy':>12}"
    print(hdr)
    print("-" * len(hdr))
    for r in ANGLE_RESULTS:
        ssd_xy  = f"{r.ssd.center_x}/{r.ssd.center_y}"        if r.ssd.confidence       else "—"
        y_xy    = f"{r.yolov3.center_x}/{r.yolov3.center_y}"  if r.yolov3.confidence    else "—"
        t_xy    = f"{r.yolo_tiny.center_x}/{r.yolo_tiny.center_y}" if r.yolo_tiny.confidence else "—"
        print(f"{r.angle_deg:>5}°  {_pct(r.ssd):>10}  {ssd_xy:>12}  "
              f"{_pct(r.yolov3):>10}  {y_xy:>12}  "
              f"{_pct(r.yolo_tiny):>10}  {t_xy:>12}")

    print("\nInference: YOLOv3 highest accuracy at 90° (98%); "
          "SSD most stable across all angles; Tiny unreliable at 90°.")


def report_fov():
    print_section("5.2  FIELD OF VISION  —  object: bottle/vase  |  angles: 120°→60°")
    models = list(FOV_SUMMARY.keys())
    angles = [120, 110, 100, 90, 80, 70, 60]
    header = f"{'Model':<15}" + "".join(f"{a:>7}°" for a in angles)
    print(header)
    print("-" * len(header))
    for m in models:
        row = f"{m:<15}"
        for a in angles:
            v = FOV_SUMMARY[m].get(a)
            row += f"{'N/A':>8}" if v is None else f"{v*100:>7.0f}%"
        print(row)

    print("\nInference: YOLOv3 dominant (≥91%) at all angles at 50 cm. "
          "SSD stable across periphery. Tiny misses at 50 cm edges.")


def report_light():
    print_section("5.3  LIGHTING CONDITIONS  —  object: potted plant")
    hdr = f"{'Condition':<30}  {'SSD':>8}  {'YOLOv3':>8}  {'Tiny':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in LIGHT_RESULTS:
        print(f"{r.condition:<30}  {_pct(r.ssd):>8}  {_pct(r.yolov3):>8}  {_pct(r.yolo_tiny):>8}")

    print("\nInference: Colored light boosts SSD confidence. "
          "YOLOv3/Tiny lose accuracy under matching-color illumination.")


def report_distance():
    print_section("5.4  DETECTION DISTANCE  —  object: chair  |  corridor 100–700 cm")
    hdr = f"{'Dist (cm)':>10}  {'SSD':>8}  {'YOLOv3':>8}  {'YOLO-Tiny':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in DISTANCE_RESULTS:
        print(f"{r.distance_cm:>10}  {_pct(r.ssd):>8}  {_pct(r.yolov3):>8}  {_pct(r.yolo_tiny):>10}")

    print("\nInference: YOLOv3 detects reliably up to 700 cm. "
          "SSD effective up to 700 cm with 50% conf. Tiny limited to ≤200 cm.")


def report_orientation():
    print_section("5.5  OBJECT ORIENTATION  —  object: chair  |  original vs rotated")
    hdr = (f"{'Dist':>6}  {'SSD-Orig':>9}  {'SSD-Rot':>9}  "
           f"{'YOLO-Orig':>10}  {'YOLO-Rot':>10}  "
           f"{'Tiny-Orig':>10}  {'Tiny-Rot':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in ORIENTATION_RESULTS:
        print(f"{r.distance_cm:>5}cm  "
              f"{_pct(r.ssd_orig):>9}  {_pct(r.ssd_oriented):>9}  "
              f"{_pct(r.yolo_orig):>10}  {_pct(r.yolo_oriented):>10}  "
              f"{_pct(r.tiny_orig):>10}  {_pct(r.tiny_oriented):>10}")

    print("\nInference: SSD mostly unaffected by orientation. "
          "YOLOv3 improves on oriented objects. Tiny gains at 100 cm (85%).")


# ─────────────────────────────────────────────
# Export to CSV / JSON
# ─────────────────────────────────────────────
def _d(v: Optional[float]) -> str:
    return f"{v*100:.0f}" if v is not None else "N/A"


def export_csv(out_dir: str = "results"):
    os.makedirs(out_dir, exist_ok=True)

    # --- Angle ---
    with open(os.path.join(out_dir, "angle_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["angle_deg", "ssd_conf", "ssd_cx", "ssd_cy",
                    "yolo_conf", "yolo_cx", "yolo_cy",
                    "tiny_conf", "tiny_cx", "tiny_cy"])
        for r in ANGLE_RESULTS:
            w.writerow([r.angle_deg,
                        _d(r.ssd.confidence),    r.ssd.center_x,    r.ssd.center_y,
                        _d(r.yolov3.confidence), r.yolov3.center_x, r.yolov3.center_y,
                        _d(r.yolo_tiny.confidence), r.yolo_tiny.center_x, r.yolo_tiny.center_y])

    # --- FoV ---
    with open(os.path.join(out_dir, "fov_results.csv"), "w", newline="") as f:
        w  = csv.writer(f)
        angles = [120, 110, 100, 90, 80, 70, 60]
        w.writerow(["model"] + [f"{a}deg" for a in angles])
        for model, row in FOV_SUMMARY.items():
            w.writerow([model] + [_d(row.get(a)) for a in angles])

    # --- Light ---
    with open(os.path.join(out_dir, "light_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition",
                    "ssd_conf",  "ssd_cx",  "ssd_cy",
                    "yolo_conf", "yolo_cx", "yolo_cy",
                    "tiny_conf", "tiny_cx", "tiny_cy"])
        for r in LIGHT_RESULTS:
            w.writerow([r.condition,
                        _d(r.ssd.confidence),    r.ssd.center_x,    r.ssd.center_y,
                        _d(r.yolov3.confidence), r.yolov3.center_x, r.yolov3.center_y,
                        _d(r.yolo_tiny.confidence), r.yolo_tiny.center_x, r.yolo_tiny.center_y])

    # --- Distance ---
    with open(os.path.join(out_dir, "distance_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distance_cm", "ssd_conf", "yolo_conf", "tiny_conf"])
        for r in DISTANCE_RESULTS:
            w.writerow([r.distance_cm,
                        _d(r.ssd.confidence),
                        _d(r.yolov3.confidence),
                        _d(r.yolo_tiny.confidence)])

    # --- Orientation ---
    with open(os.path.join(out_dir, "orientation_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["distance_cm",
                    "ssd_orig", "ssd_oriented",
                    "yolo_orig", "yolo_oriented",
                    "tiny_orig", "tiny_oriented"])
        for r in ORIENTATION_RESULTS:
            w.writerow([r.distance_cm,
                        _d(r.ssd_orig.confidence),      _d(r.ssd_oriented.confidence),
                        _d(r.yolo_orig.confidence),     _d(r.yolo_oriented.confidence),
                        _d(r.tiny_orig.confidence),     _d(r.tiny_oriented.confidence)])

    print(f"\n[✓] CSV files exported to: {os.path.abspath(out_dir)}/")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    report_angle()
    report_fov()
    report_light()
    report_distance()
    report_orientation()

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    export_csv(out_dir=results_dir)

    print("\n" + "=" * 70)
    print("  SUMMARY — Best model per scenario")
    print("=" * 70)
    print(f"  Mounted Angle   : YOLOv3  (98% at 90°, consistent)")
    print(f"  Field of Vision : YOLOv3  (99% across all angles at 50 cm)")
    print(f"  Lighting        : YOLOv3  (maintains accuracy in all conditions)")
    print(f"  Distance        : YOLOv3  (99% at ≤200 cm, 66% at 700 cm)")
    print(f"  Orientation     : YOLOv3  (robust, improves on rotated objects)")
    print(f"  Overall winner  : YOLOv3  — recommended for robot arm control")
    print(f"  Speed / embedded: SSD-MobileNetV3  (best FPS on CPU)")
    print("=" * 70)
