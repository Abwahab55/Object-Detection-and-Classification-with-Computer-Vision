"""
ODT-1 | Object Detection and Classification with Computer Vision
Test Suite — covers evaluate.py and all three detector modules

Run with:
    python test_all.py
or:
    python -m pytest test_all.py -v
"""

import csv
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# ── path setup ────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "Detectors"))
sys.path.insert(0, os.path.join(_ROOT, "Evaluate"))

# ── imports ───────────────────────────────────────────────────────────────────
from evaluate import (
    ANGLE_RESULTS,
    DISTANCE_RESULTS,
    FOV_SUMMARY,
    LIGHT_RESULTS,
    ORIENTATION_RESULTS,
    AngleResult,
    Detection,
    DistanceResult,
    LightResult,
    OrientationResult,
    _d,
    _pct,
    export_csv,
    print_section,
    report_angle,
    report_distance,
    report_fov,
    report_light,
    report_orientation,
)

import ssd_mobilenetv3_detector as ssd
import tiny_yolov3_detector as tiny
import yolov3_detector as yolo3


# ══════════════════════════════════════════════════════════════════════════════
# evaluate.py
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectionDataclass(unittest.TestCase):
    """Tests for the Detection dataclass."""

    def test_str_with_values(self):
        d = Detection(confidence=0.95, center_x=100, center_y=200)
        text = str(d)
        self.assertIn("95%", text)
        self.assertIn("cx=100", text)
        self.assertIn("cy=200", text)

    def test_str_no_detection(self):
        d = Detection(confidence=None, center_x=None, center_y=None)
        self.assertEqual(str(d), "N/A")

    def test_pct_helper_with_value(self):
        d = Detection(confidence=0.75, center_x=0, center_y=0)
        self.assertEqual(_pct(d), "75%")

    def test_pct_helper_none(self):
        d = Detection(confidence=None, center_x=None, center_y=None)
        self.assertEqual(_pct(d), "N/A")

    def test_d_helper_with_value(self):
        self.assertEqual(_d(0.80), "80")

    def test_d_helper_none(self):
        self.assertEqual(_d(None), "N/A")


class TestResultDatasets(unittest.TestCase):
    """Sanity checks on the embedded result datasets."""

    def test_angle_results_count(self):
        self.assertEqual(len(ANGLE_RESULTS), 3)

    def test_angle_result_angles(self):
        angles = [r.angle_deg for r in ANGLE_RESULTS]
        self.assertIn(90, angles)
        self.assertIn(65, angles)
        self.assertIn(45, angles)

    def test_distance_results_count(self):
        self.assertEqual(len(DISTANCE_RESULTS), 13)

    def test_distance_results_range(self):
        dists = [r.distance_cm for r in DISTANCE_RESULTS]
        self.assertEqual(max(dists), 700)
        self.assertEqual(min(dists), 100)

    def test_light_results_count(self):
        self.assertEqual(len(LIGHT_RESULTS), 7)

    def test_orientation_results_count(self):
        self.assertEqual(len(ORIENTATION_RESULTS), 13)

    def test_fov_summary_models(self):
        expected = {"SSD-50cm", "SSD-100cm", "YOLO-50cm", "YOLO-100cm",
                    "Tiny-50cm", "Tiny-100cm"}
        self.assertEqual(set(FOV_SUMMARY.keys()), expected)

    def test_fov_summary_angles(self):
        angles_expected = {120, 110, 100, 90, 80, 70, 60}
        for model, row in FOV_SUMMARY.items():
            self.assertEqual(set(row.keys()), angles_expected,
                             f"Model {model} has unexpected angle keys")

    def test_confidence_values_in_range(self):
        for r in ANGLE_RESULTS:
            for det in (r.ssd, r.yolov3, r.yolo_tiny):
                if det.confidence is not None:
                    self.assertGreaterEqual(det.confidence, 0.0)
                    self.assertLessEqual(det.confidence, 1.0)

    def test_distance_yolov3_always_detected(self):
        """YOLOv3 should detect the chair at every distance in the dataset."""
        for r in DISTANCE_RESULTS:
            self.assertIsNotNone(
                r.yolov3.confidence,
                f"YOLOv3 expected detection at {r.distance_cm} cm"
            )


class TestReportFunctions(unittest.TestCase):
    """Smoke-tests: report functions must run without raising exceptions."""

    def test_report_angle_runs(self):
        report_angle()

    def test_report_fov_runs(self):
        report_fov()

    def test_report_light_runs(self):
        report_light()

    def test_report_distance_runs(self):
        report_distance()

    def test_report_orientation_runs(self):
        report_orientation()

    def test_print_section_runs(self):
        print_section("Unit test section")


class TestExportCsv(unittest.TestCase):
    """Tests for the export_csv function."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def test_creates_all_csv_files(self):
        export_csv(out_dir=self.tmp_dir)
        expected = {
            "angle_results.csv",
            "fov_results.csv",
            "light_results.csv",
            "distance_results.csv",
            "orientation_results.csv",
        }
        created = set(os.listdir(self.tmp_dir))
        self.assertEqual(expected, created)

    def test_angle_csv_row_count(self):
        export_csv(out_dir=self.tmp_dir)
        path = os.path.join(self.tmp_dir, "angle_results.csv")
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        # header + 3 data rows
        self.assertEqual(len(rows), 4)

    def test_distance_csv_row_count(self):
        export_csv(out_dir=self.tmp_dir)
        path = os.path.join(self.tmp_dir, "distance_results.csv")
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        # header + 13 data rows
        self.assertEqual(len(rows), 14)

    def test_fov_csv_header(self):
        export_csv(out_dir=self.tmp_dir)
        path = os.path.join(self.tmp_dir, "fov_results.csv")
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        self.assertEqual(header[0], "model")
        self.assertIn("120deg", header)
        self.assertIn("60deg", header)

    def test_angle_csv_columns(self):
        export_csv(out_dir=self.tmp_dir)
        path = os.path.join(self.tmp_dir, "angle_results.csv")
        with open(path, newline="") as f:
            header = next(csv.reader(f))
        self.assertIn("angle_deg", header)
        self.assertIn("ssd_conf", header)
        self.assertIn("yolo_conf", header)
        self.assertIn("tiny_conf", header)

    def test_export_idempotent(self):
        """Running export_csv twice should not raise and produce same files."""
        export_csv(out_dir=self.tmp_dir)
        export_csv(out_dir=self.tmp_dir)
        self.assertEqual(len(os.listdir(self.tmp_dir)), 5)


# ══════════════════════════════════════════════════════════════════════════════
# Shared detector helpers (used by YOLOv3, Tiny-YOLOv3, SSD)
# ══════════════════════════════════════════════════════════════════════════════

def _make_names_file(names):
    """Write a temporary COCO-style names file and return its path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".names", delete=False)
    tmp.write("\n".join(names))
    tmp.close()
    return tmp.name


def _blank_frame(h=480, w=640):
    """Return a plain black BGR frame (numpy array)."""
    return np.zeros((h, w, 3), dtype=np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# yolov3_detector.py
# ══════════════════════════════════════════════════════════════════════════════

class TestYolov3Config(unittest.TestCase):
    """Constants defined in yolov3_detector must be sane."""

    def test_confidence_threshold(self):
        self.assertEqual(yolo3.CONFIDENCE_THRESHOLD, 0.5)

    def test_nms_threshold(self):
        self.assertEqual(yolo3.NMS_THRESHOLD, 0.3)

    def test_supported_sizes(self):
        self.assertIn(320, yolo3.YOLOV3_CFGS)
        self.assertIn(416, yolo3.YOLOV3_CFGS)
        self.assertIn(608, yolo3.YOLOV3_CFGS)


class TestYolov3LoadClassNames(unittest.TestCase):
    """load_class_names reads a names file correctly."""

    def test_loads_names(self):
        names = ["person", "bicycle", "car"]
        path = _make_names_file(names)
        try:
            loaded = yolo3.load_class_names(path)
            self.assertEqual(loaded, names)
        finally:
            os.unlink(path)

    def test_strips_whitespace(self):
        path = _make_names_file(["  cat  ", "  dog  "])
        try:
            loaded = yolo3.load_class_names(path)
            self.assertEqual(loaded, ["cat", "dog"])
        finally:
            os.unlink(path)

    def test_80_coco_classes(self):
        names = [f"class_{i}" for i in range(80)]
        path = _make_names_file(names)
        try:
            loaded = yolo3.load_class_names(path)
            self.assertEqual(len(loaded), 80)
        finally:
            os.unlink(path)


class TestYolov3GetOutputLayers(unittest.TestCase):
    """get_output_layers returns the correct layer names."""

    def _make_net(self, layer_names, unconnected):
        net = MagicMock()
        net.getLayerNames.return_value = layer_names
        net.getUnconnectedOutLayers.return_value = np.array(unconnected)
        return net

    def test_flat_indices(self):
        layers = ["conv1", "yolo_82", "yolo_94", "yolo_106"]
        net = self._make_net(layers, [2, 3, 4])
        result = yolo3.get_output_layers(net)
        self.assertEqual(result, ["yolo_82", "yolo_94", "yolo_106"])

    def test_nested_indices(self):
        layers = ["conv1", "yolo_82", "yolo_94"]
        net = self._make_net(layers, [[2], [3]])
        result = yolo3.get_output_layers(net)
        self.assertEqual(result, ["yolo_82", "yolo_94"])


class TestYolov3DrawDetections(unittest.TestCase):
    """draw_detections must draw on the frame without raising."""

    def test_no_detections(self):
        frame = _blank_frame()
        result = yolo3.draw_detections(frame, [], [], [], ["person"], [], [])
        self.assertEqual(result.shape, frame.shape)

    def test_single_detection(self):
        frame = _blank_frame()
        class_names = ["person", "car"]
        boxes = [[50, 50, 100, 150]]
        confidences = [0.87]
        class_ids = [0]
        cx_list = [100]
        cy_list = [125]
        result = yolo3.draw_detections(
            frame, boxes, confidences, class_ids, class_names, cx_list, cy_list
        )
        self.assertEqual(result.shape, frame.shape)
        # Frame should no longer be all-black after drawing
        self.assertTrue(np.any(result != 0))

    def test_multiple_detections(self):
        frame = _blank_frame()
        class_names = ["person", "car", "dog"]
        boxes = [[10, 10, 80, 120], [200, 100, 60, 90]]
        confidences = [0.92, 0.71]
        class_ids = [0, 2]
        cx_list = [50, 230]
        cy_list = [70, 145]
        result = yolo3.draw_detections(
            frame, boxes, confidences, class_ids, class_names, cx_list, cy_list
        )
        self.assertEqual(result.shape, frame.shape)


class TestYolov3RunInference(unittest.TestCase):
    """run_inference returns the right structure (mocked network)."""

    def _make_detection_vector(self, class_id, confidence, cx, cy, w, h):
        """Build a fake YOLO detection row (85 values for 80-class COCO)."""
        row = np.zeros(85, dtype=np.float32)
        row[0] = cx
        row[1] = cy
        row[2] = w
        row[3] = h
        row[4] = confidence
        row[5 + class_id] = confidence
        return row

    def test_no_detections_above_threshold(self):
        net = MagicMock()
        # All scores below threshold
        det = self._make_detection_vector(0, 0.1, 0.5, 0.5, 0.2, 0.3)
        net.forward.return_value = [np.array([det])]

        with patch.object(yolo3, "get_output_layers", return_value=["layer1"]):
            boxes, confs, ids, cx, cy, t = yolo3.run_inference(
                net, _blank_frame(), input_size=416
            )

        self.assertEqual(boxes, [])
        self.assertEqual(confs, [])
        self.assertIsInstance(t, float)

    def test_single_detection_above_threshold(self):
        net = MagicMock()
        det = self._make_detection_vector(2, 0.9, 0.5, 0.5, 0.3, 0.4)
        net.forward.return_value = [np.array([det])]

        with patch.object(yolo3, "get_output_layers", return_value=["layer1"]):
            boxes, confs, ids, cx, cy, t = yolo3.run_inference(
                net, _blank_frame(480, 640), input_size=416
            )

        self.assertEqual(len(boxes), 1)
        self.assertEqual(len(confs), 1)
        self.assertAlmostEqual(confs[0], 0.9, places=3)
        self.assertEqual(ids[0], 2)

    def test_inference_time_positive(self):
        net = MagicMock()
        net.forward.return_value = [np.array([])]

        with patch.object(yolo3, "get_output_layers", return_value=["layer1"]):
            _, _, _, _, _, t = yolo3.run_inference(
                net, _blank_frame(), input_size=416
            )

        self.assertGreaterEqual(t, 0.0)

    def test_invalid_input_size_raises(self):
        with self.assertRaises(ValueError):
            yolo3.main(input_size=999)


# ══════════════════════════════════════════════════════════════════════════════
# tiny_yolov3_detector.py
# ══════════════════════════════════════════════════════════════════════════════

class TestTinyYoloConfig(unittest.TestCase):
    def test_confidence_threshold(self):
        self.assertEqual(tiny.CONFIDENCE_THRESHOLD, 0.5)

    def test_nms_threshold(self):
        self.assertEqual(tiny.NMS_THRESHOLD, 0.3)


class TestTinyYoloLoadClassNames(unittest.TestCase):
    def test_loads_names(self):
        names = ["person", "cat", "dog"]
        path = _make_names_file(names)
        try:
            loaded = tiny.load_class_names(path)
            self.assertEqual(loaded, names)
        finally:
            os.unlink(path)


class TestTinyYoloGetOutputLayers(unittest.TestCase):
    def _make_net(self, layer_names, unconnected):
        net = MagicMock()
        net.getLayerNames.return_value = layer_names
        net.getUnconnectedOutLayers.return_value = np.array(unconnected)
        return net

    def test_flat_indices(self):
        layers = ["conv1", "yolo_23", "yolo_30"]
        net = self._make_net(layers, [2, 3])
        result = tiny.get_output_layers(net)
        self.assertEqual(result, ["yolo_23", "yolo_30"])


class TestTinyYoloDrawDetections(unittest.TestCase):
    def test_no_detections(self):
        frame = _blank_frame()
        result = tiny.draw_detections(frame, [], [], [], ["person"], [], [])
        self.assertEqual(result.shape, frame.shape)

    def test_single_detection(self):
        frame = _blank_frame()
        result = tiny.draw_detections(
            frame,
            boxes=[[30, 40, 100, 120]],
            confidences=[0.75],
            class_ids=[1],
            class_names=["person", "bicycle"],
            cx_list=[80],
            cy_list=[100],
        )
        self.assertEqual(result.shape, frame.shape)
        self.assertTrue(np.any(result != 0))


class TestTinyYoloRunInference(unittest.TestCase):
    def _make_detection_vector(self, class_id, confidence, cx, cy, w, h):
        row = np.zeros(85, dtype=np.float32)
        row[0], row[1], row[2], row[3] = cx, cy, w, h
        row[4] = confidence          # objectness score
        row[5 + class_id] = confidence
        return row

    def test_no_detections_below_threshold(self):
        net = MagicMock()
        det = self._make_detection_vector(0, 0.1, 0.5, 0.5, 0.2, 0.3)
        net.forward.return_value = [np.array([det])]

        with patch.object(tiny, "get_output_layers", return_value=["l1"]):
            boxes, confs, ids, cx, cy, t = tiny.run_inference(
                net, _blank_frame(), input_size=416
            )

        self.assertEqual(boxes, [])

    def test_inference_returns_six_values(self):
        net = MagicMock()
        net.forward.return_value = [np.array([])]

        with patch.object(tiny, "get_output_layers", return_value=["l1"]):
            result = tiny.run_inference(net, _blank_frame(), input_size=416)

        self.assertEqual(len(result), 6)


# ══════════════════════════════════════════════════════════════════════════════
# ssd_mobilenetv3_detector.py
# ══════════════════════════════════════════════════════════════════════════════

class TestSSDConfig(unittest.TestCase):
    def test_confidence_threshold(self):
        self.assertEqual(ssd.CONFIDENCE_THRESHOLD, 0.5)

    def test_input_size(self):
        self.assertEqual(ssd.INPUT_WIDTH, 320)
        self.assertEqual(ssd.INPUT_HEIGHT, 320)

    def test_scale(self):
        self.assertAlmostEqual(ssd.SCALE, 1 / 127.5, places=7)

    def test_mean(self):
        self.assertEqual(ssd.MEAN, (127.5, 127.5, 127.5))


class TestSSDLoadClassNames(unittest.TestCase):
    def test_loads_names(self):
        names = ["background", "person", "bicycle"]
        path = _make_names_file(names)
        try:
            loaded = ssd.load_class_names(path)
            self.assertEqual(loaded, names)
        finally:
            os.unlink(path)


class TestSSDDrawDetections(unittest.TestCase):
    def test_no_detections(self):
        frame = _blank_frame()
        result = ssd.draw_detections(
            frame, [], [], [], ["person", "car"], [], []
        )
        self.assertEqual(result.shape, frame.shape)

    def test_single_detection(self):
        frame = _blank_frame()
        result = ssd.draw_detections(
            frame,
            class_ids=[0],
            confidences=[0.88],
            boxes=[[60, 80, 100, 120]],
            class_names=["person", "car"],
            cx_list=[110],
            cy_list=[140],
        )
        self.assertEqual(result.shape, frame.shape)
        self.assertTrue(np.any(result != 0))

    def test_class_id_out_of_range_uses_unknown(self):
        """Class ID beyond class_names list should not raise."""
        frame = _blank_frame()
        result = ssd.draw_detections(
            frame,
            class_ids=[99],
            confidences=[0.6],
            boxes=[[10, 10, 50, 50]],
            class_names=["person"],
            cx_list=[35],
            cy_list=[35],
        )
        self.assertEqual(result.shape, frame.shape)


class TestSSDRunInference(unittest.TestCase):
    def test_no_detections_returns_empty_lists(self):
        net = MagicMock()
        net.detect.return_value = (np.array([]), np.array([]), np.array([]))
        ids, confs, boxes, cx, cy, t = ssd.run_inference(net, _blank_frame())
        self.assertEqual(ids, [])
        self.assertEqual(confs, [])
        self.assertEqual(boxes, [])

    def test_single_detection_converts_1_indexed_class(self):
        """SSD class IDs are 1-indexed; run_inference must subtract 1."""
        net = MagicMock()
        net.detect.return_value = (
            np.array([[3]]),          # class id (1-indexed = 3 → 0-indexed = 2)
            np.array([[0.72]]),
            np.array([[100, 150, 80, 60]]),
        )
        ids, confs, boxes, cx, cy, t = ssd.run_inference(net, _blank_frame())
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0], 2)   # 3 - 1 = 2
        self.assertAlmostEqual(confs[0], 0.72, places=3)

    def test_center_coordinates_computed(self):
        net = MagicMock()
        net.detect.return_value = (
            np.array([[1]]),
            np.array([[0.8]]),
            np.array([[100, 200, 60, 80]]),  # x=100, y=200, w=60, h=80
        )
        ids, confs, boxes, cx, cy, t = ssd.run_inference(net, _blank_frame())
        self.assertEqual(cx[0], 130)   # 100 + 60//2
        self.assertEqual(cy[0], 240)   # 200 + 80//2

    def test_inference_time_returned(self):
        net = MagicMock()
        net.detect.return_value = (np.array([]), np.array([]), np.array([]))
        _, _, _, _, _, t = ssd.run_inference(net, _blank_frame())
        self.assertIsInstance(t, float)
        self.assertGreaterEqual(t, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# main() error-handling (no model files present)
# ══════════════════════════════════════════════════════════════════════════════

class TestMainFileNotFound(unittest.TestCase):
    """main() must raise FileNotFoundError when weight files are missing."""

    def test_yolov3_main_missing_weights(self):
        with self.assertRaises(FileNotFoundError):
            yolo3.main(input_size=416)

    def test_tiny_main_missing_weights(self):
        with self.assertRaises(FileNotFoundError):
            tiny.main()

    def test_ssd_main_missing_weights(self):
        with self.assertRaises(FileNotFoundError):
            ssd.main()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    unittest.main(verbosity=2)
