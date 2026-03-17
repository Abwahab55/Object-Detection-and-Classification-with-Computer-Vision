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


def load_yolo_model(config_path, weights_path):
    """
    Loads the Tiny-YOLOv3 model.

    Args:
        config_path (str): Path to the YOLO configuration file (.cfg).
        weights_path (str): Path to the YOLO weights file (.weights).

    Returns:
        net: Loaded DNN model.
    """
    net = cv2.dnn.readNet(weights_path, config_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net


def detect_objects_yolo(net, source, confidence_threshold=0.5, nms_threshold=0.4):
    """
    Performs object detection using Tiny-YOLOv3 on a video source.

    Args:
        net: Loaded YOLO DNN model.
        source (int or str): Video source (e.g., webcam index or video file path).
        confidence_threshold (float): Minimum confidence for detections.
        nms_threshold (float): Non-maximum suppression threshold.

    Returns:
        None
    """
    # Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {source}")
        return

    # Load class labels
    class_labels = []
    with open("coco.names", "r") as f:
        class_labels = f.read().strip().split("\n")

    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    print("Starting object detection...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from source.")
            break

        # Prepare the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        h, w = frame.shape[:2]

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > confidence_threshold:
                    box = detection[0:4] * np.array([w, h, w, h])
                    center_x, center_y, width, height = box.astype("int")
                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{class_labels[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Tiny-YOLOv3 Object Detection", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Paths to the model files
    CONFIG_PATH = os.path.join(current_dir, "yolov3-tiny.cfg")
    WEIGHTS_PATH = os.path.join(current_dir, "yolov3-tiny.weights")

    # Argument parsing
    import argparse
    parser = argparse.ArgumentParser(description="Tiny-YOLOv3 Object Detection")
    parser.add_argument("--source", type=str, default="0", help="Video source (default: 0 for webcam)")
    args = parser.parse_args()

    # Convert source to int if it's a webcam index
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # Load the model
    print("Loading Tiny-YOLOv3 model...")
    model = load_yolo_model(CONFIG_PATH, WEIGHTS_PATH)

    # Run object detection
    detect_objects_yolo(model, source, confidence_threshold=0.5, nms_threshold=0.4)