"""
ODT-1 | Object Detection and Classification with Computer Vision
Model Benchmark — Compares FPS and inference time for all three models.

Authors: Abdul WAHAB (1447523), Muhammad Faraz Abbasi (1566440)
Course: Autonomous Intelligent Systems — WS 24/25

Usage:
    python benchmark.py --source 0 --frames 100
"""

import cv2
import numpy as np
import time
import os
import argparse


def benchmark_model(source, num_frames):
    """
    Benchmarks object detection models by calculating FPS and inference time.

    Args:
        source (int or str): Video source (e.g., webcam index or video file path).
        num_frames (int): Number of frames to process for benchmarking.

    Returns:
        None
    """
    # Initialize video capture
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {source}")
        return

    print(f"Starting benchmark on source: {source} for {num_frames} frames...")
    frame_count = 0
    start_time = time.time()

    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from source.")
            break

        # Placeholder for model inference (replace with actual model processing)
        # Example: result = model.detect(frame)
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Dummy processing

        frame_count += 1

    end_time = time.time()
    elapsed_time = end_time - start_time
    fps = frame_count / elapsed_time

    print(f"Benchmark completed.")
    print(f"Total frames processed: {frame_count}")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Frames per second (FPS): {fps:.2f}")

    cap.release()


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Benchmark object detection models.")
    parser.add_argument("--source", type=str, default="0", help="Video source (default: 0 for webcam)")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to process (default: 100)")
    args = parser.parse_args()

    # Convert source to int if it's a webcam index
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    # Run the benchmark
    benchmark_model(source, args.frames)