# Soccer Object Detection and Tracking

> **Note**: This project is currently under development.

---

## Overview

This project focuses on detecting and tracking objects in soccer videos, specifically players, referees, and the ball. It uses a YOLO  model for object detection and the ByteTrack algorithm for tracking objects across frames. The project is implemented in Python and leverages libraries such as Ultralytics, Supervision, and OpenCV.

---

## Features

- **Object Detection**: Uses a pre-trained YOLO model to detect players, referees, and the ball in each frame of a soccer video.
- **Object Tracking**: Implements the ByteTrack algorithm to maintain the identity of detected objects across multiple frames.
- **Data Serialization**: Supports serialization and deserialization of tracking data using Python's `pickle` module.
- **Visualization**: Provides functionality to draw bounding boxes and ellipses around detected objects for visualization.

---

## Project Structure
