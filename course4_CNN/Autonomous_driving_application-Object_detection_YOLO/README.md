# Autonomous Driving - Car Detection

This project notebook implements a powerful car detection system for autonomous driving using the YOLO (You Only Look Once) model. It serves as a practical guide to understanding and building a complete object detection pipeline.

## 1. Project Overview

This notebook guides you through the process of implementing and applying the YOLO model to a car detection dataset. You will learn the foundational concepts and practical steps required to build a robust object detection system from scratch.

This project uses the `drive.ai` dataset, and the license is already provided.

By completing the exercises in this notebook, you will be able to:

* Detect objects in a car detection dataset.

* Implement Intersection over Union (IoU) to measure the accuracy of bounding boxes.

* Apply Non-max Suppression (NMS) to filter out redundant detections and improve accuracy.

* Handle bounding boxes, a key component of image annotation.

## 2. Key Concepts

### YOLO (You Only Look Once)

YOLO is an innovative object detection algorithm known for its speed and accuracy. Its foundation is its unique approach of dividing the input image into a grid. Each grid cell is responsible for detecting objects within its boundaries and predicting the bounding box coordinates, a confidence score, and class probabilities in a single pass. This project utilizes the YOLO model in conjunction with `YAD2K` (Yet Another Darknet to Keras), a tool for converting pre-trained models from the Darknet framework into a format compatible with Keras.

### Intersection over Union (IoU)

IoU is a crucial metric used to evaluate the overlap between a predicted bounding box and a ground-truth bounding box. A higher IoU value indicates a more accurate prediction.

### Non-max Suppression (NMS)

NMS is a post-processing technique used to eliminate duplicate or overlapping bounding boxes for the same object, ensuring that only the most confident detection is kept. This is essential for a clean and accurate final output.

## 3. Model Details and Implementation

Training a YOLO model is a very time-consuming process that requires a large dataset with labelled bounding boxes. Therefore, this notebook loads an existing pre-trained Keras YOLO model from the file `yolo.h5`.

These weights were originally sourced from the official [YOLO](https://docs.ultralytics.com/) website and were converted from the Darknet framework using a function by Allan Zelener. While referred to as "YOLO" in this notebook, these are technically the parameters from the "YOLOv2" model.

## 4. File Structure and Prerequisites

Due to upload limitations, the full model variables are not included. However, the `model_data` directory contains essential components to get started, such as the initial anchor boxes, the model architecture, and the COCO classes.

To see the model's predictions, you can run the notebook and visualize the results on the images in the `nb_images` directory. Example outputs are also provided in the `out` folder.

Let's start !