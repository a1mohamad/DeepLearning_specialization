# Image Segmentation with U-Net

This repository presents a comprehensive guide and implementation of the U-Net convolutional neural network architecture for semantic image segmentation. The model is trained on a local self-driving car dataset to perform pixel-level classification, a critical task for autonomous systems.

## 1. Project Overview

Semantic image segmentation involves classifying each pixel in an image into a specific category, effectively creating a detailed mask of the objects present. This project utilizes the U-Net architecture, a state-of-the-art solution for dense prediction tasks. U-Net's unique design, featuring a symmetric encoder-decoder structure and skip connections, allows it to capture both high-level contextual information and fine-grained spatial details, leading to precise segmentation masks.

The accompanying Jupyter notebook assumes that you already have a CARLA semantic segmentation dataset stored locally on your machine cause I already downloded across course on coursera host. For those who need a dataset, a wide variety of autonomous driving datasets can be found and downloaded from resources such as [Kaggle](https://www.kaggle.com/datasets/kumaresanmanickavelu/lyft-udacity-challenge), [Roboflow](https://universe.roboflow.com/carla-vuxbn/self-driving-7eigg), [Hugging Face](https://huggingface.co/datasets/isp-uv-es/IPL-CARLA-dataset), or the official CARLA website, which provides datasets for autonomous driving research. Please note that for very large datasets, direct uploading might not be feasible due to file size limitations.

For more information on the CARLA project and to explore their datasets, you can visit the official [Carla](https://carla.org/) website

The notebook will guide you through the following steps:

* **Data Handling**: Loading and preprocessing images and their corresponding segmentation masks from your local dataset.

* **Model Architecture**: A detailed walkthrough of building the U-Net model from foundational layers, including custom `conv_block` and `upsampling_block` functions.

* **Training Pipeline**: Compiling the model with `SparseCategoricalCrossentropy` for pixel-wise multi-class prediction and training on the processed dataset.

* **Inference & Visualization**: Generating and visualizing predicted segmentation masks to evaluate the model's performance.

## 2. Key Concepts

* **Semantic Segmentation**: The task of assigning a class label to every pixel in an image.

* **U-Net**: A specialized CNN for segmentation, characterized by its U-shaped architecture.

* **Encoder (Contracting Path)**: The downsampling part of the network that extracts features and reduces the spatial dimensions of the input.

* **Decoder (Expanding Path)**: The upsampling part that reconstructs the segmented image from the compressed feature maps.

* **Skip Connections**: Direct links between the encoder and decoder that transfer feature maps, preserving spatial information and mitigating data loss during upsampling.

* **`SparseCategoricalCrossentropy`**: An optimal loss function for multi-class classification when labels are integer-encoded, suitable for pixel-wise predictions.

## 3. Results

The trained model demonstrates a high degree of accuracy in segmenting images from the CARLA dataset. The predictions, when compared to the ground truth masks, show that the model successfully learns to distinguish between different classes (e.g., road, person, car) at the pixel level.

The provided notebook includes a visual comparison of the input image, the true segmentation mask, and the predicted mask, showcasing the model's ability to produce precise, detailed output.