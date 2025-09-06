# Deep Residual Networks (ResNets)

This repository contains a Jupyter Notebook that guides you through building and training a deep convolutional neural network using Residual Networks (ResNets). ResNets, introduced by **He et al. (2015)**, address the vanishing gradient problem, allowing you to train much deeper networks than previously possible.

The notebook focuses on implementing the fundamental building blocks of ResNets:
- **Identity Block**: Used when the input and output dimensions of the network layer are the same.
- **Convolutional Block**: Used when the input and output dimensions differ, incorporating a stride to down-sample the activations and a 1x1 convolution in the shortcut path to match dimensions.

***

## What's Inside

The `ResNet50.ipynb` notebook provides a step-by-step implementation of a 50-layer ResNet model for image classification. You will:
1. Implement the identity block and convolutional block from scratch using Keras.
2. Stack these blocks to construct the full ResNet-50 architecture.
3. Train the model on the **SIGNS dataset**, a collection of images showing various hand signs.

***

## Running the Notebook

1. Clone this repository or download the notebook file.
2. Open the notebook in a Jupyter environment.
3. Follow the instructions within the notebook to complete the exercises and train your ResNet model.

***

## Note on Model Weights

For this assignment, we implement the ResNet-50 model from scratch. However, in practice, you can easily access pre-trained ResNet-50 models directly from `tf.keras.applications.ResNet50`. These pre-trained models are highly useful for **transfer learning** and can be loaded with their weights, saving significant training time.

Example of loading a pre-trained ResNet50 model from `tf.keras.applications`:
```python
from tensorflow.keras.applications import ResNet50

# Load model with weights pre-trained on ImageNet
model = ResNet50(weights='imagenet')
```
Due to their large size, the pre-trained weights (.h5 files) are not included in this repository. You can download these weights automatically when you load the model with weights='imagenet', or you can manually download them from the TensorFlow documentation and other online sources if needed.