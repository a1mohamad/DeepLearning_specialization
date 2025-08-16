# Introduction to TensorFlow

This repository contains a Jupyter Notebook that serves as an introduction to TensorFlow 2.3. It's designed to help you get started with building neural networks using this powerful deep learning framework, moving beyond traditional NumPy implementations. **This is also the final assignment of Course 2 in Andrew Ng's Deep Learning Specialization.**

---

## Key Learnings & Achievements 🚀

By completing this assignment, you've achieved significant milestones in your deep learning journey:

* **Manipulated Variables**: You learned to use `tf.Variable` to create and modify the state of mutable variables within TensorFlow's computational graph.
* **Differentiated Constants from Variables**: You gained a clear understanding of the distinction between TensorFlow's immutable `tf.constant` and mutable `tf.Variable` types.
* **Implemented Automatic Differentiation**: You leveraged `tf.GradientTape` to automatically compute gradients for various operations, a cornerstone of backpropagation and neural network training.
* **Preprocessed Data for Neural Networks**: You applied essential preprocessing steps like one-hot encoding for categorical labels and transformed datasets into TensorFlow-compatible formats.
* **Built and Trained a Neural Network**: You successfully constructed and trained your first neural network using TensorFlow, demonstrating end-to-end model development from data loading to optimization.

---

## Why TensorFlow?

Machine learning frameworks like TensorFlow, PaddlePaddle, Torch, Caffe, and Keras significantly **accelerate machine learning development**. Beyond speeding up coding time, TensorFlow 2.3 offers substantial improvements and **performance optimizations** that enhance code efficiency.

---

## Dataset

This assignment utilizes the **Sign Language Digits Dataset**, comprising images representing digits from 0 to 5. These images, sized `64x64x3` pixels, are loaded into TensorFlow Datasets from `train_signs.h5` and `test_signs.h5` files. The notebook demonstrates how to access and inspect elements within these TensorFlow Datasets, which function as data generators.

---

## Notebook Structure

The notebook is organized into the following sections:

1.  **Packages**: Essential imports and a check of the TensorFlow version (v2.3 for this assignment).
2.  **Basic Optimization with GradientTape**: This section introduces `tf.Tensor` as the TensorFlow equivalent of NumPy arrays and demonstrates how `tf.GradientTape` handles automatic differentiation. You implemented fundamental operations:
    * Linear Function: Creating and manipulating a basic linear model.
    * Computing the Sigmoid: Implementing the sigmoid activation function.
    * Using One Hot Encodings: Converting labels to a one-hot format for classification tasks.
    * Initialize the Parameters: Setting up initial weights and biases for your neural network.
3.  **Building Your First Neural Network in TensorFlow**: This section guides you through the process of constructing and training a simple neural network. You were tasked with:
    * Implement Forward Propagation: Defining the layers and activations for the network's forward pass.
    * Compute the Total Loss: Calculating the overall loss to guide the optimization process.
    * Train the Model: Applying an optimizer and iterating through the dataset to minimize the loss and improve model performance.
4.  **Bibliography**: Resources for further diving into `tf.GradientTape` and automatic differentiation.