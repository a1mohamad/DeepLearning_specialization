# Convolutional Neural Networks: Step by Step

Welcome to the first assignment of Course 4! In this notebook, you will implement the core building blocks of a convolutional neural network (CNN) from scratch. Using **NumPy**, you'll code both the forward and (optionally) the backward propagation for convolutional and pooling layers. This hands-on experience will give you a deep understanding of how CNNs work under the hood. 🧠

---

## What You'll Learn

By the end of this assignment, you'll be able to:

* **Explain the convolution operation** and its purpose in a neural network.
* **Apply two different types of pooling** operations: Max pooling and Average pooling.
* **Identify the key components** of a convolutional layer, such as padding, stride, and filters.
* **Build a full convolutional neural network** by combining these essential components.

---

## Assignment Breakdown

The assignment is divided into two main parts, focusing on the implementation of the core layers.

### 1. Convolutional Layer

In this section, you'll build a convolutional layer from the ground up, implementing each step as a separate function.

* **Zero-Padding:** You will start by implementing a function that adds zeros around the border of an image. This technique is vital for controlling the output size of a convolutional layer and for retaining information at the image's edges.
* **Single-Step Convolution:** Next, you'll create a function that performs a single convolution step, where a filter is applied to a specific slice of the input to produce a single output value.
* **Forward Propagation:** Finally, you'll combine these helper functions to implement the complete **forward propagation** for a convolutional layer. You'll apply multiple filters to the input volume, producing a new, transformed output volume.

The output dimensions of a convolutional layer are calculated using the following formulas:

$n_H = \lfloor \frac{n_{H_{prev}} - f + 2 \times pad}{stride} \rfloor +1$

$n_W = \lfloor \frac{n_{W_{prev}} - f + 2 \times pad}{stride} \rfloor +1$

$n_C = \text{number of filters used in the convolution}$

### 2. Pooling Layer

The pooling layer is essential for reducing the spatial dimensions of the feature maps, which helps decrease computational cost and makes the features more robust to their location in the input.

* **Forward Propagation:** You'll implement the **forward pass** for two types of pooling: **Max pooling** (taking the maximum value within a window) and **Average pooling** (computing the average value). You'll perform this operation over the input volume, producing a downsampled output.

The output dimensions for a pooling layer are determined by:

$n_H = \lfloor \frac{n_{H_{prev}} - f}{stride} \rfloor +1$

$n_W = \lfloor \frac{n_{W_{prev}} - f}{stride} \rfloor +1$

$n_C = n_{C_{prev}}$

### 3. Backpropagation (Optional)

For those who want to deepen their understanding, this section guides you through implementing the **backward pass** for both the convolutional and pooling layers. This is a crucial step for training the model and computing the gradients needed for optimization.

---

## Prerequisites

To complete this assignment, you should be familiar with Python's **NumPy** library. If you have completed the previous courses in this specialization, you will be well-prepared. Let's get started! 💻