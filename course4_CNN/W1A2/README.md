# Convolutional Neural Networks: Application

This README provides an overview of a Jupyter notebook that guides you through building and training **convolutional neural networks (ConvNets)** using TensorFlow and Keras. This **assignment** consists of two parts, each demonstrating a different application of ConvNets and a specific Keras API.

---

## What You'll Learn

The notebook is divided into two main sections:

### 1. The Sequential API: Building a Mood Classifier

In the first part, you'll tackle a **binary classification problem** using the **Happy House** dataset. This dataset contains images of faces, which are labeled as either smiling or not smiling. Your objective is to build a model that can classify a person's mood based on their facial expression.

You'll use the **TF Keras Sequential API** to construct a simple, linear model for this task. The architecture you'll implement follows a straightforward sequence of layers: a **ZeroPadding2D** layer to pad the images, a **Conv2D** layer with 32 filters, followed by **BatchNormalization**, a **ReLU** activation, and **MaxPool2D** for downsampling. Finally, a **Flatten** layer will prepare the data for the output **Dense** layer, which will use a **sigmoid** activation to predict the binary outcome.



---

### 2. The Functional API: Classifying Sign Language Digits

The second part of the assignment presents a more complex **multiclass classification problem**. You'll work with the **SIGNS** dataset, which consists of images of hand signs for digits 0 through 5. The goal is to build a ConvNet that can accurately identify the correct digit from each image.

To handle this more complex task, you'll use the **TF Keras Functional API**. This API offers a more flexible approach to model creation, allowing you to define models with non-linear topologies, multiple inputs, or multiple outputs. This section of the notebook will help you understand when to use the more powerful Functional API over the simpler Sequential API.

---

## Prerequisites

To complete this assignment, you should be familiar with the fundamentals of **TensorFlow**. If you need a refresher on the basics, please refer to the **TensorFlow Tutorial** from Course 2 ("Improving deep neural networks").