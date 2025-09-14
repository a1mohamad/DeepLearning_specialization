# Neural Style Transfer Project

## Overview

This project implements the Neural Style Transfer (NST) algorithm, a deep learning technique that combines the "content" of one image with the "style" of another to create a new, artistic image. This is achieved by defining and minimizing a cost function that measures the differences in content and style between the generated image and the original content and style images.

The algorithm, based on the work of Gatys et al. (2015), uses a pre-trained Convolutional Neural Network (CNN) as a feature extractor.

## Core Components

### 1. The VGG19 Model

The project uses the **VGG19 model**, a 19-layer CNN pre-trained on the ImageNet database. This model has learned to recognize a wide range of features, from simple edges in shallow layers to complex objects in deeper layers.

**Note on VGG19 Weights:**
For this project, we are using a version of the VGG19 weights file, specifically `vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5`. These weights were downloaded locally. Due to file size limitations, this file is not included in the repository. Users can either:

* Use the built-in `tf.keras.applications.VGG19` with `weights='imagenet'`.
* Download the "no top" version of the weights directly from Google.

### 2. Content Cost Function ($$J_{content}(C,G)$$)

The content cost measures how different the generated image ($$G$$) is from the content image ($$C$$) in terms of its high-level features. We use a middle layer of the pre-trained VGG19 network to represent the content. The cost is calculated as the squared difference between the hidden layer activations of the content image and the generated image. Minimizing this cost ensures that the generated image retains the core structure and objects of the content image.

$$J_{content}(C,G)=\frac{1}{4 \times n_H \times n_W \times n_C} \sum_{\text{all entries}}(a^{(C)} - a^{(G)})^2$$

### 3. Style Cost Function ($$J_{style}(S,G)$$)

The style cost measures how different the generated image ($$G$$) is from the style image ($$S$$) in terms of its artistic style. Style is represented by the Gram matrix, which captures the correlations between the different feature activations within a hidden layer. By minimizing the difference between the Gram matrices of the style image and the generated image, we enforce a similar "texture" and color pattern. This process is applied across multiple layers of the VGG19 network to capture different levels of style.

$$J_{style}^{[l]}(S,G)=\frac{1}{4 \times n_C^2 \times (n_H \times n_W)^2} \sum_{i=1}^{n_C} \sum_{j=1}^{n_C}(G^{(gram)}_{i,j(S)} - G^{(gram)}_{i,j(G)})^2$$

The overall style cost is a weighted sum of the style costs from several layers:

$$J_{style}(S,G)=\sum_l \lambda^{[l]} J_{style}^{[l]}(S,G)$$

### 4. Total Cost ($$J(G)$$)

The final objective is to minimize the total cost function, which is a weighted sum of the content and style costs. The hyperparameters $$\alpha$$and$$\beta$$ control the trade-off between preserving the content and adopting the style.

$$J(G)=\alpha J_{content}(C,G)+\beta J_{style}(S,G)$$

The generated image ($$G$$) is initialized as a noisy version of the content image, and its pixel values are optimized using an Adam optimizer to minimize the total cost.

## How it Works: A Simple Analogy

Imagine you are an artist's apprentice given two tasks: "Draw this" (your content image) and "Make it look like this" (your style image).

Neural Style Transfer works by using a pre-trained neural network (like VGG19) as its "eyes" to understand both images.

* **Understanding the Content**: The algorithm looks at your content image and identifies its core structure and objects—the "blueprint." The deeper layers of the network are excellent at this.
* **Understanding the Style**: It then analyzes the style image, but instead of focusing on objects, it captures the texture, color, and brushstrokes. It does this by creating a Gram matrix, which is like a computer-generated "palette" that defines the artistic feel of the painting.
* **The Creative Process**: The algorithm starts with a blank or noisy image. Its goal is to change the pixels of this new image to simultaneously:
    * Match the "blueprint" of the content image.
    * Match the "palette" of the style image.

The algorithm uses an optimizer to continuously adjust the pixels until it finds the perfect balance between these two goals, creating a new image that looks like your content but with the style of the other image.

## References

The Neural Style Transfer algorithm was due to Gatys et al. (2015). Harish Narayanan and Github user "log0" also have highly readable write-ups this lab was inspired by. The pre-trained network used in this implementation is a VGG network, which is due to Simonyan and Zisserman (2015). Pre-trained weights were from the work of the MathConvNet team.

* Leon A. Gatys, Alexander S. Ecker, Matthias Bethge, (2015). A Neural Algorithm of Artistic Style.
* Harish Narayanan, Convolutional neural networks for artistic style transfer.
* Log0, TensorFlow Implementation of "A Neural Algorithm of Artistic Style".
* Karen Simonyan and Andrew Zisserman (2015). Very deep convolutional networks for large-scale image recognition.
* MatConvNet.