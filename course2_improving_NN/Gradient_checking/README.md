# Gradient Checking Notebook

This notebook contains my solution and notes for the **Gradient Checking** assignment from **Course 2: Improving Deep Neural Networks** of Andrew Ng's Deep Learning Specialization.

This notebook demonstrates the crucial process of **gradient checking**, a technique used to verify the correctness of a neural network's backpropagation implementation. Implementing backpropagation can be complex and error-prone, so gradient checking provides a reliable way to ensure that the gradients calculated are accurate.

***

## 📝 Problem Statement

The central problem addressed in this assignment is a **mission-critical application** to detect fraudulent mobile payments. Because backpropagation implementations can have bugs, the CEO of the company requires proof that the backpropagation algorithm is working correctly before the model can be deployed. This is where gradient checking comes in. By using gradient checking, we can mathematically prove that our implementation of the backward propagation step is correct, thus providing the necessary assurance.

***

## 📝 Notebook Contents

The notebook is divided into two main parts:

1.  **1-Dimensional Gradient Checking**: We start with a simple 1D linear function, $J(\theta) = \theta x$. This section implements the forward and backward propagation for this function and then uses gradient checking to verify that the derivative of $J$ with respect to $\theta$ is correctly calculated. This serves as a foundational example to understand the core concept.

2.  **N-Dimensional Gradient Checking**: This section scales up the concept to a deep neural network with multiple layers and parameters. We use the same principle but apply it to the entire parameter vector of the network. This involves:
    * Implementing a deep neural network's forward and backward propagation.
    * Converting the parameter dictionary into a single vector.
    * Iterating through each parameter in the vector, perturbing it slightly, and calculating a numerical approximation of the gradient.
    * Comparing this approximation to the gradient calculated by backpropagation to ensure they are within a small tolerance.

***

## 🔑 Key Concepts Covered

* **Forward Propagation**: The process of computing the output and cost of a neural network for a given input.
* **Backward Propagation**: The process of computing the gradients of the cost function with respect to the network's parameters.
* **Numerical Gradient Approximation**: Using the definition of a derivative to approximate the gradient, often using a small value $\epsilon$.
    $$\frac{\partial J}{\partial \theta} \approx \frac{J(\theta + \varepsilon) - J(\theta - \varepsilon)}{2 \varepsilon}$$
* **Gradient Checking**: The process of comparing the gradients from backpropagation with the numerically approximated gradients. If the difference is sufficiently small (typically less than $10^{-7}$), the backpropagation implementation is likely correct.

***

## 🚀 How to Use

To run this notebook, you will need a Python environment with the necessary libraries, including:

* **Jupyter**: To run the notebook.
* **NumPy**: For numerical operations.

You can install these using pip:
```bash
pip install jupyter numpy
```

Happy Learning !