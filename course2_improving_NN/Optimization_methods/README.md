# Optimization Methods for Deep Learning

This folder contains a Jupyter Notebook that explores and implements various optimization algorithms for training neural networks. The content is part of **Course 2 of the Deep Learning Specialization** by Andrew Ng on Coursera.

## Overview

This notebook dives into the world of optimization methods beyond standard Gradient Descent. By implementing and comparing different algorithms, you'll gain a deeper understanding of how they can accelerate the training process and improve model performance.

## Key Concepts Covered

The notebook guides you through the implementation of several key optimization techniques:

- **(Batch) Gradient Descent**: The fundamental optimization algorithm, which updates parameters by considering all training examples at each step.
- **Stochastic Gradient Descent (SGD)**: A variant of Gradient Descent where parameters are updated after processing just a single training example. This can be faster for large datasets but leads to a more "oscillatory" path towards the minimum.
- **Mini-Batch Gradient Descent**: An intermediate approach that uses a small subset of the training data at each step. This method strikes a balance between the computational efficiency of SGD and the stable convergence of Batch Gradient Descent.
- **Momentum**: An algorithm that accelerates gradient descent in the relevant direction and dampens oscillations. It takes into account the exponentially weighted average of past gradients, which acts like a "velocity."
- **Adam (Adaptive Moment Estimation)**: One of the most popular and effective optimization algorithms for deep learning. Adam combines the best of Momentum and RMSProp, using exponentially weighted averages of both the gradients and the squared gradients to adapt the learning rate for each parameter.
- **Learning Rate Decay**: A technique to gradually decrease the learning rate over time, which can help the model converge more effectively.

---

## Notebook Structure

The notebook is divided into the following sections:

- **Packages**: Importing all necessary libraries.
- **Gradient Descent**: A brief review and implementation of the basic gradient descent update rule.
- **Mini-Batch Gradient Descent**: Detailed explanation and implementation of the process for creating mini-batches, including shuffling and partitioning the dataset.
- **Momentum**: Implementation of the `initialize_velocity` and `update_parameters_with_momentum` functions.
- **Adam**: Implementation of the `initialize_adam` and `update_parameters_with_adam` functions, including bias correction.
- **Model with Different Optimization Algorithms**: A section dedicated to training a neural network on the "moons" dataset using each of the implemented optimizers and visualizing their performance.
- **Learning Rate Decay and Scheduling**: An exploration of different strategies for decaying the learning rate.

---

## Getting Started

To run this notebook, you'll need a Python environment with the following libraries installed:

- `numpy`
- `matplotlib`
- `scipy`
- `sklearn`

You can install these using `pip`:

```bash
pip install numpy matplotlib scipy scikit-learn
```

Once the dependencies are installed, you can open and run the `Optimization_methods.ipynb` notebook in your preferred Jupyter environment (e.g., JupyterLab, VS Code with the Jupyter extension).

**Note**: This folder includes a `opt_utils_v1a.py` file, which contains several helper functions for initializing parameters, running forward and backward propagation, and testing the optimization algorithms.

## Acknowledgements
This notebook is based on the assignments and materials from Course 2 of the Deep Learning Specialization by Andrew Ng on Coursera. It's a great resource for anyone looking to build a strong foundation in deep learning.

Happy Learning !
