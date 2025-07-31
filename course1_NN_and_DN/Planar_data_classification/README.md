#  Planar Data Classification with a Single Hidden Layer



Welcome to this programming assignment!  

In this project, you’ll build your **first neural network** with a **single hidden layer**, designed to solve a **binary classification problem** on 2D planar data.



##  Objectives



By the end of this project, you will be able to:



-  Implement a binary classification neural network with a single hidden layer

-  Use non-linear activation functions like `tanh`

-  Compute cross-entropy loss for classification tasks

-  Implement forward and backward propagation

-  Train the model using gradient descent




##  Project Structure



### **1. Packages**

The project uses:



- `NumPy`: For numerical computation

- `matplotlib`: For plotting

- `scikit-learn`: For loading and managing datasets


### **2. Load the Dataset**

You’ll work with a **2D "flower-shaped" dataset**.  

Each data point is labeled as either:

- 🔴 `y = 0` (red)

- 🔵 `y = 1` (blue)


**Goal**: Train a model to separate these two regions.


### **3. Simple Logistic Regression (Baseline)**



Before building the neural network, you will:

- Train a **logistic regression model**

- Observe its performance and limitations on non-linearly separable data



> This serves as a baseline to show the power of neural networks.

##  Neural Network Model



The core part of the assignment is building a full neural network step-by-step.

### **4.1 Define the Network Architecture**

```python

layer_sizes()

```

Defines the size of input, hidden, and output layers.

### **4.2 Initialize Parameters**

```python

initialize_parameters()

```

Initializes weights and biases for the model.

### **4.3 Forward Propagation**

```python

forward_propagation()

```

Computes predictions from input using current weights.

### **4.4 Compute the Cost**

```python

compute_cost()

```

Calculates the cross-entropy loss.

### **4.5 Backward Propagation**

```python

backward_propagation()

```

Computes the gradients of the cost w.r.t. weights and biases.

### **4.6 Update Parameters**

```python

update_parameters()

```

Updates weights using gradient descent.

### **4.7 Integrate All Steps**

```python

nn_model()

```

Combines all functions into a full training loop.

##  Model Evaluation

```python

predict()

```

Predicts new data points using trained parameters.

## 5.2 Visualize Results

- Plot the decision boundary

- Compare results with logistic regression baseline

## Optional: Tuning Hidden Layer Size

* Experiment with various hidden layer sizes such as 1, 5, 10, 50 to:

* Understand the impact of model complexity

* Explore underfitting vs. overfitting

###  Bonus: Performance on Other Datasets

Try your model on additional 2D datasets to evaluate its generalization performance.


###  Tips to Explore

* Visualize gradients or activation maps

* Test different activation functions like ReLU

* Add an additional hidden layer
