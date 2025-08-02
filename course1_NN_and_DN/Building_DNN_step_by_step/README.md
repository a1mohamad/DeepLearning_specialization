# 🚀 Building Your Deep Neural Network: Step by Step



This project is part of **Week 4 (Part 1)** of the **Deep Learning Specialization by Andrew Ng** COURSE 1.  

Here, I implemented a **deep neural network from scratch using only NumPy**, building on the 2-layer network from earlier weeks.



---



## **Objective**

- Understand how **L-layer neural networks** work under the hood.

- Implement both **forward** and **backward propagation** for any depth.

- Prepare reusable functions for future assignments (e.g., image classification).



---



## **Key Features**

1. **Customizable Depth** – Build networks with as many layers as needed.

2. **Non-linear Activations** – Use **ReLU** and **Sigmoid** to enable complex decision boundaries.

3. **Cross-Entropy Cost Function** – For binary classification tasks.

4. **Gradient Descent** – Update parameters through backpropagation.



---



## **Project Structure**



### **1 - Packages**

Import essential libraries: `numpy`, `matplotlib`, and helper functions.



### **2 - Outline**

Step-by-step construction of a deep neural network.



### **3 - Initialization**

- **3.1 - 2-layer Neural Network**:  

`initialize\_parameters()` – Initialize weights and biases for a simple 2-layer model.  

- **3.2 - L-layer Neural Network**:  
`initialize_parameters_deep()` – Generalize initialization for any depth.



### **4 - Forward Propagation**

- **4.1 - Linear Forward**:  

`linear_forward()` – Compute `Z = W*A + b`.  

- **4.2 - Linear-Activation Forward**:  

`linear_activation_forward()` – Apply activation functions (ReLU/Sigmoid).  

- **4.3 - L-Layer Model**:  

`L_model_forward()` – Chain multiple layers for full forward pass.



### **5 - Cost Function**

- `compute_cost()` – Calculate cross-entropy loss.



### **6 - Backward Propagation**

- **6.1 - Linear Backward**:  

	`linear_backward()` – Compute gradients for weights and biases.  

- **6.2 - Linear-Activation Backward**:  

	`linear_activation\_backward()` – Backprop through activation functions.  

- **6.3 - L-Model Backward**:  

	`L_model_backward()` – Full backward propagation through all layers.  

- **6.4 - Update Parameters**:  

	`update_parameters()` – Gradient descent update for weights and biases.



---



Let's have some Learning stuffs !

