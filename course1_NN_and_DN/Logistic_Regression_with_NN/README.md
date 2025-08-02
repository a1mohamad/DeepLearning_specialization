#  Logistic Regression with a Neural Network Mindset

This is the **first required programming assignment** from **Course 1: Neural Networks and Deep Learning** in the **Deep Learning Specialization by Andrew Ng**.  
The goal is to build a **logistic regression classifier** to recognize **cats vs. non-cats**—but with a **neural network mindset** that sets the foundation for deeper models.

---

## **Objective**
- Understand how to **build a learning algorithm from scratch**.
- Learn the fundamental steps used in **deep learning models**:
  1. **Initialize parameters**
  2. **Compute cost and gradients**
  3. **Optimize parameters using gradient descent**
- Merge all parts into a single **model function**.

---

## **What You’ll Learn**
- Implement the **sigmoid activation function**.
- Build **forward** and **backward propagation** for logistic regression.
- Use **vectorized operations** (no loops!) for efficiency.
- Test the model on a **cat dataset** and even **your own image**.

---

## **Project Structure**

### **1 - Packages**
Import libraries such as:
- `numpy` for computations
- `h5py` to handle dataset files
- `matplotlib` for visualization

---

### **2 - Overview of the Problem Set**
Familiarize yourself with the dataset and preprocessing steps.

---

### **3 - General Architecture of the Learning Algorithm**
Outline the key components of logistic regression:
- Parameter initialization
- Cost function and gradients
- Optimization (gradient descent)

---

### **4 - Building the Parts of the Algorithm**
#### **4.1 - Helper Functions**
- `sigmoid()` – Implements the sigmoid activation.

#### **4.2 - Initializing Parameters**
- `initialize_with_zeros()` – Creates zero-initialized weights and bias.

#### **4.3 - Forward and Backward Propagation**
- `propagate()` – Computes cost and gradients.

#### **4.4 - Optimization**
- `optimize()` – Updates weights using gradient descent.
- `predict()` – Makes predictions on new data.

---

### **5 - Merge All Functions into a Model**
- `model()` – Combines all steps into one function to train and evaluate the classifier.

---

### **6 - Further Analysis (Optional/Ungraded)**
- Evaluate how different learning rates affect convergence.

---

### **7 - Test with Your Own Image (Optional/Ungraded)**
- Try the model on an external image to test generalization.

---

Good Luck and happy learning !