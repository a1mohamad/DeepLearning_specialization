# 🧠 Regularization in Neural Networks – Football Header Prediction (Course 2, Week 2)

This repository contains my implementation of the **Regularization** assignment from **Course 2 – Improving Deep Neural Networks** of the [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) by Andrew Ng.

---

## 📌 Problem Statement

You have just been hired as an AI expert by the **French Football Corporation**.

Your mission is to help France’s **goalkeeper** decide **where to kick the ball** on the field so that **French players** are most likely to **win the header** (hit the ball with their head before the opponent).

The dataset comes from France’s past 10 games and includes:
- 2D coordinates of ball landing positions on the field
- Labels indicating whether the **French team** or the **opposing team** won the header

A baseline neural network performs well on training data but overfits and generalizes poorly to unseen matches.

### 🧠 Objective

Improve model generalization using **regularization techniques**:
- **L2 Regularization** (Weight decay)
- **Dropout Regularization**

You will also visualize the **decision boundaries** to evaluate how regularization affects the model's performance.

---

## 📂 Files in This Assignment

| File | Description |
|------|-------------|
| `Regularization_Assignment.ipynb` | Main Jupyter notebook with code and analysis |
| `reg_utils.py` | Helper functions provided by the course |
| `testCases.py` & `public_tests.py` | Unit tests for verifying correctness |

---

## 🔧 What You’ll Implement

### 🏁 Baseline Model
- A standard 3-layer neural network without regularization.

### 💪 L2 Regularization
- Add L2 penalty to cost function  
- Modify backpropagation to include weight decay

### 🪂 Dropout Regularization
- Randomly shut off neurons during training  
- Adjust both forward and backward propagation accordingly

---

## 📊 Results

| Model                        | Train Accuracy | Test Accuracy |
|-----------------------------|----------------|---------------|
| Without Regularization      | 95%            | 91.5%         |
| With L2 Regularization      | 94%            | 93%           |
| With Dropout Regularization | 93%            | **95%**       |

- ✅ **Dropout** gave the best generalization in this scenario.
- 📈 Visualizations clearly show more realistic decision boundaries when regularization is applied.

---


Happy learning !

