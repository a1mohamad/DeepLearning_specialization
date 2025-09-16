# RNN and LSTM Implementation in NumPy

This project provides a comprehensive, from-scratch implementation of Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks using Python and the NumPy library. This project is designed to offer a deep, foundational understanding of the inner workings of these sequential models, including both forward and backward propagation.

## 1. Core Concepts and Notation

The project uses standard deep learning notation to describe tensor shapes and operations.

* **Superscript $[l]$**: Denotes the $l$-th layer.
* **Superscript $(i)$**: Refers to the $i$-th training example.
* **Superscript $\langle t \rangle$**: Represents an object at the $t$-th time step.
* **Subscript $i$**: Indicates the $i$-th entry of a vector.

For example, $a^{(2)[3]<4>}_5$ denotes the activation of the 2nd training example, at the 3rd layer, at the 4th time step, and the 5th entry in the vector.

---

## 2. Basic RNN Implementation

The basic RNN is a fundamental building block for processing sequential data. It maintains a hidden state that acts as a "memory," carrying information from past time steps.

### 2.1. Forward Propagation

The forward pass for an RNN involves two main functions:

* **`rnn_cell_forward`**: Implements a single time step of the RNN. It takes the current input $x^{\langle t \rangle}$ and the previous hidden state $a^{\langle t-1 \rangle}$ to compute the next hidden state $a^{\langle t \rangle}$ and the prediction $\hat{y}^{\langle t \rangle}$. The core equations are:
    * $a^{\langle t \rangle} = \tanh(W_{aa} a^{\langle t-1 \rangle} + W_{ax} x^{\langle t \rangle} + b_a)$
    * $\hat{y}^{\langle t \rangle} = \text{softmax}(W_{ya} a^{\langle t \rangle} + b_y)$
* **`rnn_forward`**: Orchestrates the entire forward pass by looping through the input sequence of length $T_x$, calling `rnn_cell_forward` at each step. It initializes the hidden state $a_0$ and stores all intermediate hidden states and predictions.

### 2.2. Backward Propagation (Optional/Ungraded)

Backpropagation Through Time (BPTT) is a variant of backpropagation for RNNs. It calculates the gradients with respect to the cost function to update the model parameters.

* **`rnn_cell_backward`**: Computes the gradients for a single RNN cell. It takes the upstream gradient `da_next` and the cached values from the forward pass to compute gradients for the input `dxt`, previous hidden state `da_prev`, and weights/biases (`dWax`, `dWaa`, `dba`).
* **`rnn_backward`**: Loops backward through the entire sequence, summing the gradients from each time step to get the total gradients for the model's parameters.

---

## 3. Long Short-Term Memory (LSTM) Network

LSTMs are a sophisticated type of RNN designed to address the vanishing gradient problem. They utilize a **cell state** ($c^{\langle t \rangle}$) to store long-term memory, regulated by a system of gates. 

### 3.1. Forward Propagation

The LSTM forward pass also consists of two functions:

* **`lstm_cell_forward`**: This function is the core of the LSTM. It computes the three gates (forget, update, output) and updates both the cell state and the hidden state.
    * **Forget Gate ($\mathbf{\Gamma}_{f}$)**: Controls what information from the old cell state ($c^{\langle t-1 \rangle}$) to discard.
    * **Update Gate ($\mathbf{\Gamma}_{i}$)**: Determines what new information from the candidate value ($\mathbf{\tilde{c}}^{\langle t \rangle}$) to add to the cell state.
    * **Output Gate ($\mathbf{\Gamma}_{o}$)**: Regulates what information from the new cell state ($c^{\langle t \rangle}$) is used to compute the hidden state ($a^{\langle t \rangle}$).
* **`lstm_forward`**: This function iterates the `lstm_cell_forward` function across the input sequence, updating the hidden and cell states at each time step.

### 3.2. Backward Propagation (Optional/Ungraded)

The backward pass for LSTMs is more complex due to the intricate gate mechanisms.

* **`lstm_cell_backward`**: Computes gradients for a single LSTM cell. This involves backpropagating through the complex gate operations to determine the gradients with respect to the previous hidden and cell states (`da_prev`, `dc_prev`) as well as the input `dxt` and all weight/bias matrices.
* **`lstm_backward`**: This function performs the full BPTT for the LSTM. It loops backward through the time steps, accumulates gradients from each cell, and correctly handles the summation of gradients for the shared parameters (`dWf`, `dWi`, `dWc`, `dWo`, `dbf`, `dbi`, `dbc`, `dbo`). The final gradients are returned in a dictionary.

## What you should remember

* An LSTM is an advanced form of an RNN that solves the vanishing gradient problem by using a **cell state** for long-term memory and **three gates** to regulate information flow.
* The **forget gate** decides what to discard from the previous state.
* The **update gate** determines what new information to add.
* The **output gate** controls what gets passed on to the hidden state and prediction.
* Both RNNs and LSTMs rely on a repeated cell structure over time, making them well-suited for processing sequences.