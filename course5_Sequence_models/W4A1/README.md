# Transformer Network: Implementation from Scratch

---

## Description

This **project** contains the completed assignment for **Week 4 of Course 5** (Sequence Models) of the Deep Learning Specialization.

The notebook focuses on building the **Transformer architecture**, a neural network that utilizes parallel processing and the attention mechanism to achieve significant speed improvements over traditional sequential models like RNNs, GRUs, and LSTMs.

This is the final graded assignment of the specialization.

## Project Objectives

Upon completing the exercises in the notebook, you'll be able to:

* Create **positional encodings** to capture sequential relationships in data.
* Calculate **scaled dot-product self-attention** with word embeddings.
* Implement **masked multi-head attention**.
* Build and train a complete **Transformer model**.

## Required Packages

The implementation relies primarily on the following libraries:

* `tensorflow` (with `keras` layers)
* `numpy`
* `matplotlib`

Specific Keras layers used include `Embedding`, `MultiHeadAttention`, `Dense`, `Input`, `Dropout`, and `LayerNormalization`.

## Notebook Contents

The notebook is structured to guide you through the implementation of each key component of the Transformer:

### 1. Positional Encoding
Implementation of the sine and cosine formulas to inject positional information into word embeddings, including the `get_angles()` and `positional_encoding()` helper functions.

### 2. Masking
Creation of the **Padding Mask** and **Look-ahead Mask** for use in the attention mechanisms.

### 3. Self-Attention
Implementation of the core `scaled_dot_product_attention()` function.

### 4. Encoder
Implementation of the `EncoderLayer` and the full `Encoder` class, which stacks multiple encoder layers.

### 5. Decoder
Implementation of the `DecoderLayer` and the full `Decoder` class, incorporating masked multi-head attention and encoder-decoder attention.

### 6. Transformer
Assembling all components into the final **Transformer** model class.

---

## References

The Transformer architecture is based on the seminal paper:

* **Attention Is All You Need** by Ashish Vaswani et al. (2017)