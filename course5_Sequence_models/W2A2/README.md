# Emojify! 🤩💫🔥

Welcome to the Emojify project! This repository contains the code for building an "Emojifier" that automatically suggests the most appropriate emoji for a given text message or sentence.

The project explores two different model architectures, **Emojifier-V1 (Baseline)** and **Emojifier-V2 (LSTM-based)**, to classify input sentences into one of five emoji categories. Both models leverage pre-trained word vector representations (GloVe embeddings) to achieve high accuracy even with a small training dataset.

## Project Goal
The core idea is to create a more flexible and intuitive emoji lookup system. Instead of having to type "heart" to find the ❤️ emoji, the Emojifier can generalize, associating related words (like "love," "treasure," or "adore") with the correct emoji.

---

## 1. Emojifier-V1: Baseline Model
This is a simple, non-sequence-aware model used as a baseline.

### Architecture Overview
1.  **Word Embedding Averaging:** Converts each word in the input sentence into its pre-trained **GloVe vector representation** (50-dimensional). It then calculates the **average** of all word vectors in the sentence.
2.  **Softmax Classifier:** The average vector is passed through a **softmax layer** to predict the probability distribution over the 5 emoji classes.

### Key Takeaway
* Achieves reasonable performance with a small dataset due to the generalization power of word vectors.
* **Limitation:** It ignores word order, which means it performs poorly on complex phrases like "not good" or "not enjoyable."

---

## 2. Emojifier-V2: LSTM-based Model
This is a more sophisticated model that accounts for word ordering using a Recurrent Neural Network (RNN).

### Architecture Overview
1.  **Input Preparation:** Input sentences are converted to sequences of word indices and **zero-padded** to a fixed maximum length (`maxLen`) for Keras mini-batching.
2.  **Embedding Layer:** A Keras `Embedding` layer is initialized with the pre-trained **GloVe 50D vectors**. This layer is set to be **non-trainable** to preserve the rich semantic information.
3.  **LSTM Layers:** The word embeddings are fed into a two-layer **LSTM** network (128 units each). The first LSTM uses `return_sequences=True`, and the second returns only the final hidden state.
4.  **Regularization:** **Dropout layers** (with a rate of 0.5) are applied after each LSTM layer to prevent overfitting.
5.  **Output:** A final **Dense layer** with **softmax activation** outputs the probability distribution over the 5 classes.

### Key Takeaway
* By using LSTMs, this model processes the sequence of words, allowing it to correctly interpret phrases where word order is crucial.
* It significantly improves accuracy and robustness over the baseline model.

---

## Setup and Dependencies

To run this project, you will need Python and several packages, including `numpy`, `tensorflow`/`keras`, and `matplotlib`.

### Prerequisites

You need the following data files inside a directory named `data/`:

1.  **GloVe Embeddings:** The script requires the **50-dimensional GloVe vectors** file, specifically named `glove.6B.50d.txt`.
2.  **Dataset:** The training and testing CSV files are required: `train_emoji.csv` and `tesss.csv`.

### Downloading GloVe Embeddings

Due to file size limitations, the GloVe file is **not included** in this repository. You can download the pre-trained word vectors from the Stanford NLP website.

* Download the `glove.6B.zip` file (400k vocabulary, 50d, 100d, 200d, & 300d vectors) from: **[Stanford NLP - GloVe Website](https://nlp.stanford.edu/projects/glove/)**
* After downloading and unzipping, place the **`glove.6B.50d.txt`** file into your local **`data/`** directory.

### Installation

1.  **Install the required libraries:**
    ```bash
    pip install numpy tensorflow matplotlib pandas emoji
    ```
2.  **Ensure all data files** (`train_emoji.csv`, `tesss.csv`, and `glove.6B.50d.txt`) are in a local **`data/`** subdirectory.

### Running the Script

The provided file `Emojify.py` is a Jupyter Notebook converted to a Python script. It is designed to be run cell-by-cell in a Jupyter environment.

1.  **Launch a Jupyter environment:**
    ```bash
    jupyter notebook
    ```
2.  **Run the `Emojify.py` file** (as a notebook) to execute all the steps:
    * Load the data and GloVe embeddings.
    * Train and test Emojifier-V1.
    * Implement, train, and test Emojifier-V2 (the LSTM model).
    * Perform a final custom test.

---

## Utility Functions (from `emo_utils.py`)

The script relies on several utility functions (assumed to be imported from `emo_utils`):

| Function | Description |
| :--- | :--- |
| `read_csv()` | Loads the training and testing datasets. |
| `read_glove_vecs()` | Loads the GloVe word vector map and word-to-index dictionaries. |
| `label_to_emoji(label)` | Converts an integer label (0-4) into its corresponding emoji character. |
| `convert_to_one_hot(Y, C)` | Converts a list of integer labels ($Y$) into an array of one-hot encoded vectors. |
| `predict(...)` | Calculates predictions and accuracy for Emojifier-V1. |
| `print_predictions(...)` | Prints the input sentences alongside their predicted emoji. |
| `plot_confusion_matrix(...)` | Visualizes the classification results. |