### README: Neural Machine Translation with Attention

---

## Neural Machine Translation with Attention (Date Translation)

This repository contains a Python script (`Neural_machine_translation_with_attention_v4a.py`) that implements a **Neural Machine Translation (NMT)** model using a **sequence-to-sequence architecture with an attention mechanism**.

The model is trained to solve a simplified translation task: converting **human-readable dates** into a standardized **machine-readable format** (YYYY-MM-DD).

### Project Goal

The primary goal is to train a model to translate a wide variety of date formats (e.g., `"25th of June, 2009"`, `"03/30/1968"`) into the consistent output format (`"2009-06-25"`, `"1968-03-30"`).

### Key Components

* **Architecture:** The model uses a standard **Encoder-Decoder** structure built with **LSTMs** from TensorFlow/Keras.
* **Encoder:** A **Bidirectional LSTM** is used to process the input sequence.
* **Attention Mechanism:** A **one-step attention** function calculates a context vector for each output time step, allowing the decoder to focus on the most relevant parts of the input date when generating the output characters.
* **Decoder:** An LSTM-based model generates the output sequence character-by-character.

### Dataset

* **Size:** The model is trained on a synthetic dataset of **10,000** (human-readable date, machine-readable date) pairs.
* **Sequence Lengths:**
    * Maximum input length ($T_x$): **30**.
    * Output length ($T_y$): **10** (for the format YYYY-MM-DD).
* **Visualization:** The script includes functionality to **plot the attention map** to visualize which input characters the model is focusing on at each step of the output generation.

### Requirements

This script is written in Python and requires the following libraries:

* **`tensorflow`** (and **`tensorflow.keras`** for the model layers)
* **`numpy`**
* **`matplotlib`**
* **`tqdm`**
* **`faker`** (used in the `nmt_utils.py` file to generate the dataset)
* **`babel.dates`** (used in the `nmt_utils.py` file for date formatting)
* **`nmt_utils`**: The script imports helper functions from a module named `nmt_utils` (e.g., `load_dataset`, `preprocess_data`, `plot_attention_map`) which is required to run the code.