# 👂 Trigger Word Detection: The "Activate" Model

This repository contains a deep learning project for **Trigger Word Detection** (also known as keyword or wake word detection). The goal is to build a model capable of listening to a stream of audio and triggering a chime whenever it detects the specific wake word: **"activate"**.

This technology is foundational to smart devices like Amazon Alexa and Google Home, allowing them to wake up upon hearing their designated trigger word.

***

## 🚀 Overview and Features

The core of this project is the construction of a custom speech recognition model and its supporting dataset.

### Key Components:

* **Trigger Word:** **`activate`**
* **Model Architecture:** A Recurrent Neural Network (RNN), likely a **GRU (Gated Recurrent Unit)**, trained to detect when the trigger word has finished being spoken.
* **Audio Preprocessing:** Raw audio is converted into **Spectrograms** ($T_x = 5511$ timesteps) to extract frequency information, which is a suitable input for the model.
* **Data Synthesis Pipeline:** A synthetic approach is used to create a robust dataset by randomly overlaying positive and negative word clips onto background noise.
* **Target Output:** A "chiming" sound is played when the word is detected.

***

## 📂 Project Structure and Prerequisites

### Files

* `Trigger_word_detection.ipynb`: The main Jupyter Notebook containing the data synthesis, model definition, training, and testing steps.
* `/raw_data/`: Directory containing raw audio snippets for data synthesis (e.g., `/activates/`, `/negatives/`, `/backgrounds/`).
* `td_utils.py`: A helper file containing utility functions for loading audio, handling files, and spectrogram generation.

### Installation

The project relies on standard Python scientific and audio processing libraries.

```bash
# Required libraries (you may need to install system dependencies for pydub)
pip install numpy pydub matplotlib IPython scipy
```

***
### ⚠️ Important Notes

**Audio Library Dependency:**
The `pydub` library often requires **`ffmpeg`** to be installed on your operating system for full audio file support (like reading `.wav` files).

***

## 💾 Dataset

Due to file size limitations, the main training and development dataset is not included in this repository. Only the raw audio files used for synthesis are provided in the `/raw_data/` folder.

You can download the full dataset required for training the model from the following link:

**[Trigger Word Detection Dataset](https://www.kaggle.com/datasets/umermjd11/triggerworddetection)**

***

## ▶️ Usage

The project is designed to be run cell-by-cell within the **`Trigger_word_detection.ipynb`** notebook.

### Workflow:
1.  **Download Data:** Download and extract the full dataset from the link above and ensure the training files are accessible by the notebook.
2.  **Data Preparation:** Run the cells to load raw audio, understand the spectrogram representation, and execute the helper functions to create the synthesized dataset.
3.  **Training:** Define and train the GRU-based model.

### Testing with Custom Audio

To test the trained model with a new 10-second audio clip (e.g., `my_audio.wav`):

1.  **Set Filename:** Update the `your_filename` variable in the notebook:

    ```python
    your_filename = "audio_examples/my_audio.wav"
    ```
2.  **Run Prediction:** The final cells will preprocess the audio, call `detect_triggerword()`, and use `chime_on_activate()` to generate an output file (**`chime_output.wav`**) where a chime is inserted immediately following each detection.

### ⚙️ Adjusting Sensitivity

The sensitivity of the detection can be adjusted by modifying the **`chime_threshold`** variable in the last section of the notebook (default is `0.5`):

```python
chime_threshold = 0.5  # Lower value (e.g., 0.1) = More sensitive; Higher value (e.g., 0.9) = Less sensitive
```