### Improvise a Jazz Solo with an LSTM Network

This project uses a Long Short-Term Memory (LSTM) recurrent neural network to generate original jazz music. The model learns the intricate patterns and structures of jazz solos from a provided corpus, allowing it to compose new, unique musical sequences. 🎷

***

### Project Overview

This deep learning project tackles the creative task of music generation by treating music as a sequential problem. The core idea is to train a neural network to predict the next musical "value"—which can be a single note or a chord—based on a sequence of preceding values. The model essentially learns the "grammar" and style of jazz music from a training dataset, then uses that knowledge to improvise.

The system is implemented in two distinct but interconnected parts:

#### Training Model

The first part of the project focuses on training an LSTM network to learn the underlying patterns of jazz.
* **Data Preparation**: The training data consists of snippets of jazz music, where each snippet is a sequence of 30 musical values. These values are represented as **one-hot encoded vectors**, with 90 unique values in total.
* **Model Architecture**: The training model uses a **many-to-many architecture**. It takes a sequence of 30 values as input and, for each time step, predicts the next value in the sequence. This is achieved by:
    * Defining a shared **LSTM cell** to capture long-term dependencies in the music.
    * Passing the hidden state of the LSTM at each time step through a shared **Dense layer** with a `softmax` activation. This layer outputs a probability distribution over the 90 possible musical values.
    * The model is compiled using the **Adam optimizer** and a **categorical cross-entropy loss** function, which is ideal for this type of multi-class classification problem.
* **Training**: The model is trained for 100 epochs, allowing its weights and biases to be fine-tuned to recognize and replicate the stylistic features of the jazz corpus.

***

#### Inference Model

After the training is complete, a separate **inference model** is built to generate new music. This model reuses the learned weights from the training phase.
* **Generative Process**: The inference model is a **one-to-many sequence generator**. It starts with an initial, zero-valued input vector. It then performs the following steps in a loop for 50 time steps:
    1.  The current input vector, along with the previous hidden and cell states, is fed into the **shared LSTM cell**.
    2.  The output hidden state from the LSTM is passed through the **shared Dense layer**.
    3.  The model's prediction—the note with the highest probability—is selected.
    4.  This predicted note is then converted back into a one-hot encoded vector and used as the input for the **next time step**.
* **Music Generation**: This iterative process creates a sequence of 50 new musical values. These values are then post-processed into a standard MIDI file, which can be played. For simplicity, the output is provided as an MP3 file, making it easily accessible for listening.

***

### Final Product

The culmination of this project is a generated jazz solo, a MIDI file that has been converted to a **`.mp3`** for easy listening. It demonstrates the remarkable ability of deep learning models to learn from creative data and produce new, original content. While the provided code includes a simple WAV conversion, the primary output for sharing is the MP3 file, which is more universally playable. The process of generating and listening to the music is a rewarding experience, showcasing the practical and artistic applications of recurrent neural networks. 

***

### References

This project draws upon significant ideas and inspiration from the following works, most notably from Ji-Sung Kim's `deepjazz` repository.

* Ji-Sung Kim, 2016, [deepjazz](https://github.com/jisungk/deepjazz)
* Jon Gillick, Kevin Tang and Robert Keller, 2009, [Learning Jazz Grammars](http://ai.stanford.edu/~kdtang/papers/smc09-jazzgrammar.pdf)
* Robert Keller and David Morrison, 2007, [A Grammatical Approach to Automatic Improvisation](http://smc07.uoa.gr/SMC07%20Proceedings/SMC07%20Paper%2055.pdf)
* François Pachet, 1999, [Surprising Harmonies](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.5.7473&rep=rep1&type=pdf)