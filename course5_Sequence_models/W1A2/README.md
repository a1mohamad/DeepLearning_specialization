# 🦖 Character-Level Language Model: Dinosaurus Island

---

## **Introduction**
Welcome to Dinosaurus Island! This project uses a **character-level recurrent neural network (RNN)** to generate new, realistic dinosaur names. Instead of generating words, the model learns the patterns of individual characters from a given dataset of existing dinosaur names (`dinos.txt`). This approach allows the model to create novel names that follow the same linguistic structure and spelling conventions as the training data.

The project is designed as a hands-on exercise to teach key concepts in deep learning, specifically related to RNNs. By completing the code, you'll learn to:
* Preprocess text data for an RNN.
* Build a character-level text generation model.
* Implement **gradient clipping** to prevent exploding gradients.
* Use a **sampling technique** to generate new sequences.
* Train the model using stochastic gradient descent.

---

## **Model Overview**
The model's architecture is a simple RNN that processes one character at a time. At each time step, it takes the current character as input and the hidden state from the previous time step. It then predicts the probability distribution of the next character in the sequence. 

The core components of the model are:
1.  **Preprocessing**: The `dinos.txt` dataset is loaded and preprocessed. Each unique character, including the newline `\n` character, is assigned a unique index. The newline character acts as an end-of-name token.
2.  **Building Blocks**:
    * **Clipping the Gradients**: To prevent the model from becoming unstable during training due to "**exploding gradients**," a `clip` function is implemented to ensure all gradients stay within a specific range `[-N, N]`.
    * **Sampling**: Once the model is trained, the `sample` function is used to generate new names. It starts with a zero vector and iteratively samples the next character based on the model's output probability distribution. Using `np.random.choice` ensures variety in the generated names.
3.  **Training Loop**: The `model` function orchestrates the training process using **stochastic gradient descent**. It iterates through the dataset, performs forward and backward propagation, clips the gradients, and updates the model's parameters.

---

## **Files**
* `Dinosaurus_Island_Character_level_language_model.ipynb`: The main Jupyter Notebook containing the code and exercises.
* `dinos.txt`: The dataset of dinosaur names used for training.
* `utils.py`: A helper file containing utility functions like `rnn_forward`, `rnn_backward`, and `update_parameters`.

---

## **Conclusion**
This project successfully demonstrates the power of a character-level language model. As the model trains, its output evolves from random characters to plausible dinosaur names with realistic endings like `saurus`, `don`, or `tor`. The implementation provides a solid foundation for more complex text generation tasks.

---

## **Beyond Dinosaur Names: Generating Shakespearean Poetry**
The final, ungraded part of this notebook extends the same principles to a more complex and creative task: generating poetry in the style of Shakespeare. While the dinosaur name generator uses a basic RNN cell and a small dataset, the Shakespeare generator leverages a more powerful architecture:
* **LSTMs (Long Short-Term Memory)**: These specialized RNN cells are used to capture **longer-term dependencies** across many characters, which are crucial for generating coherent and structured poetry.
* **Keras**: The model is built using the Keras deep learning library, which simplifies the implementation of a deeper, stacked LSTM model.

By training on a collection of Shakespearean sonnets, this part of the notebook shows how you can use these techniques to create text that mimics a specific style and tone. After training for one epoch, you can input a starting phrase, and the model will complete the poem for you.