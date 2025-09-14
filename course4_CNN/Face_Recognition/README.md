# Face Recognition with Triplet Loss

## 1. Project Overview

This notebook provides a hands-on, step-by-step guide to building a functional face recognition system. The project is based on the powerful concepts of deep metric learning and uses a pre-trained neural network inspired by the groundbreaking [FaceNet](https://arxiv.org/pdf/1503.03832.pdf) algorithm. By leveraging this pre-trained model, the project focuses on the core principles of using learned embeddings for practical applications without the extensive computational cost of training from scratch.

By completing this notebook, you will develop a deep understanding of the differences between face verification and face recognition, master the concept of one-shot learning, and apply a specialized loss function to a real-world problem. The final system allows you to build a database of known faces and verify or recognize new individuals with high accuracy.

## 2. Key Concepts

### Face Verification vs. Face Recognition

It's critical to understand the distinction between these two tasks, as they represent different problems in facial analysis:

* **Face Verification**: This is a **1:1 matching problem**. The system is given a face and an identity and must verify if the face belongs to that specific person. A common example is using facial ID to unlock a smartphone. The system compares the live image to a single stored reference image. The question is "Is this person who they claim to be?"

* **Face Recognition**: This is a **1:K matching problem**. The system is given a face and must identify who it is from a database of K possible individuals. This is a more challenging task, as it requires the system to distinguish between all people in the database. An example would be an office building security system that recognizes employees as they enter. The question is "Who is this person?"

### One-Shot Learning

Traditional classification models require many images of a single person to learn to identify them. This is impractical for a large-scale face recognition system. **One-shot learning** provides a powerful solution by allowing the model to recognize a person from a **single example image**. The key is that the model doesn't classify faces directly; instead, it learns a general-purpose function that encodes a face image into a compact, numerical vector.

### Face Encodings (Embeddings)

The heart of this system is a neural network that acts as an encoder. It takes an image of a face and outputs a 128-dimensional vector, known as a **face encoding** or **embedding**. These embeddings are designed so that the L2 distance between them reflects the similarity of the faces:

* The distance between two encodings of the **same person** is **small**.

* The distance between two encodings of **different people** is **large**.

By comparing the L2 distance between the embedding of a new face and the embeddings stored in a database, the system can determine identity.

### The Triplet Loss

The model's ability to create meaningful embeddings is achieved by minimizing the **Triplet Loss** function during training. This loss function operates on triplets of images, each consisting of:

* An **Anchor (A)**: An image of a specific person.

* A **Positive (P)**: A different image of the *same* person as the Anchor.

* A **Negative (N)**: An image of a *different* person.

The objective is to ensure that the distance between the Anchor and the Positive is significantly smaller than the distance between the Anchor and the Negative. The margin, $$\alpha$$, is a crucial hyperparameter that enforces this separation. The loss function, as shown below, only penalizes the model when this condition is violated.

$$\mathcal{J} = \sum^{m}_{i=1} [ || f(A^{(i)}) - f(P^{(i)}) ||_2^2 - || f(A^{(i)}) - f(N^{(i)}) ||_2^2 + \alpha ]_+$$

## 3. Implementation Details

This notebook uses a pre-trained model with an Inception-like architecture. You will be guided to implement the following key functions, which are critical for the system's operation:

* **`triplet_loss()`**: This function calculates the triplet loss. While the model is pre-trained, implementing this function is essential for understanding the underlying training mechanism.

* **`verify()`**: This function performs the **face verification** task. It computes the encoding of an input image, retrieves the encoding of a known identity from the database, and calculates the L2 distance. If the distance is below a predefined threshold, it confirms a match.

* **`who_is_it()`**: This function performs the more complex **face recognition** task. It calculates the encoding of an input image and then iterates through every encoding in the database to find the one with the smallest L2 distance. If the minimum distance is below the threshold, it correctly identifies the person.

## 4. How to Use

The notebook is pre-configured with a small database of 12 individuals. The `img_to_encoding()` function is provided to handle image preprocessing and encoding generation. You can test your `verify()` and `who_is_it()` functions with the provided sample images to see the system in action.

## 5. References

1. Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)

2. Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf)

3. This implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet

4. Further inspiration was found here: https://machinelearningmastery.com/how-to-develop-a-face-recognition-system-using-facenet-in-keras-and-an-svm-classifier/

5. And here: https://github.com/nyoki-mtl/keras-facenet/blob/master/notebook/tf_to_keras.ipynb