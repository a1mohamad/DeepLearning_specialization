# Transfer Learning with MobileNetV2

This Jupyter Notebook demonstrates how to build and train an image classifier using **transfer learning** with a pre-trained convolutional neural network (CNN), **MobileNetV2**.

Transfer learning is a highly effective technique that leverages a model's existing knowledge, gained from training on a large dataset like **ImageNet**, to efficiently solve a new, related task. MobileNetV2, an architecture optimized for computational efficiency, serves as the foundation for this project. By utilizing its pre-trained weights, we can achieve high-performance image classification without extensive training data or computational resources.

***

## Objectives

Upon completing this notebook, you will be proficient in:

* **Dataset Creation:** Generating a dataset directly from image files organized in a directory structure.
* **Data Augmentation:** Applying preprocessing and augmentation techniques to enhance training data and improve model robustness.
* **Model Adaptation:** Utilizing the Keras Functional API to adapt the pre-trained MobileNetV2 model for new classification tasks.
* **Model Fine-Tuning:** Implementing fine-tuning of a classifier's final layers to optimize accuracy.

***

## The Transfer Learning Process

The core of this notebook's methodology involves a two-phase training process to build the Alpaca/Not Alpaca classifier.

### Phase 1: Feature Extraction

The initial phase uses the pre-trained MobileNetV2 as a fixed feature extractor. Its early convolutional layers, which have already learned to identify fundamental features such as edges and textures, are **frozen**, meaning their weights are not updated during training. A new, trainable classifier is added on top of these layers. This approach allows the model to quickly learn the specific high-level features required to distinguish between the two classes.



### Phase 2: Fine-Tuning

In the second phase, a few of the top convolutional layers of the pre-trained model are **unfrozen**. The entire model is then trained with a very low learning rate. This step, known as fine-tuning, enables the model to make small adjustments to the weights of the pre-trained layers, making them more specific to the features of the new dataset and potentially leading to a significant increase in performance.

***

## Getting Started

1.  Clone this repository or download the notebook file.
2.  Download the dataset from this Kaggle link: [https://www.kaggle.com/datasets/sid4sal/alpaca-dataset-small](https://www.kaggle.com/datasets/sid4sal/alpaca-dataset-small).
3.  Organize your downloaded images in a `dataset/` directory with two sub-folders: `alpaca` and `not_alpaca`.
4.  Open the notebook in a Jupyter environment and execute the cells sequentially to train the classifier.