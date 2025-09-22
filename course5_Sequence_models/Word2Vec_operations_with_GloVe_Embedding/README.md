# **Word Vector Operations**

This project explores the use of pre-trained word embeddings to understand and manipulate word relationships. It demonstrates key concepts in natural language processing (NLP), including measuring word similarity, solving analogies, and addressing algorithmic bias.

***

### Project Overview

The core of this project is the use of **pre-trained word embeddings**, which are vector representations of words that capture their semantic and syntactic properties. Training these embeddings is computationally expensive, so this notebook uses a pre-trained set of **GloVe (Global Vectors for Word Representation)** embeddings.

The main tasks covered are:

* **Loading Word Vectors**: Using a pre-trained `glove.6B.50d.txt` file to load 50-dimensional GloVe vectors.
* **Cosine Similarity**: Implementing a function to calculate the similarity between word vectors. This is a crucial metric for quantifying how related two words are in the vector space.
* **Word Analogy Task**: Solving analogy problems (e.g., "man is to woman as king is to \_\_\_\_\_\_") by leveraging the linear relationships between word vectors.
* **Debiasing Word Vectors**: Exploring a method to reduce **gender bias** embedded within the word vectors. This involves two techniques:
    * **Neutralization**: Removing gender-specific components from words that should be gender-neutral (e.g., "receptionist").
    * **Equalization**: Adjusting gender-specific word pairs (e.g., "actor" and "actress") to be equally distant from other words.

***

### Prerequisites

This project relies on the **`glove.6B.50d.txt`** word embedding file, which is not included due to its size. You can download this specific file from the following sources:

* **Stanford NLP GloVe Project**: [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
* **Hugging Face**: [https://huggingface.co/datasets/glove_wiki_gigaword](https://huggingface.co/datasets/glove_wiki_gigaword)

After downloading, place the file in a `data` folder within your project directory.

***

### References

The debiasing algorithm and GloVe word embeddings are based on the following research:

* **The debiasing algorithm is from Bolukbasi et al., 2016, [Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)**
* **The GloVe word embeddings were due to Jeffrey Pennington, Richard Socher, and Christopher D. Manning, [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)**