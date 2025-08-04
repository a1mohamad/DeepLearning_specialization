# Deep Neural Network for Image Classification: Cats vs. Non-Cats

This project builds a deep L-layer neural network from scratch to classify images as cat or non-cat. 
It is implemented using only NumPy, without deep learning frameworks.
And also get help from functions that already implemented in last assignment, `Deep Learning step by step`

## 1. Project Structure
1. **Packages**  
   Import essential libraries:
   - `numpy`, `matplotlib`, `h5py`, `PIL`, `scipy`
   - Helper functions from `dnn_app_utils_v3.py`

2. **Load and Process the Dataset**  
   - Load the "Cats vs. Non-Cats" dataset (`data.h5`)
   - Flatten images to vectors of shape `(12288, m)`  
   - Standardize pixel values to `[0, 1]`

3. **Model Architectures**
   - **2-layer Neural Network**: `LINEAR -> RELU -> LINEAR -> SIGMOID`
   - **L-layer Deep Neural Network**: `[LINEAR -> RELU] × (L-1) -> LINEAR -> SIGMOID`

4. **Two-layer Model**  
   - Implement `two_layer_model()`:
     - Initialize parameters
     - Forward propagation
     - Compute cost
     - Backward propagation
     - Update parameters
   - Train using gradient descent.

5. **L-layer Model**  
   - Implement `L_layer_model()`:
     - Initialize deep parameters
     - Forward and backward propagation across multiple layers
     - Update parameters
   - Train using gradient descent.

6. **Evaluation**  
   - Use `predict()` to test on training and test sets.
   - Optionally test on a custom image placed in `images/`.

7. **Analysis (Optional)**  
   - Visualize mislabeled images to inspect common error patterns.

## 2. How to Run
1. Place dataset (`data.h5`) and helper file (`dnn_app_utils_v3.py`) in the project directory.
2. Open the notebook and run cells in order:
   - Load and preprocess dataset
   - Train the 2-layer model
   - Train the L-layer model
   - Evaluate and test with your own image (optional)
3. To test a custom image:
   ```python
   my_image = "my_pic.jpg" # Replace with your uploaded image
   my_label_y = [1]  # 1 = cat, 0 = non-cat

## 3. Files
- dnn_app_utils_v3.py – helper functions

- data.h5 – datasets/

- deep_neural_network.ipynb – main notebook

- and some test files for checking results



Let's have fun with Cats ! Meow!!!