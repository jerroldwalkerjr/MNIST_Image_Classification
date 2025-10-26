# MNIST_Image_Classification
This project implements image classification for the MNIST handwritten digit dataset, comparing classical machine learning techniques (K-Nearest Neighbors, Naive Bayes, Linear Regression) with modern deep learning approaches (Linear Model, Multilayer Perceptron, Convolutional Neural Network).

For a detailed explanation of this project’s goals, implementation, and results, see "MNIST_Image_Classification\report\MNIST Image Classification Report_Jerrold Walker.pdf"

## Project Structure
data/
    (contains digit subfolders 0–9)
report/
    MNIST Image Classification Report_Jerrold Walker.pdf (Full project report)
src/
    data_utils.py (Loads and preprocesses MNIST images)
    eval_utils.py (Provides visualization tools)
    knn_numpy.py (K-Nearest Neighbors classifier using NumPy)
    linear_numpy.py (Linear classifier with gradient descent on one-hot labels)
    main.py (Runs all experiments — loads data, trains models, and prints results)
    models_pytorch.py (Defines PyTorch versions of linear, MLP, and CNN models for comparison)
    naive_bayes_numpy.py (Bernoulli Naive Bayes classifier for binarized image data)
    train_pytorch.py (Training and evaluation loops for PyTorch models.
linear_weights.png (Visualization of learned weights from the NumPy linear classifier)

## Technologies Used
- Python 3.13
- NumPy (numerical computing and linear algebra for NumPy-based models)
- PyTorch (deep learning library for MLP and CNN models)
- Torchvision (dataset handling and transforms for PyTorch)
- PIL / Pillow (image loading and preprocessing)
- Matplotlib (visualizations for weight maps and confusion matrices)

## How to Run
1. Download/clone the repository
2. Install dependencies: pip install numpy pillow matplotlib torch torchvision scikit-learn
3. Run the main script: "MNIST_Image_Classification\src\main.py"
The script will run all models (KNN, Naive Bayes, Linear NumPy, PyTorch Linear/MLP/CNN).
Outputs include validation accuracies and a saved weight visualization (linear_weights.png).
4. Optional: For dataset subsmpling toggle "USE_SUBSAMPLE = False" in line 14 of "main.py" to True
