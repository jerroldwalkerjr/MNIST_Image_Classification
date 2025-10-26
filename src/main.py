# main.py
import numpy as np
import torch

# import functions & classes
from data_utils import load_images_numpy, make_flattened_splits, get_pytorch_dataloaders
from knn_numpy import predict_knn, accuracy as acc_knn
from naive_bayes_numpy import BernoulliNaiveBayes
from linear_numpy import train_linear_numpy
from models_pytorch import LinearModel, MLP, SimpleCNN
from train_pytorch import train_model, eval_model
from eval_utils import plot_confusion, show_weights

USE_SUBSAMPLE = False # subsampling toggle

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # Uses GPU if available, otherwise CPU

DATA_ROOT = "data"      # MNIST folders
TRAIN_FRAC = 0.8        # fraction of data for training (rest used for validation)
BATCH_SIZE = 64         # batch size for Pytorch

def main():
    # load data for NumPy models
    print("Loading MNIST images (NumPy)...")
    X, y = load_images_numpy(DATA_ROOT)
    X_train, y_train, X_val, y_val = make_flattened_splits(X, y, train_frac=TRAIN_FRAC)
    print(f"Train samples: {len(X_train)}, Validation samples: {len(X_val)}\n")

    # optinal subsampling for faster testing
    SUBSAMPLE_TRAIN = 1000 # number of training samples
    SUBSAMPLE_VAL = 200 # number of validation samples
    if USE_SUBSAMPLE:
        X_train = X_train[:SUBSAMPLE_TRAIN]
        y_train = y_train[:SUBSAMPLE_TRAIN]
        X_val = X_val[:SUBSAMPLE_VAL]
        y_val = y_val[:SUBSAMPLE_VAL]

    print(f"Subsampled Train: {len(X_train)}, Subsampled Validation: {len(X_val)}")

    # ---------------- KNN ----------------
    # Run K-Nearest Neighbor classifier for k = 1, 3, 5
    print("=== K-Nearest Neighbors ===")
    for k in [1, 3, 5]:
        preds = predict_knn(X_train, y_train, X_val, k=k) # predict labels
        acc = acc_knn(y_val, preds) #compute accuracy
        print(f"k={k} Accuracy: {acc:.4f}")

    # ---------------- Naive Bayes ----------------
    print("\n=== Naive Bayes ===")
    nb_model = BernoulliNaiveBayes(alpha=1.0) # laplace smoothing
    nb_model.fit(X_train, y_train) # estimate probailities from training data
    preds_nb = nb_model.predict(X_val) # predict validation labels
    acc_nb = acc_knn(y_val, preds_nb) # compute accuracy
    print(f"Naive Bayes Accuracy: {acc_nb:.4f}")

    # ---------------- Linear Classifier (NumPy) ----------------
    print("\n=== Linear Classifier (NumPy) ===")
    # train linear classifier with gradient descent and L2 regularization
    W = train_linear_numpy(X_train, y_train, X_val, y_val, epochs=50, lr=0.001, reg=1e-4)
    show_weights(W)  # optional visualization

    # ---------------- PyTorch models (MLP & CNN) ----------------
    print("\n=== PyTorch Models ===")
    # prepare PyTorch dataloaders for training/validation
    train_loader, val_loader, _ = get_pytorch_dataloaders(DATA_ROOT, batch_size=BATCH_SIZE, train_frac=TRAIN_FRAC)

    # Linear (PyTorch version, optional)
    print("\n--- Linear Model ---")
    lin_model = LinearModel() # simple linear model
    train_model(lin_model, train_loader, val_loader, device=DEVICE, epochs=10, lr=0.01)

    # MLP
    print("\n--- Multilayer Perceptron (MLP) ---")
    mlp_model = MLP() # at least one hidden layer with ReLU
    train_model(mlp_model, train_loader, val_loader, device=DEVICE, epochs=15, lr=0.001)

    # CNN
    print("\n--- Convolutional Neural Network (CNN) ---")
    cnn_model = SimpleCNN() # two conv layers & pooling
    train_model(cnn_model, train_loader, val_loader, device=DEVICE, epochs=10, lr=0.001)

if __name__ == "__main__":
    main()