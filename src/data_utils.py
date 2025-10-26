# src/data_utils.py
import os
import random
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split

def load_images_numpy(root_dir, img_size=(28,28)):
    X, y = [], []
    # loops through subdirectories
    for label in sorted(os.listdir(root_dir)):
        lab_dir = os.path.join(root_dir, label)
        if not os.path.isdir(lab_dir): continue
        
        # loops through image files
        for fname in os.listdir(lab_dir):
            path = os.path.join(lab_dir, fname)
            try:
                img = Image.open(path).convert('L').resize(img_size)
                arr = np.array(img, dtype=np.float32) / 255.0  # normalize [0,1]
                X.append(arr)
                y.append(int(label))
            except Exception as e:
                print("skipping", path, e)
    X = np.stack(X)  # (N, H, W)
    y = np.array(y, dtype=np.int64)
    return X, y

# Splits datasets between training and validation
def make_flattened_splits(X, y, train_frac=0.8, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    n_train = int(len(X)*train_frac)
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    X_train = X[train_idx].reshape(len(train_idx), -1)
    X_val = X[val_idx].reshape(len(val_idx), -1)
    y_train, y_val = y[train_idx], y[val_idx]
    return X_train, y_train, X_val, y_val

# PyTorch ImageFolder loader 
def get_pytorch_dataloaders(root_dir, batch_size=64, train_frac=0.8, seed=42):
    transform = transforms.Compose([
        transforms.Grayscale(), transforms.Resize((28,28)),
        transforms.ToTensor(), # will be in [0,1]
        transforms.Normalize((0.5,), (0.5,)) # [-1,1]
    ])
    dataset = datasets.ImageFolder(root_dir, transform=transform)
    n_train = int(len(dataset) * train_frac)
    n_val = len(dataset) - n_train
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    # DataLoaders
    from torch.utils.data import DataLoader
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size), dataset.classes