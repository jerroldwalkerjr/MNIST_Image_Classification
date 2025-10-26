# src/linear_numpy.py
import numpy as np

def one_hot(y, C):
    out = np.zeros((y.size, C))
    out[np.arange(y.size), y] = 1.0
    return out

def train_linear_numpy(X_train, y_train, X_val, y_val, epochs=100, lr=0.1, reg=1e-4):
    # Train a simple linear classifier using Mean Squared Error (MSE) loss.
    
    N, D = X_train.shape
    C = len(np.unique(y_train))
    W = np.zeros((C, D))  # linear W (C, D) so y = W x
    Y = one_hot(y_train, C)
    for ep in range(epochs):
        scores = W.dot(X_train.T).T  # (N, C)
        # MSE loss to one-hot
        loss = ((scores - Y)**2).mean() + 0.5*reg*(W**2).sum()
        grad = (2.0/N) * (scores - Y).T.dot(X_train) + reg*W  # (C, D)
        W -= lr * grad
        if ep % (epochs//10 + 1) == 0:
            preds = np.argmax(W.dot(X_val.T), axis=0)
            acc = (preds == y_val).mean()
            print(f"ep {ep} loss {loss:.4f} val_acc {acc:.4f}")
    return W