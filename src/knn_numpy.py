# src/knn_numpy.py
import numpy as np
from collections import Counter

def predict_knn(X_train, y_train, X_test, k=3, batch_size=200, max_test=None):
    """
    Parameters:
        X_train: (N_train, D) training data
        y_train: (N_train,) training labels
        X_test: (N_test, D) test data
        k: number of neighbors
        batch_size: number of test images processed at once
        max_test: if set, only use first max_test test samples
    """
    # used for subset testing
    if max_test is not None:
        X_test = X_test[:max_test]
    
    Ntest = X_test.shape[0]
    preds = np.zeros(Ntest, dtype=int) #predictions placeholder
    
    # batch testing
    for i in range(0, Ntest, batch_size):
        end = min(Ntest, i+batch_size)
        xt = X_test[i:end]
        print(f"Processing images {i} to {end-1}...")

        xt_sq = (xt**2).sum(axis=1)[:, None]
        tr_sq = (X_train**2).sum(axis=1)[None, :]
        cross = xt @ X_train.T
        d2 = xt_sq + tr_sq - 2*cross

        nearest_idx = np.argsort(d2, axis=1)[:, :k]
        for j, inds in enumerate(nearest_idx):
            votes = y_train[inds]
            from collections import Counter
            c = Counter(votes)
            # for ties choose smallest label
            preds[i+j] = min([lab for lab,count in c.items() if count==max(c.values())])
    
    return preds


def accuracy(y_true, y_pred):
    """
    Parameters:
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.

    Returns:
        float : Proportion of correctly classified samples.
    """
    return (y_true == y_pred).mean()