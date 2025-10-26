# src/naive_bayes_numpy.py
import numpy as np

class BernoulliNaiveBayes:
    # implementation of the Bernoulli Naive Bayes classifier using NumPy.

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        # X: (N, D) with values in [0,1] -> binarize at 0.5
        Xb = (X > 0.5).astype(np.int64)
        N, D = Xb.shape
        classes = np.unique(y)
        self.classes_ = classes
        C = len(classes)
        self.class_log_prior_ = np.zeros(C)
        self.feature_log_prob_ = np.zeros((C, D))  # log P(feature=1 | class)
        for i, c in enumerate(classes):
            idx = (y == c)
            nc = idx.sum()
            self.class_log_prior_[i] = np.log(nc / N)
            # counts of 1s per feature
            count1 = Xb[idx].sum(axis=0)
            # Laplace smoothing
            prob1 = (count1 + self.alpha) / (nc + 2*self.alpha)
            self.feature_log_prob_[i] = np.log(prob1)
            self.feature_log_prob_ = self.feature_log_prob_  # keep shape
        # Also keep log prob of feature=0: log(1-prob1) computed on the fly

    def predict(self, X):
        Xb = (X > 0.5).astype(np.int64)
        C, D = self.feature_log_prob_.shape
        N = Xb.shape[0]
        loglike = np.zeros((N, C))
        for i in range(C):
            logp1 = self.feature_log_prob_[i]
            logp0 = np.log(1 - np.exp(logp1))  # careful: log(1 - p), but we stored log p, so exp(logp1) = p
            # safer compute p = exp(logp1) then log(1-p)
            p1 = np.exp(logp1)
            logp0 = np.log(1 - p1 + 1e-12)
            # sum over features: if Xb==1 use logp1 else use logp0
            loglike[:, i] = (Xb * logp1 + (1 - Xb) * logp0).sum(axis=1) + self.class_log_prior_[i]
        idx = np.argmax(loglike, axis=1)
        return self.classes_[idx]