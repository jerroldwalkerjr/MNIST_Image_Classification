import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot(cmap='Blues')
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def show_weights(W, filename="linear_weights.png"):
    """
    Visualize and save the learned weights for each class in a 2x5 grid.
    Each weight vector is reshaped to 28x28 to represent an MNIST digit.
    """
    # Normalize weights for better contrast
    W_min, W_max = np.min(W), np.max(W)
    W_norm = (W - W_min) / (W_max - W_min)

    fig, axes = plt.subplots(2, 5, figsize=(8, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(W_norm[i, :].reshape(28, 28), cmap="gray")
        ax.set_title(str(i))
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(filename)  # save image to project folder
    plt.close()
    print(f"✅ Saved weight visualization to '{filename}'")