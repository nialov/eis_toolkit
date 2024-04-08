from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from beartype import beartype

from eis_toolkit.exceptions import InvalidDataShapeException


@beartype
def plot_confusion_matrix(confusion_matrix: np.ndarray, cmap: Optional[str] = None) -> plt.Axes:
    """Plot confusion matrix to visualize classification results.

    Args:
        confusion_matrix: The confusion matrix as 2D Numpy array. Expects the first element
            (upper-left corner) to have True negatives.
        cmap: Colormap name to be used in the plot. Optional parameter.

    Returns:
        Matplotlib axes containing the plot.

    Raises:
        InvalidDataShapeException: Raised if input confusion matrix is not square.
    """
    shape = confusion_matrix.shape
    if shape[0] != shape[1]:
        raise InvalidDataShapeException(f"Expected confusion matrix to be square, found shape: {shape}")
    names = None

    counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
    percentages = ["{0:.2%}".format(value) for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]

    if shape == (2, 2):  # Binary classificaiton
        names = ["True Neg", "False Pos", "False Neg", "True Pos"]
        labels = np.asarray([f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(names, counts, percentages)]).reshape(shape)
    else:
        labels = np.asarray([f"{v1}\n{v2}" for v1, v2 in zip(counts, percentages)]).reshape(shape)

    ax = sns.heatmap(confusion_matrix, annot=labels, fmt="", cmap=cmap)
    ax.set(xlabel="Predicted label", ylabel="True label")

    return ax
