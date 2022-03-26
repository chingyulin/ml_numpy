from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from hjerter.algorithms.k_mean import k_mean


def plot(
    data: np.ndarray,
    dim0: int = 0,
    dim1: int = 1,
    clusters: Optional[np.ndarray] = None,
) -> None:
    """Plot two dimensions of data in `data`, optionally colored with `clusters`

    Args:
        data (np.ndarray): (N, D), N is # of data points,
                         D is the feature dimension
        dim0 (int, optional): dim0-th column of data to be plot on x-axis. Defaults to 0.
        dim1 (int, optional): dim1-th column of data to be plot on y-axis. Defaults to 1.
        clusters (Optional[np.ndarray], optional): (N, 1), the value is the clusters
                        where the data belong to. Defaults to None.
    """
    fig, ax = plt.subplots()
    ax.scatter(x=data[:, dim0], y=data[:, dim1], c=clusters)
    plt.show()


if __name__ == "__main__":
    N = 10000
    D = 2
    data = np.random.normal(10, size=(N, D))
    clusters = k_mean(data=data, k=100, eps=0.01)
    plot(data, clusters=clusters)
