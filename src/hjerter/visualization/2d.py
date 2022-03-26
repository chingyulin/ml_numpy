import matplotlib.pyplot as plt
import numpy as np

from hjerter.algorithms.k_mean import k_mean


def plot(data: np.ndarray, clusters: np.ndarray) -> None:
    fig, ax = plt.subplots()
    ax.scatter(x=data[:, 0], y=data[:, 1], c=clusters)
    plt.show()


if __name__ == "__main__":
    N = 10000
    D = 2
    data = np.random.normal(10, size=(N, D))
    clusters = k_mean(data=data, k=100, eps=0.01)
    plot(data, clusters)
