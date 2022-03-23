from typing import cast

import numpy as np

from hjerter.utils import one_hot


def k_mean(data: np.ndarray, k: int, eps: float) -> np.ndarray:
    """cluster data into k clusters using k-mean algorithm

    Args:
        data (np.ndarray): (N, D), N is # of data points,
                         D is the feature dimension
        k (int): # of clusters
        eps (float): the convergence criteria

    Returns:
        np.ndarray: (N, ), the value is the clusters where the data belong to
    """
    # Randomly pick k data points as initial centroids
    centroid_idxs = np.random.choice(data.shape[0], size=(k,))
    centroids = data[centroid_idxs]  # (k, D)
    diff = np.inf

    while diff > eps:
        centroids = np.expand_dims(centroids, axis=0)  # (1, k, D)

        distances: np.ndarray = np.sum(
            (np.expand_dims(data, axis=1) - centroids) ** 2, axis=-1
        )  # (N, k)
        clusters = cast(np.ndarray, np.argmin(distances, axis=-1))  # (N, )
        one_hot_clusters = one_hot(clusters, num_classes=k)  # (N, k)
        new_centroids = (one_hot_clusters.T @ data) / np.expand_dims(
            one_hot_clusters.sum(axis=0), axis=1
        )  # (k, D)

        diff = ((centroids - new_centroids) ** 2).sum()
        centroids = new_centroids

    return clusters


if __name__ == "__main__":
    N = 100
    D = 202
    data = np.random.uniform(size=(N, D))
    clusters = k_mean(data=data, k=7, eps=0.01)
    print(clusters)
