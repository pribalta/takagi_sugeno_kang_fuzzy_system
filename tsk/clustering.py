from typing import Any, Tuple

import numpy as np


def normalize(u: np.ndarray) -> np.ndarray:
    """
    Normalize a vector u with respect to itself
    :param u: (np.ndarray) Input vector
    :return: Normalized vector
    """
    return u / np.sum(u, axis=0, keepdims=True)


def distance(u: np.ndarray, v: np.ndarray) -> float:
    """
    Calculate the euclidean distance between two vectors u and v
    :param u: (np.ndarray) First input vector
    :param v: (np.ndarray) Second input vector
    :return: (np.ndarray) Euclidean distance
    """
    v = np.expand_dims(np.transpose(v), 0)
    u = np.expand_dims(u, 2)

    return np.sqrt(np.sum((u - v) ** 2, axis=1))


class CMeansClustering:
    def __init__(self, n_clusters: int, error: float = 1e-4, max_iters: int = 200):
        """
        Constructor of C-Means Clustering
        :param n_clusters: (int) Number of clusters
        :param error: (float) Stop condition
        :param max_iters: (int) Total iteration number
        """
        self._n_clusters = n_clusters
        self._error = error
        self._max_iters = max_iters

        self._fuzzy_index = None
        self._fitted = None
        self._center = None
        self._train_u = None
        self._variance = None
        self._losses = None
        self._dimensions = None

    @property
    def center(self) -> np.ndarray:
        assert self._fitted, 'You need to fit the clustering algorithm first!'

        return self._center

    @property
    def delta(self) -> np.ndarray:
        assert self._fitted, 'You need to fit the clustering algorithm first!'

        return self._variance

    def fit(self, x: np.ndarray, y: np.ndarray = None) -> Any:
        """
        Fit a set of measurements to the model
        :param x: (np.ndarray) Vector with inputs
        :param y: (np.array) Vector with measurements
        :return: (CMeansClustering) self
        """
        if min(x.shape[0], x.shape[1] - 1) >= 3:
            self._fuzzy_index = min(x.shape[0], x.shape[1] - 1) / (min(x.shape[0], x.shape[1] - 1) - 2)
        else:
            self._fuzzy_index = 2

        n = x.shape[0]
        self._dimensions = x.shape[1]

        u = np.random.rand(self._n_clusters, n)

        self._losses = []

        for t in range(self._max_iters):
            u, v, loss, signal = self._update(x, u)
            self._losses.append(loss)
            print('\titer: {} - loss: {:.4f}'.format(t, loss))
            if signal:
                break

        self._fitted = True
        self._center = v
        self._train_u = u
        self._variance = np.zeros(self._center.shape)

        for i in range(self._dimensions):
            self._variance[:, i] = np.sum(
                u * ((x[:, i][:, np.newaxis] - self._center[:, i].transpose()) ** 2).T, axis=1
            ) / np.sum(u, axis=1)

        self._variance = np.fmax(self._variance, np.finfo(np.float64).eps)

        return self

    def _update(self, x: np.ndarray, u: np.ndarray) -> Tuple:
        """
        Update the internal state of the model
        :param x: (np.array) Input sample
        :param u: (np.array) Input vector
        :return: (Tuple) Vectors u,v and loss and signal
        """
        old_u = u.copy()
        old_u = np.fmax(old_u, np.finfo(np.float64).eps)
        old_u_prev = old_u.copy()

        old_u = normalize(old_u) ** self._fuzzy_index

        v = np.dot(old_u, x) / old_u.sum(axis=1, keepdims=True)

        dist = distance(x, v).T
        dist = np.fmax(dist, np.finfo(np.float64).eps)

        loss = (old_u * dist ** 2).sum()
        dist = dist ** (2 / (1 - self._fuzzy_index))
        dist = np.fmax(dist, np.finfo(np.float64).eps)

        u = normalize(dist)
        if np.linalg.norm(u - old_u_prev) < self._error:
            signal = True
        else:
            signal = False

        return u, v, loss, signal
