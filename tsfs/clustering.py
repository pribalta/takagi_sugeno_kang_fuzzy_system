from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Cluster(ABC):
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> Any:
        """
        Fit input to distribution
        :param x: (np.ndarray) Input values
        :param y: (np.ndarray) Evaluation result
        :return: None
        """
        return NotImplementedError


class CMeansClustering(Cluster):
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

    def fit(self, x: np.ndarray, y: np.ndarray) -> Any:
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
            print('[FCM Iter {}] Loss: {:.4f}'.format(t, loss))
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
