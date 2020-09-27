from abc import ABC, abstractmethod

import numpy as np


class Cluster(ABC):
    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
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

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        pass
