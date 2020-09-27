from typing import Any

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

from tsk.clustering import CMeansClustering


def compute_firing_level(data: np.ndarray, centers: int, delta: float) -> np.ndarray:
    """
    Compute firing strength using Gaussian model

    :param data: n_Samples * n_Features
    :param centers: data center，n_Clusters * n_Features
    :param delta: variance of each feature， n_Clusters * n_Features
    :return: firing strength
    """
    d = -(np.expand_dims(data, axis=2) - np.expand_dims(centers.T, axis=0)) ** 2 / (2 * delta.T)
    d = np.exp(np.sum(d, axis=1))
    d = np.fmax(d, np.finfo(np.float64).eps)
    return d / np.sum(d, axis=1, keepdims=True)


def apply_firing_level(x: np.ndarray, firing_levels: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Convert raw input to tsk input, based on the provided firing levels

    :param x: (np.ndarray) Raw input
    :param firing_levels: (np.ndarray) Firing level for each rule
    :param order: (int) TSK order. Valid values are 0 and 1
    :return:
    """
    if order == 0:
        return firing_levels
    else:
        n = x.shape[0]
        firing_levels = np.expand_dims(firing_levels, axis=1)
        x = np.expand_dims(np.concatenate((x, np.ones([n, 1])), axis=1), axis=2)
        x = np.repeat(x, repeats=firing_levels.shape[1], axis=2)
        output = x * firing_levels
        output = output.reshape([n, -1])

        return output


class Classifier:
    def __init__(self, c: float = 1., regressor_iters: int = 200, n_cluster: int = 2, order: int = 1):
        """
        Fuzzy classifier class

        :param c: (float) c-coefficient for linear regressor estimator
        :param regressor_iters: (int) max iters for logistic regression fitting
        :param n_cluster: (int) Number of clusters
        :param order: (int) Order of the method. Valid values are 0 or 1
        """
        self._c = c
        self._regressor_iters = regressor_iters
        self._n_cluster = n_cluster
        self._order = order

        self._n_classes = None
        self._center = None
        self._variance = None
        self._experts = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> Any:
        """
        Fit a set of measurements to the model
        :param x: (np.ndarray) Vector with inputs
        :param y: (np.array) Vector with measurements
        :return: (Classifier) self
        """
        self._n_classes = len(np.unique(y))

        cluster = CMeansClustering(self._n_cluster).fit(x, y)

        self._center = cluster.center
        self._variance = cluster.delta

        mu_a = compute_firing_level(x, self._center, self._variance)
        computed_input = apply_firing_level(x, mu_a, self._order)

        self._experts = LogisticRegression(C=self._c, max_iter=self._regressor_iters)
        self._experts.fit(computed_input, y)

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict to which class a given set of inputs belongs

        :param x: (np.ndarray) Input vector
        :return: (np.ndarray) Output vector with predicted classes
        """
        firing_levels = compute_firing_level(x, self._center, self._variance)

        computed_input = apply_firing_level(x, firing_levels, self._order)

        logits = self._experts.decision_function(computed_input)

        return np.argmax(softmax(logits, axis=1), axis=1)
