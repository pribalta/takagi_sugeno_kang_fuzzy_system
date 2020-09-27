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
