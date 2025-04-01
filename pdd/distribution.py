__all__ = ["Distribution", "RandomDistribution", "NormalDistribution"]

from abc import ABC, abstractmethod

import numpy as np


class Distribution(ABC):
    """
    This abstract class is used to define the distribution of perturbations.
    It is used to set the distribution of perturbations.
    """

    def __init__(self, max_idx: int):
        assert max_idx > 0, "The maximum index should be greater than 0."
        self.max_idx = max_idx

    @abstractmethod
    def generate(self) -> int:
        """
        This function is used to generate an index that will be perturbed.
        :return: The index of perturbations.
        """
        pass


class RandomDistribution(Distribution):
    """
    This class is used to define the random distribution of perturbations.
    It is used to set the distribution of perturbations.
    """

    def __init__(self, max_idx: int):
        super().__init__(max_idx)

    def generate(self) -> int:
        return int(np.random.randint(0, self.max_idx, 1))


class NormalDistribution(Distribution):
    """
    This class is used to define the normal distribution of perturbations.
    It is used to set the distribution of perturbations.
    """

    def __init__(self, max_idx: int, mean: float = None, std: float = None):
        super().__init__(max_idx)
        self.mean = mean
        self.std = std

    def generate(self):
        idx = -1
        while idx < 0 or idx >= self.max_idx:
            idx = int(np.random.normal(self.mean, self.std, 1))

        return idx
