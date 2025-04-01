__all__ = ["Sampler"]

import math

from .distribution import Distribution


class Sampler:
    def __init__(self, max_idx: int, distribution: Distribution, density: float):
        """
        This class is used to sample the indexes of perturbations.
        :param max_idx: The maximum index of perturbations.
        :param distribution: The distribution of perturbations.
        :param density: The density of perturbations, e.g., 0.1 means 10% of the indexes
            are perturbed.
        """
        assert max_idx > 0, "The maximum index should be greater than 0."
        assert 0.0 <= density <= 1.0, "The density should be in the range of [0, 1]."

        self.max_idx = max_idx
        self.distribution = distribution
        self.density = density

    def sample(self) -> list[int]:
        """
        This function is used to sample the indexes of perturbations.
        :return: The indexes of perturbations.
        """
        # No perturbation
        if self.density == 0.0:
            return []

        # All indexes are perturbed
        if self.density == 1.0:
            return list(range(self.max_idx))

        # Choose the unique indexes of perturbations by using the distribution
        n_idxes = int(math.floor(self.max_idx * self.density))
        idxes = []
        while len(idxes) < n_idxes:
            idx = self.distribution.generate()
            if idx not in idxes:
                idxes.append(idx)

        return idxes
