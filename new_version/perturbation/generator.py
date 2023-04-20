from typing import List, Tuple, Dict, Set

from .distribution import *
from .sampler import Sampler


class Generator:
    def __init__(self, distribution: str, density: float, diversity_dict: Dict[str, List[str]]=None, **kwargs):
        """
        This class is used to generate a perturbed string.
        :param distribution: The distribution of perturbations.
        :param density: The density of perturbations, e.g., 0.1 means 10% of the indexes are perturbed.
        :param diversity_dict: The diversity dictionary, which describes the candidate perturbed characters for each
        index.
        :param kwargs: The keyword arguments, which are used to set the distribution of perturbations, e.g., mean and
        std.
        """
        assert 0. <= density <= 1., 'The density should be in the range of [0, 1].'
        assert distribution in ["uniform", "normal"], 'The distribution should be "uniform" or "normal".'

        self.density = density
        self.diversity_dict = diversity_dict
        self.distribution = distribution
        self.kwargs = kwargs

    def generate(self, string_ori: str) -> str:
        """
        This function is used to generate a perturbed string.
        :param string_ori: The original string.
        :return: The perturbed string.
        """
        n_chars = len(string_ori)

        # Set the distribution of perturbations
        if self.distribution == "uniform":
            self.distribution = RandomDistribution(n_chars)
        elif self.distribution == "normal":
            mean = self.kwargs.get("mean", n_chars / 2)
            std = self.kwargs.get("std", n_chars / 4)
            self.distribution = NormalDistribution(n_chars, mean, std)

        # Set the sampler
        self.sampler = Sampler(n_chars, self.distribution, self.density)

        # Sample the indexes of perturbations
        idxes_ptb = self.sampler.sample()


        # Copy the original string
        string_ptb = list(string_ori).copy()
        for idx in idxes_ptb:
            char_ptb = string_ptb[idx]  # Get the original character

            random_idx = np.random.randint(0, len(self.diversity_dict[char_ptb]))  # Choose a random perturbed character
            string_ptb[idx] = self.diversity_dict[char_ptb][random_idx]  # Perturb the character

        string_ptb = "".join(string_ptb)
        return string_ptb
