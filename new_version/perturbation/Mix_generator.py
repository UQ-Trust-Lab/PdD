from typing import List, Tuple, Dict, Set
from .distribution import *
from .sampler import Sampler
from new_version.perturbation import DELETION_DICT, HOMOGLYPHS_DICT, INVISIBLE_UNICODE_DICT, KEYBOARD_TYPO_ADVANCED_DICT
import random
from collections import defaultdict



class MixGenerator:
    def __init__(self, distribution: str, density: float, diversity_dict: Dict[str, List[str]] = None,
                 insert_mode: bool = False,
                 **kwargs):
        """
        This class is used to generate a perturbed string.
        :param distribution: The distribution of perturbations.
        :param density: The density of perturbations, e.g., 0.1 means 10% of the indexes are perturbed.
        :param diversity_dict: The diversity dictionary, which describes the candidate perturbed characters for each
        index.
        :param insert_mode: The insert mode, which is used to insert perturbations into the original string.
        :param kwargs: The keyword arguments, which are used to set the distribution of perturbations, e.g., mean and
        std.
        """
        assert 0. <= density <= 1., 'The density should be in the range of [0, 1].'
        assert distribution in ["uniform", "normal"], 'The distribution should be "uniform" or "normal".'

        self.density = density
        self.diversity_dict = diversity_dict
        self.distribution = distribution
        self.insert_mode = insert_mode
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

        PERTURBATION_DICT = {
            "deletion": DELETION_DICT,
            "typo": KEYBOARD_TYPO_ADVANCED_DICT,
            "homoglyphs": HOMOGLYPHS_DICT,
            "invisible": INVISIBLE_UNICODE_DICT
        }

        diversity_dict_typo = defaultdict(lambda: [' '])
        diversity_dict_homo = defaultdict(lambda: [' '])
        diversity_dict_invisible = defaultdict(lambda: [' '])

        diversity_dict_deletion = PERTURBATION_DICT["deletion"]
        diversity_dict_typo.update(PERTURBATION_DICT["typo"])
        diversity_dict_homo.update(PERTURBATION_DICT["homoglyphs"])
        diversity_dict_invisible.update(PERTURBATION_DICT["invisible"])

        diversity_dict_list = [diversity_dict_deletion, diversity_dict_typo, diversity_dict_homo,
                               diversity_dict_invisible]

        # Copy the original string
        string_ptb = list(string_ori).copy()
        for idx in idxes_ptb:
            char_ptb = string_ptb[idx]  # Get the original character
            self.diversity_dict = diversity_dict_list[random.randint(0, 3)]
            random_idx = np.random.randint(0, len(self.diversity_dict[char_ptb]))  # Choose a random perturbed character
            while self.diversity_dict[char_ptb][random_idx] == char_ptb:  # Avoid choosing the same character
                if len(self.diversity_dict[char_ptb]) == 1:
                    break
                random_idx = np.random.randint(0, len(self.diversity_dict[char_ptb]))
            if not self.insert_mode:
                string_ptb[idx] = ''  # Delete the original character
            string_ptb[idx] += self.diversity_dict[char_ptb][random_idx]  # Perturb the character

        string_ptb = "".join(string_ptb)
        return string_ptb
