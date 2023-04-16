from generator.sampler import TypoSampler
from generator.distribution import *
import numpy as np
import math

def TypoAttack(input, sampler=TypoSampler(density = 0.05, distribution = (0, -1), diversity = 0.5)):
    """
    Generates typos in the input string based on the specified density and distribution in the provided sampler.

    Args:
        input (str): The input string to generate typos in.
        sampler (TypoSampler): The sampler object to determine the density and distribution of typos. 
            Defaults to TypoSampler().

    Note:
        Here I use `diversity` to indicate the range of TYPO. i.e. how many keys around the selected key.
        For smaller diversity, 'a' -> ['q', 's']; for bigger diversity 'a' -> ['q', 's', 'w', 'z'].

    """

    strlen = len(input)
    assert strlen > 0, "The input must be a string with its length greater than 0"

    sizeOfTypo = math.ceil(sampler.density * strlen)
    assert strlen > sizeOfTypo, "The size of modification is greater than the original input"

    # Create a list of indices in different ways for different distributions
    indicesToModify = []

    (mu, sigma) = sampler.distribution
    if sigma == -1:  # uniform distribution
        indicesToModify = np.random.randint(strlen, size=sizeOfTypo)
    else:
        indicesToModify = []
        while len(indicesToModify) < sizeOfTypo:
            index = int(np.random.normal(mu, sigma))
            if index not in indicesToModify and 0 <= index < strlen:
                indicesToModify.append(index)


    inputList = list(input)  # "Oh hi" -> ['O','h',' ','h','i']
    for index in indicesToModify:
        original_char = inputList[index]
        lowercase_char = original_char.lower()
        if lowercase_char in sampler.typo.keys():
            typo_char = np.random.choice(sampler.typo[lowercase_char])
            if original_char.isupper():  # Handle upper cases
                typo_char = typo_char.upper()
            inputList[index] = typo_char

    output = "".join(inputList)

    return output, indicesToModify
