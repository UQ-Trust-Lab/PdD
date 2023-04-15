from generator.sampler import TypoSampler
from generator.distribution import *
import numpy as np
import math

def TypoAttack(input, sampler=TypoSampler()):
    """
    Generates typos in the input string based on the specified density and distribution in the provided sampler.

    Args:
        input (str): The input string to generate typos in.
        sampler (TypoSampler): The sampler object to determine the density and distribution of typos. 
            Defaults to TypoSampler().

    TODO: The attribute `diversity` is not involved at this stage. 
          Maybe It can be used to indicate the range of TYPO. i.e. how many keys around the selected key.
          For smaller diversity, 'a' -> ['q', 's']; for bigger diversity 'a' -> ['q', 's', 'w', 'z'].

    """

    strlen = len(input)
    assert strlen > 0, "The input must be a string with its length greater than 0"

    sizeOfTypo = math.ceil(sampler.density * strlen)
    assert strlen > sizeOfTypo, "The size of deletion is greater than the original input"

    indicesToModify = []

    if sampler.distribution.sigma == -1:
        indicesToModify = np.random.randint(strlen, size=sizeOfTypo)
    else:
        mu, sigma = sampler.distribution.mu, sampler.distribution.sigma
        indicesToModify = []
        while len(indicesToModify) < sizeOfTypo:
            index = int(np.random.normal(mu, sigma))
            if index not in indicesToModify and 0 <= index < strlen:
                indicesToModify.append(index)


    inputList = list(input)
    for index in indicesToModify:
        original_char = inputList[index]
        lowercase_char = original_char.lower() 
        typo_char = np.random.choice(sampler.typo[lowercase_char])
        if original_char.isupper():
            typo_char = typo_char.upper()
        inputList[index] = typo_char

    output = "".join(inputList)

    return output, indicesToModify
