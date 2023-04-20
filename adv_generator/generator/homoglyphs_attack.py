from generator.sampler import HomoglyphsSampler
from generator.distribution import *
import numpy as np
import math

def HomoglyphsAttack(input, sampler=HomoglyphsSampler(density = 0.05, distribution = (0, -1), diversity = 0.5)):
    """
    Generates homoglyphs in the input string based on the specified density and distribution in the provided sampler.

    Args:
        input (str): The input string to generate homoglyphs in.
        sampler (TypoSampler): The sampler object to determine the density and distribution of homoglyphs. 
            Defaults to HomoglyphsSampler().

    Note:
        Diversity is used to control the range of the possible homoglyphs.


    """

    strlen = len(input)
    assert strlen > 0, "The input must be a string with its length greater than 0"

    sizeOfHomoglyphs = math.ceil(sampler.density * strlen)
    assert strlen > sizeOfHomoglyphs, "The size of modification is greater than the original input"

    # Create a list of indices in different ways for different distributions
    indicesToModify = []

    (mu, sigma) = sampler.distribution
    if sigma == -1:  # uniform distribution
        indicesToModify = np.random.randint(strlen, size=sizeOfHomoglyphs)
    else:
        count = 0
        while count < sizeOfHomoglyphs:
            index = int(np.random.normal(mu, sigma))
            if index not in indicesToModify:
                indicesToModify.append(index)
                count += 1
    indicesToModify.sort()



    inputList = list(input)  # "Oh hi" -> ['O','h',' ','h','i']
    for index in indicesToModify:
        original_char = inputList[index]
        lowercase_char = original_char.lower()
        if lowercase_char in sampler.homoglyphs.keys():
            homos = sampler.homoglyphs[lowercase_char]
            if sampler.diversity == 1:
                homos_char = np.random.choice(sampler.homoglyphs[lowercase_char])
            else:
                # Diversity controls the range of the possible homoglyphs.
                num_homoglyphs_use = int(sampler.diversity * len(homos))
                homoglyphs_char_range = np.random.choice(homos, size=num_homoglyphs_use).tolist()
                homos_char = np.random.choice(homoglyphs_char_range)
            
            if original_char.isupper() and homos_char.isalpha():  # Handle upper cases
                homos_char = homos_char.upper()
            inputList[index] = homos_char

    output = "".join(inputList)

    return output, indicesToModify
