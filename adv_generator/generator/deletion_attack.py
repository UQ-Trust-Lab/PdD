from generator.sampler import *
from generator.distribution import *
import numpy as np
import math

def DeletionAttack(input, sampler=DeletionSampler()):

    strlen = len(input)
    assert strlen > 0, "The input must be a string with its length greater than 0"

    sizeOfDeletion = math.ceil(sampler.density * strlen)
    assert strlen > sizeOfDeletion, "The size of deletion is greater than the original input"

    # We create a list of indices in different ways for different distributions
    indicesToDelete = []
    
    if sampler.distribution.sigma == -1: # uniform distribution
        indicesToDelete = np.random.randint(strlen, size=sizeOfDeletion)
    else:
        count = 0
        while count < sizeOfDeletion:
            index = int(np.random.normal(sampler.distribution.mu, sampler.distribution.sigma))
            if not index in indices:
                indicesToDelete.append(index)
                count += 1
        indicesToDelete.sort()

    # Now we have generated a list of indices to manipulate
    output = ""
    for index, char in enumerate(input):
        if not index in indicesToDelete:
            output += char

    return output, indicesToDelete