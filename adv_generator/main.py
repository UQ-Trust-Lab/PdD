from generator.sampler import *
from generator.distribution import *
from generator.deletion_attack import *
import numpy as np

sampler = Sampler(density=0.2, distribution=(10, -1), diversity=1)
print(sampler.description())

list = np.random.normal(10, 10, size=1)
#list = np.random.randint(25, size=5)
print(list)

input = "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 abcdefghijklmnopqrstuvwxyz"

sampler = DeletionSampler(density = 0.2, distribution = (len(input)/2, len(input)/4))
output, indices = DeletionAttack(input)