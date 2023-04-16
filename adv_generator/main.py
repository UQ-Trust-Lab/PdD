from generator.sampler import *
from generator.distribution import *
from generator.typo_attack import *
from generator.deletion_attack import *
import numpy as np
import csv, os


input = "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 abcdefghijklmnopqrstuvwxyz"

sampler = DeletionSampler(density = 0.05, distribution = (len(input)/2, len(input)/4))
output, indices = DeletionAttack(input, sampler)
print(sampler)
print(repr(sampler))
print(output, indices)

sampler = DeletionSampler(density = 0.05, distribution = (len(input)/2, -1))
output, indices = DeletionAttack(input, sampler)
print(sampler)
print(output, indices)

sampler = TypoSampler(density = 0.05, distribution = (len(input)/2, -1), diversity=0.5)
output, indices = TypoAttack(input, sampler)
print(sampler)
print(repr(sampler))
print(output, indices)

sampler = TypoSampler(density = 0.05, distribution = (len(input)/2, len(input)/4), diversity=0.5)
output, indices = TypoAttack(input, sampler)
print(sampler)
print(output, indices)