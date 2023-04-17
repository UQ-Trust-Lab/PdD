from generator.sampler import *
from generator.distribution import *
from generator.typo_attack import *
from generator.deletion_attack import *
import numpy as np
import csv, os

def test_attack(attack_method, input_string, density, distribution):
    print("Attack Method: " + attack_method)
    print("Original Input: " + input_string)

    if attack_method == 'Deletion':
        sampler = DeletionSampler(density = density, distribution = distribution)
        output, indices = DeletionAttack(input_string, sampler)
    elif attack_method == 'Typo':
        sampler = TypoSampler(density = density, distribution = distribution, diversity = 0.5)
        output, indices = TypoAttack(input_string, sampler)
    elif attack_method == 'Homo':
        pass
    elif attack_method == 'Reorder':
        pass

    modified_output = ''.join(output)
    print("Modified Output: " + modified_output)
    
    comparison = ''.join([input_string[i] if i not in indices else '*' for i in range(len(input_string))])
    valid_indices = [i for i in indices if i >= 0 and i < len(input_string)]
    print("Change: " + str(tuple(valid_indices)) + " -> " + comparison + "\n")

# Tests
input_string1 = "Hello World"
input_string2 = "Attack Example"
density = 0.2

distribution1 = (len(input_string1)/2, len(input_string1)/4)
distribution2 = (len(input_string2)/2, -1)

test_attack('Deletion', input_string1, density, distribution1)
test_attack('Deletion', input_string2, density, distribution2)
test_attack('Typo', input_string1, density, distribution1)
test_attack('Typo', input_string2, density, distribution2)

input_string3 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789 abcdefghijklmnopqrstuvwxyz"
input_string4 = "Test Case Example 12345"
density = 0.1

distribution3 = (len(input_string3)/3, len(input_string3)/5)
distribution4 = (len(input_string4)/4, -1)

test_attack('Deletion', input_string3, density, distribution3)
test_attack('Deletion', input_string4, density, distribution4)
test_attack('Typo', input_string3, density, distribution3)
test_attack('Typo', input_string4, density, distribution4)

