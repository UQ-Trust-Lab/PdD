from generator.sampler import *
from generator.distribution import *
from generator.typo_attack import *
from generator.homoglyphs_attack import *
from generator.deletion_attack import *
import numpy as np
import csv
import os

def test_attack(attack_method, input_string, density, distribution):
    if attack_method == 'Deletion':
        sampler = DeletionSampler(density=density, distribution=distribution)
        output, indices = DeletionAttack(input_string, sampler)
    elif attack_method == 'Typo':
        sampler = TypoSampler(density=density, distribution=distribution, diversity=0.5)
        output, indices = TypoAttack(input_string, sampler)
    elif attack_method == 'Homo':
        sampler = HomoglyphsSampler(density=density, distribution=distribution, diversity=0.5)
        output, indices = HomoglyphsAttack(input_string, sampler)
    else:
        raise ValueError("Invalid attack method: " + attack_method)

    modified_output = ''.join(output)
    return modified_output

input_file = "dataset.txt"
output_file = "new_dataset.txt"
density = 0.2

with open(input_file, "r") as file:
    sentences = file.readlines()

distribution = (len(sentences)/2, len(sentences)/4)


# Apply attacks to each sentence and write to output file
with open(output_file, "w") as file:
    for sentence in sentences:
        sentence = sentence.strip()
        original_sentence = sentence
        file.write(original_sentence + "\n")

        # Deletion attack
        modified_sentence = test_attack('Deletion', sentence, density, distribution)
        file.write(modified_sentence + "\n")

        # Typo attack
        modified_sentence = test_attack('Typo', sentence, density, distribution)
        file.write(modified_sentence + "\n")

        # Homoglyphs attack
        modified_sentence = test_attack('Homo', sentence, density, distribution)
        file.write(original_sentence + "\n")
        file.write(modified_sentence + "\n\n")
