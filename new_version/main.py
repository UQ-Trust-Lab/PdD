from collections import defaultdict

from perturbation import *

if __name__ == '__main__':
    distribution = 'uniform'
    density = 0.5
    diversity_dict = {' ': [' ']}  # space -> space
    diversity_dict.update(HOMOGLYPHS_DICT)
    generator = Generator(distribution, density, diversity_dict)

    string_ori = 'I am the best student in the world.'

    string_ptb = generator.generate(string_ori)

    print('Use HOMOGLYPHS_DICT: ', string_ptb)

    diversity_dict = INVISIBLE_UNICODE_DICT.copy()  # space -> space
    diversity_dict.update({' ': [' ']})
    generator = Generator(distribution, density, diversity_dict, insert_mode=True)

    string_ptb = generator.generate(string_ori)

    print('Use INVISIBLE_UNICODE_DICT: ', string_ptb)
