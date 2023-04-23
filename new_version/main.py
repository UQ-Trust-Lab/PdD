from collections import defaultdict

from perturbation import *

if __name__ == '__main__':
    distribution = 'uniform'
    density = 0.5
    diversity_dict = defaultdict(lambda: ['']) # space -> space
    diversity_dict[' '] = [' '] # space -> space
    diversity_dict.update(HOMOGLYPHS_DICT)
    assert type(diversity_dict) == defaultdict, 'defaultdict is required.'
    generator = Generator(distribution, density, diversity_dict)

    string_ori = 'I am the best student in the world.'

    string_ptb = generator.generate(string_ori)

    print('Use HOMOGLYPHS_DICT: ', string_ptb)

    diversity_dict = INVISIBLE_UNICODE_DICT.copy()
    diversity_dict.update({' ': [' ']})  # space -> space
    assert type(diversity_dict) == defaultdict, 'defaultdict is required.'
    generator = Generator(distribution, density, diversity_dict, insert_mode=True)

    string_ptb = generator.generate(string_ori)

    print('Use INVISIBLE_UNICODE_DICT: ', string_ptb)

    diversity_dict = DELETION_DICT.copy()
    diversity_dict.update({' ': [' ']})  # space -> space
    assert type(diversity_dict) == defaultdict, 'defaultdict is required.'
    generator = Generator(distribution, density, diversity_dict, insert_mode=False)

    string_ptb = generator.generate(string_ori)

    print('Use DELETION_DICT: ', string_ptb)
