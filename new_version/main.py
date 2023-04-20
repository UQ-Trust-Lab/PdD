from perturbation import *


if __name__ == '__main__':
    distribution = 'uniform'
    density = 0.1
    diversity_dict = {' ': [' ']} # Do not perturb the space character
    diversity_dict.update(LETTER_LOWER2UPPER_DICT)
    generator = Generator(distribution, density, diversity_dict)

    string_ori = 'I am the best student in the world.'

    string_ptb = generator.generate(string_ori)

    print(string_ptb)