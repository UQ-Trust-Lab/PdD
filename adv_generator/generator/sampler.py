'''
This is the super class of all adversary methods, which defines the basis of 3Ds principle.
It can be initialized with three parameters, i.e., density, distribution, and diversity. 
These three parameters altogether constitute the 3Ds principle of all sorts of our attack.
'''

class Sampler(object):
    def __init__(self, density = 0.05, distribution = (0, -1), diversity = 0.5):
        self.density = density
        self.distribution = distribution
        self.diversity = diversity
        assert self.density >= 0 and self.density <= 1, "Density must be a float number between 0 and 1."
        assert self.diversity >= 0 and self.diversity <= 1, "Diversity must be a float number between 0 and 1."
    
        (mu, sigma) = self.distribution
        self.repr_dict = {
            'density': self.density,
            'distribution': {'mu': mu, 'sigma': sigma},
            'diversity': self.diversity
        }

    def __repr__(self):
        return str(self.repr_dict)

    def __str__(self):
        description = []
        description.append("Density " + str(self.density))
        
        mu, sigma = self.distribution
        if sigma == -1:
            description.append("Uniform Dist.")
        else:
            description.append("Normal Dist. (" + str(mu) + ", " + str(sigma) +")")
        description.append("Diversity " + str(self.diversity))
        
        return str(description)


class InvisibleCharSampler(Sampler):
    # TODO: define and implement a default set of invisible charset and set it as the default value
    def __init__(self, charset = []):
        super().__init__()
        self.charset = charset
    
 
class HomoglyphsSampler(Sampler):
    # TODO: define and implement a default dictionary of homoglyphs for all latin characters and set it as the default value
    def __init__(self, homoglyphs = {}):
        super().__init__()
        self.homoglyphs = homoglyphs   
    

class ReorderingSampler(Sampler):
    # TODO: need to design how reordering attack is implemented. The diveristy of 3Ds in this case is not used.
    def __init__(self):
        super().__init__()   

    
class DeletionSampler(Sampler):
    # TODO: need to design how reordering attack is implemented. The diveristy of 3Ds in this case is not used.
    def __init__(self, density = 0.5, distribution = (0, -1), diversity = 0.5):
        super().__init__(density=density, distribution=distribution, diversity=diversity)
        self.repr_dict['name'] = 'Deletion'

    def __str__(self):
        return "Deletion " + super().__str__()
    
    def __repr__(self):
        return super().__repr__()
        
    
class TypoSampler(Sampler):
    # Aim to define and implement a default dictionary of typos for all latin characters and set it as the default value
    DEFAULT_TYPO_DICT = {
        'a': ['q', 's', 'z'],
        'b': ['v', 'n'],
        'c': ['x', 'v'],
        'd': ['s', 'f', 'e'],
        'e': ['r', 'w', 'd'],
        'f': ['d', 'g', 'r'],
        'g': ['f', 'h', 't'],
        'h': ['g', 'j', 'y'],
        'i': ['u', 'o', 'k'],
        'j': ['h', 'k', 'u'],
        'k': ['j', 'l', 'i'],
        'l': ['k', 'o', 'p'],
        'm': ['n'],
        'n': ['m', 'b'],
        'o': ['i', 'p', 'l'],
        'p': ['o', 'l'],
        'q': ['a', 'w'],
        'r': ['e', 't', 'f'],
        's': ['a', 'd'],
        't': ['r', 'y', 'g'],
        'u': ['y', 'i', 'j'],
        'v': ['c', 'b'],
        'w': ['q', 'e'],
        'x': ['z', 'c'],
        'y': ['t', 'u', 'h'],
        'z': ['x'],
    }


    ADV_TYPO_DICT = {
        'a': ['q', 's', 'w', 'z'],
        'b': ['v', 'n', 'g', 'h'],
        'c': ['x', 'v', 'd', 'f'],
        'd': ['s', 'e', 'f', 'c', 'x', 'r'],
        'e': ['w', 'r', 's', 'd', '3', '4'],
        'f': ['d', 'r', 'g', 'v', 'c', 't'],
        'g': ['f', 't', 'h', 'b', 'v', 'y'],
        'h': ['g', 'y', 'j', 'n', 'b', 'u'],
        'i': ['u', 'o', 'k', 'j', '8', '9'],
        'j': ['k', 'i', 'm', 'u', 'h', 'n'],
        'k': ['j', 'l', 'i', 'o', 'm', ',', '<'],
        'l': ['k', 'o', 'p', ';', ':', ',', '.', '<', '>'],
        'm': ['n', 'j', 'k', ',', '<'],
        'n': ['b', 'm', 'h', 'j'],
        'o': ['i', 'p', 'l', 'k', '9', '0'],
        'p': ['o', 'l', '0', '[', '{', ':', ';'],
        'q': ['a', 'w', '1', '2'],
        'r': ['e', 't', 'f', 'd', '4', '5'],
        's': ['a', 'd', 'w', 'x', 'e', 'z'],
        't': ['r', 'y', 'g', 'f', '5', '6'],
        'u': ['y', 'i', 'j', 'h', '7', '8'],
        'v': ['c', 'b', 'f', 'g'],
        'w': ['q', 'e', 's', 'a', '2', '3'],
        'x': ['z', 'c', 's', 'd'],
        'y': ['u', 't', 'h', 'g', '6', '7'],
        'z': ['x', 'a', 's'],

        '1': ['q', 'a', '2'],
        '2': ['1', 'q', 'w', '3'],
        '3': ['2', 'w', 'e', '4'],
        '4': ['3', 'e', 'r', '5'],
        '5': ['4', 'r', 't', '6'],
        '6': ['5', 't', 'y', '7'],
        '7': ['6', 'y', 'u', '8'],
        '8': ['7', 'u', 'i', '9'],
        '9': ['8', 'i', 'o', '0'],
        '0': ['9', 'o', 'p'],

        '!': ['@'],
        '@': ['!', '#'],
        '#': ['@', '$'],
        '$': ['#', '%'],
        '%': ['$', '^'],
        '^': ['%', '&'],
        '&': ['^', '*'],
        '*': ['&', '('],
        '(': ['*', ')'],
        ')': ['(', '-'],
        '-': [')', '_'],
        '_': ['-', '='],
        '=': ['_', '+'],
        '+': ['='],

        '`': ['~'],
        '~': ['`'],
        '{': ['['],
        '[': ['{', ']'],
        ']': ['[', '}'],
        '}': [']'],
        '|': ['\\'],
        '\\': ['|'],
        ':': [';', "'"],
        ';': [':', '"'],
        "'": [';', '"'],
        '"': ["'", '<'],
        '<': ['"', '>'],
        '>': ['<', '?'],
        '?': ['>'],

        ',': ['<', '.'],
        '.': [',', '>'],
        '/': ['?', '.'],
        '<': [',', '.'],
        '>': ['.', '/'],
        '?': ['/', ' '],
        ' ': ['?', '<', '>'],

    }


    def __init__(self, density, distribution, diversity, typo=None):
        super().__init__(density=density, distribution=distribution, diversity=diversity)
        if typo is None:
            if diversity >= 0.5:
                # ADV_TYPO_DICT contains more advanced typo mappings.
                typo = TypoSampler.ADV_TYPO_DICT
            else:
                # DEFAULT_TYPO_DICT contains basic typo mappings, 
                typo = TypoSampler.DEFAULT_TYPO_DICT
        self.typo = typo
        self.repr_dict['name'] = 'Typo'
        self.repr_dict['typo'] = self.typo
        
    def __str__(self):
        return "Typo " + super().__str__()
    
    def __repr__(self):
        return super().__repr__()
