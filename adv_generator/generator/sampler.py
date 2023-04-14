'''
This is the super class of all adversary methods, which defines the basis of 3Ds principle.
It can be initialized with three parameters, i.e., density, distribution, and diversity. 
These three parameters altogether constitute the 3Ds principle of all sorts of our attack.
'''

class Sampler:
    def __init__(self, density = 0.5, distribution = (0, -1), diversity = 0.5):
        self.density = density
        self.distribution = distribution
        self.diversity = diversity
        assert self.density >= 0 and self.density <= 1, "Density must be a float number between 0 and 1."
        assert self.diversity >= 0 and self.diversity <= 1, "Diversity must be a float number between 0 and 1."

    def description(self):
        description = []
        description.append("Density " + str(self.density))
        
        mu, sigma = self.distribution
        if sigma == -1:
            description.append("Uniform Dist.")
        else:
            description.append("Normal Dist. (" + str(mu) + ", " + str(sigma) +")")
        description.append("Diversity " + str(self.diversity))
        
        return description


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
    def __init__(self, density, distribution, diversity):
        super().__init__(density=density, distribution=distribution, diversity=diversity)


class TypoSampler(Sampler):
    # TODO: define and implement a default dictionary of typos for all latin characters and set it as the default value
    def __init__(self, typo = {}):
        super().__init__()
        self.typo = typo   
    