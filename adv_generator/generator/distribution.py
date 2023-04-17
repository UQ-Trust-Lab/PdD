"""
This class defines distribution, which is one key component of 3Ds principle.
It is defined as a normal distribution, defined with two parameters, mu and sigma.
The parameter 'mu' is the mean or expectation of the distribution, and the parameter 'sigma' is its standard deviation.
We recommend 'mu' to be set as the mid of the string so as it equals 1/2(length_of_input_string). The 'sigma' can be set to a hugh value or -1 if a uniform distribution is expected.
"""


class Distribution:
    def __init__(self, mu = 0, sigma = -1):
        self.mu = mu
        self.sigma = sigma
        assert self.sigma >= 0 or self.sigma == -1, "Sigma must be a float number greater than 0, otherwise -1 (uniform distribution)."


