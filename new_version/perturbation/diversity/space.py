from types import MappingProxyType
from collections import defaultdict

# Create a dictionary with a default value of a list of space and make it immutable
SPACE_DICT = MappingProxyType(defaultdict(lambda: [' ']))
