__all__ = ["DELETION_DICT"]

from collections import defaultdict
from types import MappingProxyType

# Create a dictionary with a default value of a list of space
DELETION_DICT = MappingProxyType(defaultdict(lambda: [""]))
