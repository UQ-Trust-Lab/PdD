__all__ = ["SPACE_DICT"]

from collections import defaultdict
from types import MappingProxyType

# Create a dictionary with a default value of a list of space and make it immutable
SPACE_DICT = MappingProxyType(defaultdict(lambda: [" "]))
