__all__ = ["INVISIBLE_UNICODE_CHARS", "INVISIBLE_UNICODE_DICT"]

from collections import defaultdict
from types import MappingProxyType

INVISIBLE_UNICODE_CHARS = [
    "\u115f",  # hangul chosung filler
    "\u1160",  # hangul jungsung filler
    "\u180e",  # mongolian vowel separator
    "\u2000",  # en quad
    "\u2001",  # em quad
    "\u2002",  # en space
    "\u2003",  # em space
    "\u2004",  # three-per-em space
    "\u2005",  # four-per-em space
    "\u2006",  # six-per-em space
    "\u2007",  # figure space
    "\u2008",  # punctuation space
    "\u2009",  # thin space
    "\u200a",  # hair space
    "\u200b",  # zero width space
    "\u200c",  # zero width non-joiner
    "\u200d",  # zero width joiner
    "\u2028",  # line separator
    "\u2029",  # paragraph separator
    "\u202a",  # left-to-right embedding
    "\u202b",  # right-to-left embedding
    "\u202c",  # pop directional formatting
    "\u202d",  # left-to-right override
    "\u202e",  # right-to-left override
    "\u202f",  # narrow no-break space
    "\u205f",  # medium mathematical space
    "\u2060",  # word joiner
    "\u2061",  # function application
    "\u2062",  # invisible times
    "\u2063",  # invisible separator
    "\u2064",  # invisible plus
    "\u2066",  # invisible ltr
    "\u2067",  # invisible rtl
    "\u2068",  # invisible pop direction
    "\u2069",  # invisible separator
    "\u206a",  # invisible narrow no-break space
    "\u206b",  # invisible ideographic space
    "\u206c",  # invisible mathematical operator
    "\u206d",  # invisible opening parenthesis
    "\u206e",  # invisible closing parenthesis
    "\u206f",  # invisible times
    "\u3000",  # ideographic space
    "\u3164",  # hangul filler
    "\ufeff",  # zero width no-break space
    "\ufffc",  # object replacement character
]


INVISIBLE_UNICODE_DICT = MappingProxyType(defaultdict(lambda: INVISIBLE_UNICODE_CHARS))
