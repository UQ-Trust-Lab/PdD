from collections import defaultdict
from types import MappingProxyType
from typing import List

INVISIBLE_UNICODE_CHARS = [
    u'\u115f',  # hangul chosung filler

    u'\u1160',  # hangul jungsung filler

    u'\u180e',  # mongolian vowel separator

    u'\u2000',  # en quad
    u'\u2001',  # em quad
    u'\u2002',  # en space
    u'\u2003',  # em space
    u'\u2004',  # three-per-em space
    u'\u2005',  # four-per-em space
    u'\u2006',  # six-per-em space
    u'\u2007',  # figure space
    u'\u2008',  # punctuation space
    u'\u2009',  # thin space
    u'\u200a',  # hair space
    u'\u200b',  # zero width space
    u'\u200c',  # zero width non-joiner
    u'\u200d',  # zero width joiner

    u'\u2028',  # line separator
    u'\u2029',  # paragraph separator
    u'\u202a',  # left-to-right embedding
    u'\u202b',  # right-to-left embedding
    u'\u202c',  # pop directional formatting
    u'\u202d',  # left-to-right override
    u'\u202e',  # right-to-left override
    u'\u202f',  # narrow no-break space

    u'\u205f',  # medium mathematical space

    u'\u2060',  # word joiner
    u'\u2061',  # function application
    u'\u2062',  # invisible times
    u'\u2063',  # invisible separator
    u'\u2064',  # invisible plus
    u'\u2066',  # invisible ltr
    u'\u2067',  # invisible rtl
    u'\u2068',  # invisible pop direction
    u'\u2069',  # invisible separator
    u'\u206a',  # invisible narrow no-break space
    u'\u206b',  # invisible ideographic space
    u'\u206c',  # invisible mathematical operator
    u'\u206d',  # invisible opening parenthesis
    u'\u206e',  # invisible closing parenthesis
    u'\u206f',  # invisible times

    u'\u3000',  # ideographic space

    u'\u3164',  # hangul filler

    u'\ufeff',  # zero width no-break space

    u'\ufffc',  # object replacement character

]


INVISIBLE_UNICODE_DICT = MappingProxyType(defaultdict(lambda: INVISIBLE_UNICODE_CHARS))



