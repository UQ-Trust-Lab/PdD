__all__ = ["SHIFT_KEY_FORWARD_DICT", "SHIFT_KEY_REVERSE_DICT", "SHIFT_KEY_DICT"]

from types import MappingProxyType

from .letter_cases import LETTER_LOWER2UPPER_DICT, LETTER_UPPER2LOWER_DICT

SHIFT_KEY_FORWARD_DICT = MappingProxyType(
    {
        "`": ["~"],
        "1": ["!"],
        "2": ["@"],
        "3": ["#"],
        "4": ["$"],
        "5": ["%"],
        "6": ["^"],
        "7": ["&"],
        "8": ["*"],
        "9": ["("],
        "0": [")"],
        "-": ["_"],
        "=": ["+"],
        "[": ["{"],
        "]": ["}"],
        "\\": ["|"],
        ";": [":"],
        "'": ['"'],
        ",": ["<"],
        ".": [">"],
        "/": ["?"],
    }
)

SHIFT_KEY_REVERSE_DICT = MappingProxyType(
    {
        "~": ["`"],
        "!": ["1"],
        "@": ["2"],
        "#": ["3"],
        "$": ["4"],
        "%": ["5"],
        "^": ["6"],
        "&": ["7"],
        "*": ["8"],
        "(": ["9"],
        ")": ["0"],
        "_": ["-"],
        "+": ["="],
        "{": ["["],
        "}": ["]"],
        "|": ["\\"],
        ":": [";"],
        '"': ["'"],
        "<": [","],
        ">": ["."],
        "?": ["/"],
    }
)

SHIFT_KEY_DICT = {}
SHIFT_KEY_DICT.update(SHIFT_KEY_FORWARD_DICT)
SHIFT_KEY_DICT.update(SHIFT_KEY_REVERSE_DICT)
SHIFT_KEY_DICT.update(LETTER_LOWER2UPPER_DICT)
SHIFT_KEY_DICT.update(LETTER_UPPER2LOWER_DICT)
SHIFT_KEY_DICT = MappingProxyType(SHIFT_KEY_DICT)
