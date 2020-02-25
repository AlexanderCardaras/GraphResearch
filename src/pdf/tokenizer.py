import numpy as np


def get_word_list(text):
    unique_tokens = {}
    split = text.split()
    for t in split:
        if t not in unique_tokens:
            t = t.lower()
            unique_tokens[t] = 0

    return unique_tokens.keys()
