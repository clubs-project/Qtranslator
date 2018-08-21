#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
EXTRACTED AND MODIFIED FROM apply_bpe.py

Use operations learned with learn_bpe.py to encode a new text.
The text will not be smaller, but use only a fixed vocabulary, with rare words
encoded as variable-length sequences of subword units.

Reference:
Rico Sennrich, Barry Haddow and Alexandra Birch (2015). Neural Machine Translation of Rare Words with Subword Units.
Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
"""

from __future__ import unicode_literals, division

import sys
import os
import inspect
import codecs
import io
import re
import warnings


def applyBPE(self, word):
    """segment a word with BPE encoding"""
    output = []
    new_word = encode(word,self.bpe_codes,self.bpe_codes_reverse,self.separator,self.version,self.cache)

    for item in new_word[:-1]:
        output.append(item + self.separator)
    output.append(new_word[-1])
    return output


def get_pairs(word):
    """Return set of symbol pairs in a word.
    word is represented as tuple of symbols (symbols being variable-length strings)
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def encode(orig, bpe_codes, bpe_codes_reverse, separator, version, cache):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """

    if orig in cache:
        return cache[orig]

    if version == (0, 1):
        word = tuple(orig) + ('</w>',)
    elif version == (0, 2): # more consistent handling of word-final segments
        word = tuple(orig[:-1]) + ( orig[-1] + '</w>',)
    else:
        raise NotImplementedError

    pairs = get_pairs(word)

    if not pairs:
        return orig

    while True:
        bigram = min(pairs, key = lambda pair: bpe_codes.get(pair, float('inf')))
        if bigram not in bpe_codes:
            break
        first, second = bigram
        new_word = []
        i = 0
        while i < len(word):
            try:
                j = word.index(first, i)
                new_word.extend(word[i:j])
                i = j
            except:
                new_word.extend(word[i:])
                break

            if word[i] == first and i < len(word)-1 and word[i+1] == second:
                new_word.append(first+second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_word = tuple(new_word)
        word = new_word
        if len(word) == 1:
            break
        else:
            pairs = get_pairs(word)

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word = word[:-1] + (word[-1].replace('</w>',''),)

    cache[orig] = word
    return word


class BPE(object):
    def __init__(self, codesFile):

        merges=-1
        separator='@@'
        codes= codecs.open(codesFile, encoding='utf-8')
        codes.seek(0)
        offset=1

        # check version information
        firstline = codes.readline()
        if firstline.startswith('#version:'):
            self.version = tuple([int(x) for x in re.sub(r'(\.0+)*$','', firstline.split()[-1]).split(".")])
            offset += 1
        else:
            self.version = (0, 1)
            codes.seek(0)

        self.bpe_codes = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(codes) if (n < merges or merges == -1)]

        for i, item in enumerate(self.bpe_codes):
            if len(item) != 2:
                sys.stderr.write('Error: invalid line {0} in BPE codes file: {1}\n'.format(i+offset, ' '.join(item)))
                sys.stderr.write('The line should exist of exactly two subword units, separated by whitespace\n')
                sys.exit(1)

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(self.bpe_codes)))])
        self.bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair,i in self.bpe_codes.items()])
        self.separator = separator
        self.cache = {}


# Initialisation
if __name__ == '__main__':
    bpe = BPE(codesFile)

 


