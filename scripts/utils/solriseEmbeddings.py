#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Applies the same processing PubPsych solr does to the queries to the word embeddings file
    It uses the processing of:
    https://github.com/clubs-project/DBtranslator/blob/master/scripts/utils/preprocess_dicts.py
    Date: 22.08.2018
    Author: cristinae
"""

import sys
import os.path
import unicodedata
import string
import re

BPEseparator = '@@'

def replace_punctuation(word):
    # Solr replaces punctuation signs with whitespaces when parsing queries
    regex = re.compile('[%s]' % re.escape(string.punctuation))
    return regex.sub(' ', word)


def process(source_word):
    '''mimicking Solr processing:
    - lowercase the token
    - removes diacritics ('ü' -> 'u')
    - replaces 'ß' with 'ss' 
    - replaces punctuation including hyphens with whitespace
    - keeps BPE notation
    '''
 
    addBPEsep = False
    if (source_word.endswith(BPEseparator)):
        addBPEsep = True
    # lowercase
    source_word = source_word.lower()
    # ß has to be replaced manually, since unicodedata.normalize simply deletes it instead of replacing it with ss
    source_word = source_word.replace('ß', 'ss')
    # remove diacritics
    source_word = unicodedata.normalize('NFKD', source_word).encode('ASCII', 'ignore').decode()
    # String.punctuation only knows ASCII punctuation
    source_word = replace_punctuation(source_word)
    source_word = source_word.rstrip()
    if (addBPEsep == True):
        source_word = source_word + BPEseparator
    return source_word 


def main(inF, outF):

    fOUT = open(outF, 'w')
    # Read the input embedding file
    with open(inF) as f:
       for line in f:
           line = line.strip()
           word, vector = line.split(' ', 1)
           newWord = process(word)
           fOUT.write(newWord+" "+vector+"\n")

    fOUT.close()   



if __name__ == "__main__":
    
    if len(sys.argv) is not 3:
        sys.stderr.write('Usage: python3 %s inputFile outputFile\n' % sys.argv[0])
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])

