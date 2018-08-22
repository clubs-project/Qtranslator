#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Extract training samples and their features from a dictionary
    Date: 22.08.2018
    Author: cristinae
"""

import sys
import os.path
import easyBPE

from gensim.models import KeyedVectors


# File location
# TODO: config file
modelPath = "../../models/"
embeddingRoot = modelPath + "embeddingsL1solr."
BPEcodes = modelPath + "L1L2.final.bpe"


def extractFeatures(w1, w2, embeddingL1):
    dWE = embeddingL1.similarity(w1, w2)
    return dWE

def sourceLang(entry):
    '''determines the language of the source word'''

    entries = entry.split("|||")
    translation_languages = set()
    language = ""
    for i in range(1, len(entries)):
        translation_languages.add(entries[i].split(":")[0])

    if "en" not in translation_languages:
        language = "en"
    elif "de" not in translation_languages:
        language = "de"
    elif "fr" not in translation_languages:
        language = "fr"
    else:
        language = "es"
    return language


def main(inF):

    # Load the embeddings
    print("\n..Loading embeddings")
    embeddingL1 = KeyedVectors.load_word2vec_format(embeddingRoot+'w2v', binary=False)
     # Load BPE model
    print("\n..Loading BPE model")
    bpe = easyBPE.BPE(self.BPEcodes)

    outF = inF+'.feat'
    fOUT = open(outF, 'w')
    # Read the quad-lexicon
    with open(inF) as f:
       for line in f:
           line = line.strip()
           entries = line.split("|||")
           source_word = entries[0]
           # For learning we only want one-to-one translations
           multiple = source_word.split(' ')
           if len(multiple) > 1:
              continue

           entriesOutput = ''
           src_lang = sourceLang(line)

           word = {}
           word[src_lang]=source_word
           single = True
           for i in range(1, len(entries)):
               l = entries[i].split(":")[0]  # language
               t = entries[i].split(":")[1]  # token
               tmultiple = t.split(' ')
               if len(tmultiple) > 1:
                  single = False
                  break
               word[l]=t
               pair = source_word +","+ src_lang +","+ t +","+ l +","
               feats = extractFeatures(source_word, t, embeddingL1)
               entriesOutput = entriesOutput + pair + feats +"\n" 
           
           if (single == False):     # For learning we only want one-to-one translations
               continue
           fOUT.write(entriesOutput)


    fOUT.close()   



if __name__ == "__main__":
    
    if len(sys.argv) is not 2:
        sys.stderr.write('Usage: python3 %s lexiconFile\n' % sys.argv[0])
        sys.exit(1)
    main(sys.argv[1])

