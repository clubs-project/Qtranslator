#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Given a quadlexion, it extracts training samples (positive and negative) 
    for reranking word embeddings and several features associated to each sample.
    Current features include:
    - (4) words and languages (w1, L1, w2, L2)
    - (2) subunits: source is a BPE subunit --instead of word--, or src and tgt 
          have a BPE mark (srcSubUnit, bothBPEmark)
    - (1) Levenshtein distance between tokens w1 and w2 (lev)
    - (3) length in characters without bpe mark @@ (l1,l2,l1/l2)
    - (2) embeddings: rank and cosine similarity (rankW2, cosSimWE)
    - (3) character n-gram similarity (cosSimN2,cosSimN3,cosSimN4)
    - (1) Levenshtein distance between Metaphone 2 phonetic keys (levM2)
    
    Date: 22.08.2018
    Author: cristinae
"""

import sys
import os.path
from collections import defaultdict as ddict
import math
import numpy as np 

import load
import easyBPE

from gensim.models import KeyedVectors
import Levenshtein 
import phonetics

bpeMark = '@@'
emptyMark = 'EMPTY'
header = 'Gold,w1,L1,w2,L2,srcSubUnit,bothBPEmark,rankW2,cosSimWE,l1,l2,l1/l2,lev,cosSimN2,cosSimN3,cosSimN4,levM2\n'
# for debugging
countUp = 0 
countDown = 0

def findNonTrad(w1, l1, w2, l2, proc):
    '''
    Find a word nonTrad which is close to the true translation of w1, w2, but it is not.
    Currently we use the word that is +1 or -1 in the cosine similarity ranking.
    This word will be used as a negative example
    Probably good for margin-based algorithms, but for others?
    '''

    global countUp 
    global countDown

    if w1 not in proc.embeddingL1.vocab:
       return 0,emptyMark,0

    vector =  proc.embeddingL1[w1]
    # Load the adequate subembeddings
    newSpace = proc.embeddingEn
    try:
       if l2 is "en":
          pass
       elif l2 is "de":
          newSpace = proc.embeddingDe
       elif l2 is "es":
          newSpace = proc.embeddingEs
       elif l2 is "fr":
          newSpace = proc.embeddingFr
    except ValueError:
       print("No correct language specified")   

    # add the source word to the target language subspace
    # since que cannot remove a word afterwards we create a new space
    newSpace.add(w1,vector,replace=True)
    # we look for the closest alternative to w2 (+1 or -1 in rank) to create an
    # entry for a non-translation 
    if w2 in newSpace.vocab:
       w2Rank = newSpace.rank(w1,w2)
       if w2Rank == 1:           # if w1 was the top1 we can only do rank+1
          rank = 2
          countUp = countUp+1
       else:                     # if not, we can randomly select rank +-1
                                 # we give higher probability to -1 to get a balanced corpus
          sumVal = np.random.choice([-1, 1], size=1, p=[5./6, 1./6])
          rank = w2Rank + sumVal[0]
          if sumVal[0]==1:
             countUp = countUp+1
          if sumVal[0]==-1:
             countDown = countDown+1
 
       # we look for the rank_th word
       toprank = newSpace.similar_by_vector(vector,topn=rank)
       nonTrad = toprank[-1][0]
    else:
       return 0,emptyMark,0

    # cleaning
    newSpace = None

    return w2Rank, nonTrad, rank


def extractFeatures(w1, w2, proc):
    '''
    Extracts the set of features for a pair of word (w1, w2)
    Returns a string with a cvs format for the features
    '''

    # length of the inputs
    s1 = w1.replace(bpeMark, '')
    s2 = w2.replace(bpeMark, '')
    lengths = str(len(s1))+','+str(len(s2))+','+str("{0:.2f}".format(len(s1)/len(s2)))

    # Levenshtein between tokens
    leven = Levenshtein.distance(w1, w2)

    # cosine similarity between common n-grams
    n2 = round(char_ngram(w1, w2, 2),4)
    n3 = round(char_ngram(w1, w2, 3),4)
    n4 = round(char_ngram(w1, w2, 4),4)
    ngrams = str(n2)+','+str(n3)+','+str(n4)

    # cosine similarity between word embeddings
    if w1 in proc.embeddingL1.vocab and w2 in proc.embeddingL1.vocab:
       dWE = proc.embeddingL1.similarity(w1, w2)
    else:
       dWE = 0

    # Levenshtein between Metaphone 2 phonetic keys of the tokens
    # TODO: port the java version of metaphone 3
    w1M2 = phonetics.dmetaphone(w1)
    w2M2 = phonetics.dmetaphone(w2)
    levenM2 = Levenshtein.distance(w1M2[0], w2M2[0])

    features = str(dWE) +','+ lengths+','+str(leven)+','+ ngrams +','+ str(levenM2)
    return features


def cosine_sim(dictionary):
    '''
    Cosine similarity estimation for the dictionaries
    from https://github.com/adamcsvarga/parallel_creator/blob/master/rerank/rerank.py
    '''

    cosine_sim = 0
    for value in dictionary.values():
        cosine_sim += value[0] * value[1]

    length1 = sum([value[0] ** 2 for value in dictionary.values()]) 
    length2 = sum([value[1] ** 2 for value in dictionary.values()])
    
    try:            
        cosine_sim /= math.sqrt(length1) * math.sqrt(length2)
    except ZeroDivisionError:
        cosine_sim = 0
        
    return cosine_sim


def char_ngram(w1, w2, n=2):
    '''
    cosine distance of n-gram character vectors
    from https://github.com/adamcsvarga/parallel_creator/blob/master/rerank/rerank.py
    '''
    
    ngramdict = ddict(lambda: ddict(lambda: 0))
    line_pair = [w1, w2]
    
    for i, line in enumerate(line_pair):
        line = ''.join(line.split())
        j = 0
        while (j + (n - 1)) < len(line):
            ngramdict[line[j:j+n]][i] += 1
            j += 1
    
    return cosine_sim(ngramdict)
   


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

    isWord='0'
    isSubword='1'

    # Initialise a new process for translation, loading the models
    # Loads more stuff than we need, we should parametrise that
    modelPath = "../../models/"
    proc = load.QueryTrad(modelPath)

    outF = inF+'.feat'
    fOUT = open(outF, 'w')
    fOUT.write(header)
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
           # or n-to-n bpe subunits
           units = easyBPE.applyBPE(proc.bpe, source_word)
           
           entriesOutput = ''
           src_lang = sourceLang(line)

           single = True
           subunits = True
           for i in range(1, len(entries)):
               l = entries[i].split(":")[0]  # language
               t = entries[i].split(":")[1]  # token
               tmultiple = t.split(' ')
               if len(tmultiple) > 1:   # For learning we only want one-to-one translations in
                  single = False        # all the languages simultaneousy to have a balanced corpus
                  break
               tunits = easyBPE.applyBPE(proc.bpe, t)
               if len(tunits) != len(units):  # we only keep n-to-n pairs in all the languages
                  subunits = False            # simultaneousy to have a balanced corpus
                  break

               if len(units)==1:
                  # we look for a negative example (bad translation of source_word)
                  w2Rank,tNeg,rank = findNonTrad(source_word, src_lang, t, l, proc)
                  # basic features, including rank in WE similarity 
                  bothBPE = '0'
                  pairPos = source_word +","+src_lang+","+ t +","+ l +","+isWord+","+bothBPE+","+str(w2Rank)+","
                  # extraction of non-basic features
                  featsPos = extractFeatures(source_word, t, proc)
                  entriesOutput = entriesOutput +"1,"+ pairPos + featsPos +"\n"
                  # add features for a negative example in case there is one 
                  if tNeg is not "EMPTY":
                     pairNeg = source_word +","+ src_lang +","+ tNeg +","+ l +","+isWord+","+bothBPE+","+str(rank)+","
                     featsNeg = extractFeatures(source_word, tNeg, proc)
                     entriesOutput = entriesOutput +"0,"+ pairNeg + featsNeg +"\n" 
               # if the input token has been bped do the same for all the subunits
               else:
                  for subunit_src, subunit_tgt in zip(units, tunits):
                      bothBPE = '0'
                      if bpeMark in subunit_src and bpeMark in subunit_tgt:
                         bothBPE = '1'
                      w2Rank,tNeg,rank = findNonTrad(subunit_src, src_lang, subunit_tgt, l, proc)
                      pairPos = subunit_src +","+ src_lang +","+ subunit_tgt +","+ l +","+isSubword+","+bothBPE+","+str(w2Rank)+","
                      featsPos = extractFeatures(subunit_src, subunit_tgt, proc)
                      entriesOutput = entriesOutput +"1,"+ pairPos + featsPos +"\n" 
                      if tNeg is not "EMPTY":
                         pairNeg = subunit_src +","+ src_lang +","+ tNeg +","+ l +","+isSubword+","+bothBPE+","+str(rank)+","
                         featsNeg = extractFeatures(subunit_src, tNeg, proc)
                         entriesOutput = entriesOutput +"0,"+ pairNeg + featsNeg +"\n" 

           if (single == False):     
               continue
           if (subunits == False):     
               continue
           fOUT.write(entriesOutput)


    print("\nDone")
    print("countUp: "+str(countUp))
    print("countDown: "+str(countDown))

    fOUT.close()   



if __name__ == "__main__":
    
    if len(sys.argv) is not 2:
        sys.stderr.write('Usage: python3 %s lexiconFile\n' % sys.argv[0])
        sys.exit(1)
    main(sys.argv[1])

