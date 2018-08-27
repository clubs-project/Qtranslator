#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Module with functions to calculate semantic and lexical features

    Date: 27.08.2018
    Author: cristinae
"""

import sys
import os.path
from collections import defaultdict as ddict
import math
import numpy as np 

from gensim.models import KeyedVectors
import Levenshtein 
import phonetics

bpeMark = '@@'


def basicFeatures(w1, l1, w2, l2, isSubWord, bothBPE):
    '''
    Creates a string with a cvs format for the basic features
    '''
    return w1+","+l1+","+ w2+","+l2+","+isSubWord+","+bothBPE+","



def extractSemFeatures(w1, w2, l2, proc):
    '''
    Extracts the set of semantic features related to word embeddings for a pair of word (w1, w2)
    Returns a string with a cvs format for the features
    '''

    noSimsRank = '0,0,0,0,0,0,'
    #WEsim,rankW2,simRankt1,simRankWnext,simRankt10,simRankt100

    if w1 not in proc.embeddingL1.vocab:
       return noSimsRank

    #TODO I probably can do everything with L1, and use vocabularies for languages
    vector =  proc.embeddingL1[w1]
    # Load the adequate subembeddings
    try:
       if l2 == "en":
          newSpace = proc.embeddingEn
       elif l2 == "de":
          newSpace = proc.embeddingDe
       elif l2 == "es":
          newSpace = proc.embeddingEs
       elif l2 == "fr":
          newSpace = proc.embeddingFr
    except ValueError:
       newSpace = None 
       print("No correct language specified")   

    # add the source word to the target language subspace
    # since que cannot remove a word afterwards we create a new space
    newSpace.add(w1,vector,replace=True)
    if w2 in newSpace.vocab:
       w2Rank = newSpace.rank(w1,w2)
       toprank = newSpace.similar_by_vector(vector,topn=explore)
       sim = newSpace.similarity(w1,w2)
       simRankt1 = toprank[w2Rank-1][1] - toprank[0][1]  # how far in similarity is w2 to top1
       simRanktnext = toprank[w2Rank-1][1] - toprank[w2Rank][1]  # how far in similarity is w2 to the next word
       simRankt10 = toprank[w2Rank-1][1] - toprank[9][1] # how far in similarity is w2 to top10
       simRankt100 = toprank[w2Rank-1][1] - toprank[99][1] # how far in similarity is w2 to top100
       simsRankW2 = str(sim)+','+str(w2Rank)+','+str(simRankt1)+','+str(simRanktnext)+','+str(simRankt10)+','+str(simRankt100)+','
       
    else:
       return noSimsRank

    # cleaning
    newSpace = None

    return simsRankW2


def extractLexFeatures(w1, w2):
    '''
    Extracts the set of features for a pair of word (w1, w2)
    Returns a string with a cvs format for the features
    '''

    # length of the inputs
    s1 = w1.replace(bpeMark, '')
    s2 = w2.replace(bpeMark, '')    
    try:            
      lengths = str(len(s1))+','+str(len(s2))+','+str("{0:.2f}".format(len(s1)/len(s2)))
    except ZeroDivisionError:
      lengths = str(len(s1))+','+str(len(s2))+','+str("{0:.2f}".format(0.00))

    # Levenshtein between tokens
    leven = Levenshtein.distance(w1, w2)

    # cosine similarity between common n-grams
    n2 = round(char_ngram(w1, w2, 2),4)
    n3 = round(char_ngram(w1, w2, 3),4)
    n4 = round(char_ngram(w1, w2, 4),4)
    ngrams = str(n2)+','+str(n3)+','+str(n4)

    # moved to the estimation of semantic features
    # cosine similarity between word embeddings
    # if w1 in proc.embeddingL1.vocab and w2 in proc.embeddingL1.vocab:
    #   dWE = proc.embeddingL1.similarity(w1, w2)
    #else:
    #   dWE = 0
 
    # Levenshtein between Metaphone 2 phonetic keys of the tokens
    # TODO: port the java version of metaphone 3
    w1M2 = phonetics.dmetaphone(w1)
    w2M2 = phonetics.dmetaphone(w2)
    levenM2 = Levenshtein.distance(w1M2[0], w2M2[0])

    features = lengths+','+str(leven)+','+ ngrams +','+ str(levenM2)
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
   


