#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Module with an only class that simply loads the models    
    Date: 23.08.2018
    Author: cristinae
"""

import sys
import resource
import os.path
import easyBPE
from gensim.models import KeyedVectors

class QueryTrad:

    def __init__(self, modelPath):
  
        # File location
        # modelPath = "../models/"
        self.modelPath =modelPath
        self.ctFile = modelPath + "CT/meshSplit2.solr.all-languages.txt"
        self.swFile = modelPath + "DeEnEsFr.plain.sw"
        self.BPEcodes = modelPath + "L1L2.final.bpe"
        self.embeddingRoot = modelPath + "embeddingsL1solr."

        # Load stopword plain file
        swList = {}
        for line in open(self.swFile):
            line = line.strip()
            swList[line] = 1
        mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("..List of stop words loaded. Using "+str(mem)+" MB of memory")
        self.swList = swList

        # Load the lexicon
        ctDict = {}
        targets = []
        # data ordered as es->fr->de->en so that in case of ambiguity 'en' remains
        for line in open(self.ctFile):
            line = line.strip()
            if (line.startswith('<<<')):  #the new version of the dictionary has a comment line
               continue
            source, targets = line.split("|||",1)
            ctDict[source] = targets
        mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("..Multilingual lexicon loaded. Using "+str(mem)+" MB of memory")
        self.ctDict = ctDict

        # Load BPE model
        self.bpe = easyBPE.BPE(self.BPEcodes)
        mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("..BPE model loaded. Using "+str(mem)+" MB of memory")

        # Load embeddings
        self.embeddingL1 = KeyedVectors.load_word2vec_format(self.embeddingRoot+'w2v', binary=False)
        mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("\n..Multilingual embeddings loaded. Using "+str(mem)+" MB of memory")
        self.embeddingEn = KeyedVectors.load_word2vec_format(self.embeddingRoot+'en.w2v')
        mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("..English embeddings loaded. Using "+str(mem)+" MB of memory")
        self.embeddingEs = KeyedVectors.load_word2vec_format(self.embeddingRoot+'es.w2v')
        mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("..Spanish embeddings loaded. Using "+str(mem)+" MB of memory")
        self.embeddingDe = KeyedVectors.load_word2vec_format(self.embeddingRoot+'de.w2v')
        mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("..German embeddings loaded. Using "+str(mem)+" MB of memory")
        self.embeddingFr = KeyedVectors.load_word2vec_format(self.embeddingRoot+'fr.w2v')
        mem=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        print("..French embeddings loaded. Using "+str(mem)+" MB of memory\n")
 
    def getCtDict(self):
        return self.ctDict

    def getSWlist(self):
        return self.swList


