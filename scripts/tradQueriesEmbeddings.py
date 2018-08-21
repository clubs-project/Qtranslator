#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Translates the query terms previously extracted with extractParse.py.
    Only the MeSH quad-lexicon is used. 
    Translation at word level using multilingual word embeddings extracted from a ML-NMT system
    is applied if the complete term is not in the dictionary    
    Date: 16.08.2018
    Author: cristinae
"""

import sys
import resource
import os.path
import unicodedata
import easyBPE
from gensim.models import KeyedVectors

class QueryTrad(object):

    # File location
    # TODO: config file
    modelPath = "../../models/"
    ctFile = modelPath + "CT/meshSplit2.concat.lc.txt"
    swFile = modelPath + "DeEnEsFr.plain.sw"
    BPEcodes = modelPath + "L1L2.final.bpe"
    embeddingRoot = modelPath + "embeddingsL1."

    numTermsUntrad = 0
    numTerms = 0
    numWordsUntrad = 0
    numWords = 0

    def __init__(self):

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


def rreplace(s, old, new, occurrence):
    """ Replace last occurrence of a substring in a string
    https://stackoverflow.com/questions/2556108/rreplace-how-to-replace-the-last-occurrence-of-an-expression-in-a-string
    """
    li = s.rsplit(old, occurrence)
    return new.join(li)

def percentage2d(part, whole):
    if (part != 0):
       value = 100*float(part)/float(whole)
       return "{0:.1f}".format(value)
    else:
       return "0"

def remove_diacritic(input):
    '''
    Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    '''
    return unicodedata.normalize('NFKD', input).encode('ASCII', 'ignore')

def cleanEndString(toClean):
    """ Removes ',' and '"' from the end of the string
    """
    clean=toClean.replace('Ü', 'ü')
    if len(clean)>1:
       if clean[-1] == ',': 
          clean = clean[:-1]
    if len(clean)>1:
       if clean[-1] == '"': 
          clean = clean[:-1]
    if len(clean)>1:
       if clean[0] == '"': 
          clean = clean[1:]
    return clean


def checkCase(toCheck, ctDict):
    """ Looks for a case that matches the casing in the lexicon and returns the 
        string to translate and the capitalisation that has to be restored 
    """
    # we check for all the capitalizations
    capitalized = False
    if toCheck.istitle():
       capitalized = True
    if toCheck in ctDict:
       toTrad = toCheck
    elif toCheck.lower() in ctDict:
       toTrad = toCheck.lower()
    elif toCheck.capitalize() in ctDict:
       toTrad = toCheck.capitalize()
    else:
       toTrad = toCheck
    return capitalized, toTrad


def extractTradFromDict(toTrad, capitalized, stringTrad, ctDict):
    """ Extracts the translation in the four languages of a term in the lexicon
        and returns a string with the translation in the four languages 
    """
    # 11-Desoxycortison|||en:Cortodoxone|||es:Cortodoxona|||fr:Cortodoxone
    entries = ctDict[toTrad]
    trads = entries.split("|||")
    for trad in trads:
        (lang, translation) = trad.split(":")
         # recover the source casing in the translation
        if capitalized == True:
           translation = translation.capitalize()
        else:
           translation = translation.lower()
        stringTrad = stringTrad + " "+lang+"::"+translation
    return stringTrad


def translate(string, proc):
    """ Translates an input string. If it is not found in the dictionary, the string
        is split into words and translated independently with word embeddings. 
    """

    ctDict = proc.getCtDict()
    swList = proc.getSWlist()
    string=cleanEndString(string)
    capitalized, toTrad =  checkCase(string, ctDict)
    print(toTrad)
    #if (complete):
    #    numTerms += 1
    #    words = string.split(" ")
    #    numWords = numWords + len(words)
    # entries are read in a specific language order (python >3.6)
    stringTrad = ""
    # First we check if the full phrase is in the lexicon
    if toTrad in ctDict:
       stringTrad = extractTradFromDict(toTrad, capitalized, stringTrad, ctDict)
    else:
        words = toTrad.split()
        # if it is not we split by word
        stringTrad = ""
        for word in words:
        # we ignore any word that is a stopword in any language
            if word in swList:
               continue
        # and check if they are in the lexicon
            capitalized, toTrad =  checkCase(string, ctDict)
            if toTrad in swList:
               continue
            if toTrad in ctDict:
               stringTrad = stringTrad + extractTradFromDict(toTrad, capitalized, stringTrad, ctDict)
            else:
           # if not, we look for the closest translation(s) in the embeddings space
               bped = easyBPE.applyBPE(proc.bpe, word)
               for subunit in bped:
                   print(subunit)
                   vector =  proc.embeddingL1[subunit]
                   allSubunit = proc.embeddingL1.similar_by_vector(vector,topn=5)
                   enSubunit = proc.embeddingEn.similar_by_vector(vector,topn=2)
                   esSubunit = proc.embeddingEs.similar_by_vector(vector,topn=2)
                   deSubunit = proc.embeddingDe.similar_by_vector(vector,topn=2)
                   frSubunit = proc.embeddingFr.similar_by_vector(vector,topn=2)
                   print(allSubunit)
                   print(enSubunit)
                   print(esSubunit)
                   print(deSubunit)
                   print(frSubunit)

    return stringTrad


def main(inF, outF):

    # Initialise a new process for translation
    proc = QueryTrad()

    # Read the queries from file
    fOUT = open(outF, 'w')
    with open(inF) as f:
       for line in f:
           line = line.strip()
           fields = line.split('\t')
           lineTrad = fields[0] + "\t["
           # eliminate the list format. Is there a better way?
           terms = fields[1].replace("[","")
           terms = terms.replace("]","")
           terms = terms.replace("',","")
           termsArray = terms.split("'")
           # split terms in subunits
           for term in termsArray[1:]:
               if term=="":
                  continue
               stringTrad = ''
               stringTrad = translate(term, proc)
               termTrad = "'" + stringTrad
               lineTrad = lineTrad + termTrad + "', "
           #rof termsArray
           lineTrad = rreplace(lineTrad, "', ", "", 2) + "]"
           fOUT.write(lineTrad+"\n")

    fOUT.close()   


if __name__ == "__main__":
    
    if len(sys.argv) is not 3:
        sys.stderr.write('Usage: python3 %s inputFile outputFile\n' % sys.argv[0])
        sys.exit(1)
    print("WARNING: This software needs python >3.6 to run properly\n")
    main(sys.argv[1], sys.argv[2])

    # CHECK: source==target doesn't mean untranslated
    #numTermTrad = numTerms-numTermsUntrad
    #numWordTrad = numWords-numWordsUntrad

    # LaTeX friendly, human unfriendly
    #print(str(numTermTrad) + " ("+percentage2d(numTermTrad, numTerms)+"\\%) "+" & "+ str(numTermsUntrad) + " ("+percentage2d(numTermsUntrad, numTerms)+"\\%) "+" & "+ str(numWordTrad) + " ("+percentage2d(numWordTrad, numWords)+"\\%) "+" & "+ str(numWordsUntrad) + " ("+percentage2d(numWordsUntrad, numWords)+"\\%) \\\\")
    #print(str(numTermsUntrad) + " untranslated parts, " + str(numTerms) + " total parts")
    #print(str(numWordsUntrad) + " untranslated words " + str(numWords)+ " total words")
 

