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
import os.path
import unicodedata
from io import StringIO

import load
import easyBPE
import features
from gensim.models import KeyedVectors

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

trainingSet = 'untradDEallkeys.solr.all-languages'
bpeMark = '@@'

def preprocessingRead(data):
    """
     Basic preprocessing for most of ML algorithms: binarisation, normalisation, scaling 
     Reads from a saved scaling model for test usage
    """

    global modelPath

    # extract the data into a dataframe
    df = pd.read_csv(StringIO(data))

    # convert categorical column in four binary columns, one per language
    df4ML = df.join(pd.get_dummies(df['L2'],prefix='L2'))

    # scale columns
    # rankW2 has huge numbers in a wide interval, we should cut and/or move to a log scale
    df4ML['rankW2'] = df4ML['rankW2'].apply(lambda x: 1000 if x>1000 else x)
    #df4ML['rankW2'] = df4ML['rankW2'].apply(lambda x: 0 if x<= 0 else math.log10(x))
	
    #colums2scale = ['rankW2','WEsim','l1','l2','l1/l2','lev','cosSimN2','cosSimN3','cosSimN4','levM2']
    colums2scale = ['rankW2','l1','l2','l1/l2','lev','levM2','simRankt1','simRankWnext','simRankt10','simRankt100']
    scaler = joblib.load(modelPath+'reranker/'+trainingSet+'.scaler.pkl') 
    df4ML[colums2scale] = scaler.fit_transform(df4ML[colums2scale])

    return df4ML


def predictBestTrad(df):
    """
    """

    feature_cols = ['L2_de','L2_en','L2_es','L2_fr','srcSubUnit','bothBPEmark','WEsim','rankW2','simRankt1','simRankWnext','simRankt10','simRankt100','l1','l2','l1/l2','lev','cosSimN2','cosSimN3','cosSimN4','levM2']
    nbest = df.loc[:, feature_cols]
 
    clf = xgb.Booster()  # init model

    # Load previously trained model
    clf.load_model(modelPath+'reranker/'+trainingSet+'.model')

    # make prediction, a probability by default with XGB
    nbestProbs = clf.predict(nbest)  
    indexTrad = np.argmax(nbestProbs)

    return indexTrad


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
    """ Accept a unicode string, and return a normal string (bytes in Python 3)
    without any diacritical marks.
    """
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
               isSubWord = '0'
               bped = easyBPE.applyBPE(proc.bpe, word)
               if len(bped) >1:
                  isSubWord = '1'
               for subunit in bped:
                   print(subunit)
                   vector =  proc.embeddingL1[subunit]

                   enSubunits = proc.embeddingEn.similar_by_vector(vector,topn=1000)
                   allFeats = features.getHeader()
                   for subunitTrad in enSubunits:
                       # populate for a dataframe with the n-best list
                       w2 = subunit[0]
                       bothBPE = '0'
                       if bpeMark in subunit and bpeMark in w2:
                         bothBPE = '1'
                       basicFeats = features.basicFeatures(subunit,'xx', w2, 'en', isSubWord, bothBPE)
                       semFeats = features.extractSemFeatures(subunit, w2, 'en', 40000, proc)
                       lexFeats = features.extractLexFeatures(subunit, w2)
                       allFeats = allFeats + basicFeats+semFeats+lexFeats+'\n'
                    # create preprocessed data frame
                   df = preprocessingRead(allFeats)
                   indexTrad = predictBestTrad(df)
                   trad = enSubunits[indexTrad][0]
                   print(trad)




                   #esSubunit = proc.embeddingEs.similar_by_vector(vector,topn=2)
                   #deSubunit = proc.embeddingDe.similar_by_vector(vector,topn=2)
                   #frSubunit = proc.embeddingFr.similar_by_vector(vector,topn=2)
                   #print(enSubunits)
                   #print(esSubunit)
                   #print(deSubunit)
                   #print(frSubunit)
              # we need to reconstruct BPE

    return stringTrad


def main(inF, scriptPath):

    global modelPath
    modelPath = scriptPath+"../models/"
    # Initialise a new process for translation, loading the models
    proc = load.QueryTrad(modelPath)

    outF = inF+'trad'
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
    
    if len(sys.argv) is not 2:
        sys.stderr.write('Usage: python3 %s inputFile\n' % sys.argv[0])
        sys.exit(1)
    print("WARNING: This software needs python >3.6 to run properly\n")
    scriptPath = os.path.dirname(os.path.abspath( __file__ ))
    main(sys.argv[1], scriptPath+'/')

    # CHECK: source==target doesn't mean untranslated
    #numTermTrad = numTerms-numTermsUntrad
    #numWordTrad = numWords-numWordsUntrad

    # LaTeX friendly, human unfriendly
    #print(str(numTermTrad) + " ("+percentage2d(numTermTrad, numTerms)+"\\%) "+" & "+ str(numTermsUntrad) + " ("+percentage2d(numTermsUntrad, numTerms)+"\\%) "+" & "+ str(numWordTrad) + " ("+percentage2d(numWordTrad, numWords)+"\\%) "+" & "+ str(numWordsUntrad) + " ("+percentage2d(numWordsUntrad, numWords)+"\\%) \\\\")
    #print(str(numTermsUntrad) + " untranslated parts, " + str(numTerms) + " total parts")
    #print(str(numWordsUntrad) + " untranslated words " + str(numWords)+ " total words")
 

