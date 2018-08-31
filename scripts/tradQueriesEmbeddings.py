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
import pickle
import xgboost
#from xgboost import XGBClassifier
from sklearn.externals import joblib

trainingSet = 'untradDEallkeys.solr.all-languages'
explora = 10


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
	
    colums2scale = features.getColums2scale()
    scaler = joblib.load(modelPath+'reranker/'+trainingSet+'.scaler.pkl') 
    df4ML[colums2scale] = scaler.fit_transform(df4ML[colums2scale])

    return df4ML


def predictBestTrad(df):
    """ Given a previously trained model, it returns the index of the best translation
        for a test set in a dataframe
    """

    nbest = df.loc[:, features.getFeatureCols()]

    #X = df4ML.loc[:, feature_cols]
    clf = xgboost.Booster()  # init model

    # Load previously trained model
    #clf.load_model(modelPath+'reranker/'+trainingSet+'.model.pkl')
    clf = pickle.load(open(modelPath+'reranker/'+trainingSet+'.model.pkl', "rb"))
    # make prediction
    nbestProbs = clf.predict_proba(nbest)
    #print(nbestProbs)
    indexTrad = np.argmax(nbestProbs[:,1])

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

def recoverCasing(word, capitalized):
    """ Recovers the initial capitalisation of a word
    """
    if capitalized == True:
       word = word.capitalize()
    else:
       word = word.lower()
    return word

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
        translation = recoverCasing(translation, capitalized)
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
    # print(toTrad)
    stringTrad = ""
    # First we check if the full phrase is in the lexicon
    if toTrad in ctDict:
       stringTrad = extractTradFromDict(toTrad, capitalized, stringTrad, ctDict)
    else:
       words = toTrad.split()
       # if it is not we split by word
       stringTrad = ''
       for word in words:
           # we check if words are in the lexicon
           capitalized, toTrad =  checkCase(string, ctDict)
           # we ignore any word that is a stopword in any language
           if word in swList:
              stringTrad = stringTrad + " ##SW##"
           elif toTrad in ctDict:
              stringTrad = stringTrad + extractTradFromDict(toTrad, capitalized, stringTrad, ctDict)
           else:
           # if not, we look for the closest translation(s) in the embeddings space
              isSubWord = '0'
              bped = easyBPE.applyBPE(proc.bpe, word)
              if len(bped) >1:
                 isSubWord = '1'
              wordEn = ''
              wordEs = ''
              wordDe = ''
              wordFr = ''
              for subunit in bped:
                  vector =  proc.embeddingL1[subunit]
                  for lan in "en", "es", "fr", "de":
                      try:
                        if lan == "en":
                           lanSpace = proc.embeddingEn
                        elif lan == "de":
                           lanSpace = proc.embeddingDe
                        elif lan == "es":
                           lanSpace = proc.embeddingEs
                        elif lan == "fr":
                           lanSpace = proc.embeddingFr
                      except ValueError:
                        lanSpace = None 
                        print("No correct language specified")   

                      lanSubunits = lanSpace.similar_by_vector(vector,topn=explora)

                      allFeats = features.getHeaderTest()
                      prevw1 = features.getEmptyMark()
                      prevw2 = features.getEmptyMark()
                      for subunitTrad in lanSubunits:
                          # populate for a dataframe with the n-best list
                          w2 = subunitTrad[0]
                          #print(subunit +"   "+ w2)
                          bothBPE = '0'
                          if features.getBpeMark() in subunit and features.getBpeMark() in w2:
                             bothBPE = '1'
                          basicFeats = features.basicFeatures(subunit,'xx', w2, lan, isSubWord, bothBPE)
                          semFeats = features.extractSemFeatures(subunit, w2, lan, explora+100, proc)
                          lmFeats = features.extractSimBigram(subunit, w2, prevw1, prevw2, proc, lanSpace)
                          lexFeats = features.extractLexFeatures(subunit, w2)
                          allFeats = allFeats + basicFeats+semFeats+lmFeats + lexFeats +'\n'
                          prevw1 = subunit
                          prevw2 = w2

                      # create preprocessed data frame
                      df = preprocessingRead(allFeats)
                      indexTrad = predictBestTrad(df)
                      # reconstructing BPE without BPE mark
                      #print("index: "+str(indexTrad))
                      if lan == "en":
                         wordEn = wordEn+lanSubunits[indexTrad][0]
                      elif lan == "de":
                         wordDe = wordDe+lanSubunits[indexTrad][0]
                      elif lan == "es":
                         wordEs = wordEs+lanSubunits[indexTrad][0]
                      else:
                         wordFr = wordFr+lanSubunits[indexTrad][0]

              wordEn = recoverCasing(wordEn, capitalized)
              wordEs = recoverCasing(wordEs, capitalized)
              wordDe = recoverCasing(wordDe, capitalized)
              wordFr = recoverCasing(wordFr, capitalized)
              stringTrad = stringTrad + " en::"+wordEn + " es::"+wordEs + " de::"+wordDe + " fr::"+wordFr
              #print(stringTrad)
              stringTrad = stringTrad.replace(features.getBpeMark(), '')

    return stringTrad


def main(inF, scriptPath):

    global modelPath
    modelPath = scriptPath+"../models/"
    # Initialise a new process for translation, loading the models
    proc = load.QueryTrad(modelPath)

    outF = inF+'.trad'
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
           lineTrad = rreplace(lineTrad, ", ", "", 1) + "]"
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
 

