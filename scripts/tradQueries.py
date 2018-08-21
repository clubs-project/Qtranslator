#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Translates the query terms previously extracted with extractParse.py.
    MeSH+Wikipedia+Apertium+manual quad-lexicons are used. 
    Translation at word level is applied if the complete term is not in the dictionary, 
    and the source is used as translation in case it is missing
    Date: 06.04.2018
    Author: cristinae
"""

import sys
import os.path
import unicodedata

ctPath = "../models/CT/"

numTermsUntrad = 0
numTerms = 0
numWordsUntrad = 0
numWords = 0

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

def removePluralEnding(plural, language):
    """ Removes -s, -n, -en and -e as plural forms 
    """
    singular=plural
    # for Spanish
    if len(singular)>1 and language=="es":
       if singular[:-2] == 'es': 
          singular = singular[:-2]
       elif singular[-1] == 's': 
          singular = singular[:-1]
    # for French
    if len(singular)>1 and language=="fr":
       if singular[-1] == 's': 
          singular = singular[:-1]
    # for English
    if len(singular)>3 and language=="en":
       if singular[:-3] == 'ies': 
          return singular[:-3]+"y"
    if len(singular)>2 and language=="en":
       if singular[:-2] == 'es': 
          return singular[:-2]
    if len(singular)>1 and language=="en":
       if singular[-1] == 's': 
          singular = singular[:-1]
    # for German
    if len(singular)>2 and language=="de":
       if singular[:-2] == 'er': 
          return remove_diacritic(singular[:-2]).decode() #we need to remove the umlaut too
       if singular[-1] == 'n': 
          singular = singular[:-1]
    if len(singular)>1 and language=="de":
       if singular[-1] == 'e' or  singular[-1] == 's': 
          singular = singular[:-1]

    return singular


def translate(string, ctDict, complete, plurals, original):
    """ Translates an input string. If it is not found in the dictionary, the string
        is split into words and translated independently. If it words are not there
        either, the source is copied
    """

    global numTermsUntrad
    global numTerms
    global numWords
    global numWordsUntrad
    #global l1 #needed only for counting, so that we do it only once
    global language
    global stringTrad

    toTrad = ""
    #stringTrad = ""
    
    string=cleanEndString(string)
    # we check for all the capitalizations
    capitalized = False
    if string.istitle():
       capitalized = True
    if string in ctDict:
       toTrad = string
    elif string.lower() in ctDict:
       toTrad = string.lower()
    elif string.capitalize() in ctDict:
       toTrad = string.capitalize()
 
    if (complete):
        numTerms += 1
        words = string.split(" ")
        numWords = numWords + len(words)
    # entries are read in a specific language order (python >3.6)
    if toTrad in ctDict:
       # 11-Desoxycortison|||en:Cortodoxone|||es:Cortodoxona|||fr:Cortodoxone
       entries = ctDict[toTrad]
       if "de:" not in entries:
          language = "de"
       elif "en:" not in entries:
          language = "en"
       elif "fr:" not in entries:
          language = "fr"
       elif "es:" not in entries:
          language = "es"
       trads = entries.split("|||")
       #(trads)
       for trad in trads:
           (lang, translation) = trad.split(":")
           #if trad.startswith(lang):
           #   translation = trad.replace(lang+":","")
           # recover the source casing in the translation
           if capitalized == True:
              translation = translation.capitalize()
           else:
              translation = translation.lower()
           stringTrad = stringTrad + " "+lang+"::"+translation
       #return allTrads  
    else:
       if (complete):
           numTermsUntrad += 1
       words = string.split(" ")
       complete = False
       if len(words)==1 and plurals==True:
          #if (original!='' and lang==l1): print(original) #4debug
          if (original!=''): 
             numWordsUntrad += 1
             stringTrad = stringTrad + " cp::"+ original
          #return stringTrad + original
       # That's a mess because of the recursive function
       # Don't need to make it recursive!
       if len(words)==1 and plurals==False:
          matched = False
          newWordTMP = removePluralEnding(words[0], "en")
          newWord = newWordTMP
          if newWordTMP in ctDict or newWordTMP.lower() in ctDict or newWordTMP.capitalize() in ctDict:
             matched=True
             newWord=newWordTMP
          newWordTMP = removePluralEnding(words[0], "de")
          if not matched and (newWordTMP in ctDict or newWordTMP.lower() in ctDict or newWordTMP.capitalize() in ctDict):
             matched=True
             newWord=newWordTMP
          newWordTMP = removePluralEnding(words[0], "es")
          if not matched and (newWordTMP in ctDict or newWordTMP.lower() in ctDict or newWordTMP.capitalize() in ctDict):
             matched=True
             newWord=newWordTMP
          newWordTMP = removePluralEnding(words[0], "fr")
          if not matched and (newWordTMP in ctDict or newWordTMP.lower() in ctDict or newWordTMP.capitalize() in ctDict):
             newWord=newWordTMP
          #return translate(newWord, ctDict, complete, True, words[0])
          translate(newWord, ctDict, complete, True, words[0])
       if len(words)>1: 
          for word in words:
          #return translate(word, ctDict, complete, False, word)
              translate(word, ctDict, complete, False, word)
 

def main(inF, outF):

    global numTermsUntrad
    global numTerms
    global numWords
    global numWordsUntrad
    global language
    global stringTrad

    ctFile = ctPath + "quadLexicon.concat.lc.txt"
    #ctFile = ctPath + "meshSplit2.concat.lc.txt"
    print(ctFile)

    # Assumption
    language = "en"

    # Load the lexicon
    ctDict = {}
    targets = []
    # data ordered as es->fr->de->en so that in case of ambiguity 'en' remains
    for line in open(ctFile):
        line = line.strip()
        source, targets = line.split("|||",1)
        ctDict[source] = targets

    # Read the CTs from file
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
           for ct in termsArray[1:]:
               #ctTrad = translate(ct, ctDict, True, False, ct)
               #termTrad = "'" + ctTrad 
               stringTrad = ''
               translate(ct, ctDict, True, False, ct)
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
    main(sys.argv[1], sys.argv[2])

    print(sys.argv[1])
    print("WARNING: This software needs python >3.6 to run properly")
    # CHECK: source==target doesn't mean untranslated
    numTermTrad = numTerms-numTermsUntrad
    numWordTrad = numWords-numWordsUntrad
    # LaTeX friendly, human unfriendly
    print(str(numTermTrad) + " ("+percentage2d(numTermTrad, numTerms)+"\\%) "+" & "+ str(numTermsUntrad) + " ("+percentage2d(numTermsUntrad, numTerms)+"\\%) "+" & "+ str(numWordTrad) + " ("+percentage2d(numWordTrad, numWords)+"\\%) "+" & "+ str(numWordsUntrad) + " ("+percentage2d(numWordsUntrad, numWords)+"\\%) \\\\")
    #print(str(numTermsUntrad) + " untranslated parts, " + str(numTerms) + " total parts")
    #print(str(numWordsUntrad) + " untranslated words " + str(numWords)+ " total words")
 

# TRUECASE
#python3 tradQueries.py queries.nonEmpty.2trad queries.mesh2.trad
# 144645 (6.6\%)  & 2032976 (93.4\%)  & 2140408 (61.7\%)  & 1325990 (38.3\%) \\
#python3 tradQueries.py queries.nonEmpty.2trad queries.quadLex.trad
#291493 (13.4\%)  & 1886128 (86.6\%)  & 2826779 (81.5\%)  & 639619 (18.5\%) \\
# LOWERCASE
# python3 tradQueries.py queries.nonEmpty.2trad queries.mesh2.lc.trad
# 167152 (7.7\%)  & 2010469 (92.3\%)  & 2225598 (64.2\%)  & 1240800 (35.8\%) \\
# python3 tradQueries.py queries.nonEmpty.2trad queries.quadLex.lc.trad
# 324033 (14.9\%)  & 1853588 (85.1\%)  & 2945959 (85.0\%)  & 520439 (15.0\%) \\

