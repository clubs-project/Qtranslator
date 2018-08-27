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
    - (6) embeddings: cosine similarity, rank, distances in similarities to certain ranks 
          (WEsim,rankW2,simRankt1,simRankWnext,simRankt10,simRankt100)
    - (3) character n-gram similarity (cosSimN2,cosSimN3,cosSimN4)
    - (1) Levenshtein distance between Metaphone 2 phonetic keys (levM2)
    
    Date: 22.08.2018
    Author: cristinae
"""

import sys
import os.path
import numpy as np 

import load
import easyBPE
import features

from gensim.models import KeyedVectors

bpeMark = '@@'
emptyMark = 'EMPTY'
header = 'Gold,w1,L1,w2,L2,srcSubUnit,bothBPEmark,WEsim,rankW2,simRankt1,simRankWnext,simRankt10,simRankt100,l1,l2,l1/l2,lev,cosSimN2,cosSimN3,cosSimN4,levM2\n'

# for debugging
countUp = 0 
countDown = 0

def findSimsNonTrad(w1, w2, l2, proc):
    '''
    Find a word nonTrad which is close to the true translation of w1, w2, but it is not.
    Currently we use the word that is +1 or -1 in the cosine similarity ranking. 
    For top1 translations we use top2 as negative example; for top5 traslations we consider 
    top1 a negative example instead of the closest one.
    Probably good for margin-based algorithms, but for others, should it be random?
    '''

    global countUp 
    global countDown
    noSimsRank = '0,0,0,0,0,0,'
    #WEsim,rankW2,simRankt1,simRankWnext,simRankt10,simRankt100

    if w1 not in proc.embeddingL1.vocab:
       return noSimsRank,emptyMark,noSimsRank

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
    maxSize = len(newSpace.vocab)
    # we look for a non-translation 
    if w2 in newSpace.vocab:
       w2Rank = newSpace.rank(w1,w2)
       if w2Rank>20000:  # we are not going to rerank so many and introduces errors
          return noSimsRank,emptyMark,noSimsRank
       if w2Rank == 1:           # if w1 was the top1 we can only do rank+1
          rank = 2
          countUp = countUp+1
       #elif w2Rank < 6:         # for top5 we give top1 as negative example
       #   rank = 1
       #   countDown = countDown+1
       else:                     # if not, we can randomly select rank +-1
                                 # we give higher probability to -1 to get a balanced corpus
          sumVal = np.random.choice([-1, 1], size=1, p=[5./6, 1./6])
          rank = w2Rank + sumVal[0]
          if sumVal[0]==1:
             countUp = countUp+1
          if sumVal[0]==-1:
             countDown = countDown+1
 
       # we look for the rank_th word to extract the token of the non-translation
       #toprank = newSpace.similar_by_vector(vector,topn=rank)
       if rank > w2Rank:
          tmp = rank
       else:
          tmp = w2Rank
       explore=tmp+100
       if (explore > maxSize-1):
           explore = maxSize-1
       if (tmp+10 > maxSize-1):
           explore = maxSize-1
  
       toprank = newSpace.similar_by_vector(vector,topn=explore)
       #print("out lenToprank1: "+str(len(toprank))+" "+str(len(newSpace.vocab)))
       #toprank2 = newSpace.similar_by_word(w1,topn=explore)
       # does not keep track of the index of a new added word
       if rank <= len(toprank):
          nonTrad = toprank[rank-1][0]
       else:   # This should not happen but happens without the cut at 20000 (error above)
          nonTrad = toprank[len(toprank)-1][0]
          print("kk")
         
       # since all calculations are ready here we use them to estimate WE-related features
       # for the negative example
       sim = newSpace.similarity(w1,nonTrad)
       simRankt1 = toprank[rank-1][1] - toprank[0][1]  # how far in similarity is noTrad to top1
       simRanktnext = toprank[rank-1][1] - toprank[rank][1]  # how far in similarity is noTrad to the next word
       simRankt10 = toprank[rank-1][1] - toprank[9][1] # how far in similarity is noTrad to top10
       simRankt100 = toprank[rank-1][1] - toprank[99][1] # how far in similarity is noTrad to top100
       simsRankNoTrad = str(sim)+','+str(rank)+','+str(simRankt1)+','+str(simRanktnext)+','+str(simRankt10)+','+str(simRankt100)+','
       # for the positive example
       sim = newSpace.similarity(w1,w2)
       simRankt1 = toprank[w2Rank-1][1] - toprank[0][1]  # how far in similarity is w2 to top1
       simRanktnext = toprank[w2Rank-1][1] - toprank[w2Rank][1]  # how far in similarity is w2 to the next word
       simRankt10 = toprank[w2Rank-1][1] - toprank[9][1] # how far in similarity is w2 to top10
       simRankt100 = toprank[w2Rank-1][1] - toprank[99][1] # how far in similarity is w2 to top100
       simsRankW2 = str(sim)+','+str(w2Rank)+','+str(simRankt1)+','+str(simRanktnext)+','+str(simRankt10)+','+str(simRankt100)+','
       
    else:
       return noSimsRank,emptyMark,noSimsRank

    # cleaning
    newSpace = None

    return simsRankW2, nonTrad, simsRankNoTrad



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
                  # and retrieve semantic features for them
                  simsRankW2, tNeg, simsRankNoTrad = findSimsNonTrad(source_word, t, l, proc)
                  # basic and semantic features features  
                  bothBPE = '0'
                  pairPos = features.basicFeatures(source_word,src_lang,t,l,isWord,bothBPE)
                  # extraction of lexical features
                  featsPos = features.extractLexFeatures(source_word, t)
                  entriesOutput = entriesOutput +"1,"+ pairPos + simsRankW2 + featsPos +"\n"
                  # add features for a negative example in case there is one 
                  if tNeg is not "EMPTY":
                     pairNeg = features.basicFeatures(source_word,src_lang,tNeg,l,isWord,bothBPE)
                     featsNeg = features.extractLexFeatures(source_word, tNeg)
                     entriesOutput = entriesOutput +"0,"+ pairNeg + simsRankNoTrad + featsNeg +"\n" 
               # if the input token has been bped do the same for all the subunits
               else:
                  for subunit_src, subunit_tgt in zip(units, tunits):
                      bothBPE = '0'
                      if bpeMark in subunit_src and bpeMark in subunit_tgt:
                         bothBPE = '1'
                      simsRankW2, tNeg, simsRankNoTrad = findSimsNonTrad(subunit_src, subunit_tgt, l, proc)
                      pairPos = features.basicFeatures(subunit_src,src_lang,subunit_tgt,l,isSubword,bothBPE)
                      featsPos = features.extractLexFeatures(subunit_src, subunit_tgt)
                      entriesOutput = entriesOutput +"1,"+ pairPos + simsRankW2 + featsPos +"\n" 
                      if tNeg is not "EMPTY":
                         pairNeg = features.basicFeatures(subunit_src,src_lang,tNeg,l,isSubword,bothBPE)
                         featsNeg = features.extractLexFeatures(subunit_src, tNeg)
                         entriesOutput = entriesOutput +"0,"+ pairNeg + simsRankNoTrad + featsNeg +"\n" 

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

