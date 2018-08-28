#!/usr/bin/env python3 
# -*- coding: utf-8 -*-

""" 
    Training a reranker for word embeddings given an input file with a set of features
    generated by extractFeaturesLexicon.py

    Date: 23.08.2018
    Author: cristinae
"""

import sys
import os.path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")


def preprocessingWrite (df, modelPath):
    '''
     Basic preprocessing for most of ML algorithms: binarisation, normalisation, scaling 
     Saves the scaling model for test usage
    '''

    # convert categorical column in four binary columns, one per language
    df4ML = df.join(pd.get_dummies(df['L2'],prefix='L2'))

    # scale columns
    # there is need for xfb?
    # rankW2 has huge numbers in a wide interval, we should cut and/or move to a log scale
    df4ML['rankW2'] = df4ML['rankW2'].apply(lambda x: 1000 if x>1000 else x)
    #df4ML['rankW2'] = df4ML['rankW2'].apply(lambda x: 0 if x<= 0 else math.log10(x))
	
    #colums2scale = ['rankW2','WEsim','l1','l2','l1/l2','lev','cosSimN2','cosSimN3','cosSimN4','levM2']
    colums2scale = ['rankW2','l1','l2','l1/l2','lev','levM2','simRankt1','simRankWnext','simRankt10','simRankt100']
    scaler = MinMaxScaler()
    df4ML[colums2scale] = scaler.fit_transform(df4ML[colums2scale])

    # save final model
    joblib.dump(scaler, modelPath+'.scaler.pkl') 

    return df4ML



def main(inF, path):

    modelPath = path + '/../../models/reranker/'
    fileName = os.path.basename(inF)
    fileName = os.path.splitext(fileName)[0]  # yes, twice, it has 2 extensions
    baseName = modelPath + '' + os.path.splitext(fileName)[0]
    outModel = baseName +'.model'

    # read original training file
    df = pd.read_csv(inF)
    print (df.head())

    # shuffle all examples to mix all types
    df4ML = df.sample(frac=1)

    # Aply the preprocessing pipeline
    df4ML = preprocessingWrite(df4ML, baseName)


    # Original features
    #feature_cols = ['w1','L1','w2','L2','srcSubUnit','bothBPEmark','WEsim','rankW2','simRankt1','simRankWnext','simRankt10','simRankt100','l1','l2','l1/l2','lev','cosSimN2','cosSimN3','cosSimN4','levM2']
    # Features to use
    feature_cols = ['L2_de','L2_en','L2_es','L2_fr','srcSubUnit','bothBPEmark','WEsim','rankW2','simRankt1','simRankWnext','simRankt10','simRankt100','l1','l2','l1/l2','lev','cosSimN2','cosSimN3','cosSimN4','levM2']
    #feature_cols = ['L2_de','L2_en','L2_es','L2_fr','srcSubUnit','bothBPEmark','WEsim','rankW2','l1','l2','l1/l2','lev','cosSimN2','cosSimN3','cosSimN4','levM2']

    scoring = ['accuracy', 'precision_macro', 'recall_macro']
    X = df4ML.loc[:, feature_cols]
    y = df4ML.Gold
    clf = XGBClassifier(max_depth=6, n_estimators=300, learning_rate=0.05)
    # TODO: fit parameters
    #clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
    #   beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08, hidden_layer_sizes=(8, 2), learning_rate='constant',
    #   learning_rate_init=0.001, max_iter=200, momentum=0.9, nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
    #   solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False, warm_start=True)
    #clf = SVC(kernel='rbf')
    #clf = SVC(kernel='linear')

    scores = cross_validate(clf, X, y, scoring=scoring, cv=10, return_train_score=False)
    #print(scores['test_precision_macro'])
    #print(scores['test_recall_macro'])
    #print(scores['test_accuracy'])

    print("Accuracy: %0.3f (+/- %0.3f)" % (scores['test_accuracy'].mean(), scores['test_accuracy'].std()*2))
    clf.fit(X, y) 
    clf.save_model(outModel)
    ##plt.bar(range(len(clf.feature_importances_)), clf.feature_importances_)
    plot_importance(clf)
    plt.show()

    #thresholds = np.sort(clf.feature_importances_)
    #for thresh in thresholds:
	# select features using threshold
    #    selection = SelectFromModel(clf, threshold=thresh, prefit=True)
    #    select_X = selection.transform(X)
    #    scores = cross_validate(clf, select_X, y, scoring=scoring, cv=5, return_train_score=False)
    #    print("Thresh=%.3f, n=%d, Accuracy: %0.4f (+/- %0.4f)" % (thresh, select_X.shape[1], scores['test_accuracy'].mean(), scores['test_accuracy'].std()*2))



if __name__ == "__main__":
    
    if len(sys.argv) is not 2:
        sys.stderr.write('Usage: python3 %s trainingFile\n' % sys.argv[0])
        sys.exit(1)
    scriptPath = os.path.dirname(os.path.abspath( __file__ ))
    main(sys.argv[1], scriptPath)


