#!/usr/bin/env python3 -W ignore::DeprecationWarning
# -*- coding: utf-8 -*-


""" 
    
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

def main(inF):

    outF = inF+'.model'
    # read original training file
    df = pd.read_csv(inF)
    print (df.head())

    # shuffle all examples to mix all types
    df4ML = df.sample(frac=1)
    # convert categorical column in four binary columns, one per language
    df4ML = df4ML.join(pd.get_dummies(df['L2'],prefix='L2'))

    # Original features
    #feature_cols = ['w1','L1','w2','L2','srcSubUnit','bothBPEmark','WEsim','rankW2','simRankt1','simRankWnext','simRankt10','simRankt100','l1','l2','l1/l2','lev','cosSimN2','cosSimN3','cosSimN4','levM2']
    # Features to use
    feature_cols = ['L2_de','L2_en','L2_es','L2_fr','srcSubUnit','bothBPEmark','WEsim','rankW2','simRankt1','simRankWnext','simRankt10','simRankt100','l1','l2','l1/l2','lev','cosSimN2','cosSimN3','cosSimN4','levM2']
    #feature_cols = ['L2_de','L2_en','L2_es','L2_fr','srcSubUnit','bothBPEmark','WEsim','rankW2','l1','l2','l1/l2','lev','cosSimN2','cosSimN3','cosSimN4','levM2']



    # scale columns
    # rankW2 has huge numbers in a wide interval, we should cut and/or move to a log scale
    df4ML['rankW2'] = df4ML['rankW2'].apply(lambda x: 1000 if x>1000 else x)
    #df4ML['rankW2'] = df4ML['rankW2'].apply(lambda x: 0 if x<= 0 else math.log10(x))
	
    #colums2scale = ['rankW2','WEsim','l1','l2','l1/l2','lev','cosSimN2','cosSimN3','cosSimN4','levM2']
    colums2scale = ['rankW2','l1','l2','l1/l2','lev','levM2','simRankt1','simRankWnext','simRankt10','simRankt100']
    scaler = MinMaxScaler()
    df4ML[colums2scale] = scaler.fit_transform(df4ML[colums2scale])

    scoring = ['accuracy', 'precision_macro', 'recall_macro']
    X = df4ML.loc[:, feature_cols]
    y = df4ML.Gold
    clf = XGBClassifier(max_depth=6, n_estimators=300, learning_rate=0.05)
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
    #print("Precision: %0.3f (+/- %0.3f)" % (scores['test_precision_macro'].mean(), scores['test_precision_macro'].std()*2))
    clf.fit(X, y) 
    clf.save_model('prova.model')
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
    main(sys.argv[1])



    # fearure importance for SVMlinear
    #rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(2),
    #          scoring='accuracy')
    #rfecv.fit(X, y)
    #print("Optimal number of features : %d" % rfecv.n_features_)
    #print(rfecv.ranking_)
    # Plot number of features VS. cross-validation scores
    #plt.figure()
    #plt.xlabel("Number of features selected")
    #plt.ylabel("Cross validation score (nb of correct classifications)")
    #plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    #plt.show()

'''
Thresh=0.003, n=20, Accuracy: 0.8712 (+/- 0.0104)
Thresh=0.003, n=20, Precision: 0.8699 (+/- 0.0113)
Thresh=0.004, n=19, Accuracy: 0.8714 (+/- 0.0068)
Thresh=0.004, n=19, Precision: 0.8701 (+/- 0.0073)
Thresh=0.006, n=18, Accuracy: 0.8721 (+/- 0.0070)
Thresh=0.006, n=18, Precision: 0.8708 (+/- 0.0075)
Thresh=0.009, n=16, Accuracy: 0.8692 (+/- 0.0088)
Thresh=0.009, n=16, Precision: 0.8679 (+/- 0.0095)
Thresh=0.011, n=15, Accuracy: 0.8712 (+/- 0.0077)
Thresh=0.011, n=15, Precision: 0.8700 (+/- 0.0080)
Thresh=0.012, n=14, Accuracy: 0.8706 (+/- 0.0068)
Thresh=0.012, n=14, Precision: 0.8693 (+/- 0.0074)
Thresh=0.019, n=13, Accuracy: 0.8696 (+/- 0.0050)
Thresh=0.019, n=13, Precision: 0.8685 (+/- 0.0060)
Thresh=0.023, n=12, Accuracy: 0.8709 (+/- 0.0054)
Thresh=0.023, n=12, Precision: 0.8698 (+/- 0.0061)
Thresh=0.027, n=11, Accuracy: 0.8696 (+/- 0.0055)
Thresh=0.027, n=11, Precision: 0.8684 (+/- 0.0067)
Thresh=0.034, n=10, Accuracy: 0.8687 (+/- 0.0060)
Thresh=0.034, n=10, Precision: 0.8674 (+/- 0.0064)
Thresh=0.043, n=9, Accuracy: 0.8687 (+/- 0.0077)
Thresh=0.043, n=9, Precision: 0.8673 (+/- 0.0083)
Thresh=0.053, n=8, Accuracy: 0.8674 (+/- 0.0109)
Thresh=0.053, n=8, Precision: 0.8658 (+/- 0.0115)
Thresh=0.062, n=7, Accuracy: 0.8682 (+/- 0.0072)
Thresh=0.062, n=7, Precision: 0.8666 (+/- 0.0071)
Thresh=0.068, n=6, Accuracy: 0.8642 (+/- 0.0083)
Thresh=0.068, n=6, Precision: 0.8625 (+/- 0.0085)
Thresh=0.073, n=5, Accuracy: 0.8622 (+/- 0.0067)
Thresh=0.073, n=5, Precision: 0.8607 (+/- 0.0069)
Thresh=0.085, n=4, Accuracy: 0.8450 (+/- 0.0108)
Thresh=0.085, n=4, Precision: 0.8442 (+/- 0.0108)
Thresh=0.099, n=3, Accuracy: 0.8085 (+/- 0.0114)
Thresh=0.099, n=3, Precision: 0.8116 (+/- 0.0118)
Thresh=0.156, n=2, Accuracy: 0.7579 (+/- 0.0113)
Thresh=0.156, n=2, Precision: 0.7605 (+/- 0.0121)
Thresh=0.208, n=1, Accuracy: 0.6871 (+/- 0.0248)
Thresh=0.208, n=1, Precision: 0.6845 (+/- 0.0248)
'''
