#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 10:32:24 2018

@author: nicolas
"""

import pandas as pd
import utils
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score, KFold, LeaveOneOut, StratifiedKFold

def execute_OneVsRestSVM(X_train, y_train, X_test, y_test):
    classifier = OneVsRestClassifier(SVC(kernel = 'linear', probability = True, random_state = 0))
    classifier.fit(X_train, y_train)
    y_score = classifier.predict_proba(X_test)
    utils.compute_ROC(n_classes, y_test, y_score)
    
def execute_OneVsRestNB(X_train, y_train, X_test, y_test):
    classifier = OneVsRestClassifier(BernoulliNB())
    classifier.fit(X_train, y_train)
    y_score = classifier.predict_proba(X_test)
    utils.compute_ROC(n_classes, y_test, y_score)
    
def execute_SVM(X_train, y_train, X_test, y_test):
    classifier = SVC(kernel = 'linear', random_state = 0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    utils.compute_metrics(y_test, y_pred)
    
def execute_NB(X_train, y_train, X_test, y_test):
    classifier = BernoulliNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    utils.compute_metrics(y_test, y_pred)
    
if __name__ == "__main__":    
    csv_path = "/home/nicolas/Documents/Scripts/Rumour Belief/csv/"
    stopwords_filepath = "/home/nicolas/Documents/Scripts/Rumour Belief/stopwords.txt"
    
    df = pd.read_csv(csv_path + "submissions_with labels.csv")
    X = df.Submission
    y = df.Label
    tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 3, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words = 'english', lowercase='True')
    X = tfidf.fit_transform(X).toarray()
    
    #y = label_binarize(y, classes=['Believe', 'Disbelieve', 'Neutral'])
    #n_classes = y.shape[1]
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    #execute_SVM(X_train, y_train, X_test, y_test)
    #execute_NB(X_train, y_train, X_test, y_test)
    
# =============================================================================
#     loo = LeaveOneOut() 
#     classifier = BernoulliNB()
#     scores = cross_val_score(classifier, X, y, cv = loo)
#     print(np.mean(scores))
# =============================================================================
    
    kf = StratifiedKFold(n_splits = 10, shuffle = True)
    classifier = BernoulliNB()
    scores = cross_val_score(classifier, X, y, cv = kf)
    avg_score = np.mean(scores)
    print(avg_score)
    
    #execute_OneVsRestSVM(X_train, y_train, X_test, y_test)
    #execute_OneVsRestNB(X_train, y_train, X_test, y_test)