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
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import Normalizer, MinMaxScaler
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
    print("SVM results")
    utils.compute_metrics(y_test, y_pred)
    
def execute_NB(X_train, y_train, X_test, y_test):
    classifier = BernoulliNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    print("NB results")
    utils.compute_metrics(y_test, y_pred)

def execute_KNN(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier(n_neighbors = 19)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)  
    print("KNN results")
    utils.compute_metrics(y_test, y_pred)
    #params_knn = {'n_neighbors': np.arange(1, 25)} 
    #knn_gs = GridSearchCV(knn, params_knn, cv=5)
    #knn_gs.fit(X_train, y_train)         
    #knn_best = knn_gs.best_estimator_
   # print(knn_gs.best_params_)

def execute_voting_classifier(X, y):
    clf1 = BernoulliNB()
    clf2 = SVC(probability = True, kernel = 'sigmoid', random_state = 0)
    clf3 = KNeighborsClassifier(n_neighbors = 2)
    eclf = VotingClassifier(estimators=[('nb', clf1), ('svm', clf2), ('knn', clf3)], voting='soft', weights = [2, 2, 1])
    kf = StratifiedKFold(n_splits = 10, shuffle = True)
   
    for clf, label in zip([clf1, clf2, clf3, eclf], ['nb', 'svm', 'knn', 'Ensemble']):
        scores = cross_val_score(clf, X, y, cv = kf, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
        
if __name__ == "__main__":    
    csv_path = "D:\\Code\\Workspace\\Belief\\csv\\"
    df = pd.read_csv(csv_path + "data.csv")
    df = pd.get_dummies(df, columns = ['Sentiment', 'Type'])
    
    # Moving the feature 'Sarcasm score' to the range 0 < x < 1
    sarc_score = df['Sarcasm score'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    df['Sarcasm score'] = scaler.fit_transform(sarc_score)
    
    # Feature normalization
    features = df.iloc[:, 3:]
    transformer = Normalizer().fit(features)
    normalized_features = transformer.transform(features)
    
    # TF-IDF    
    tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 3, norm='l2', encoding='utf-8', ngram_range=(1, 2), stop_words = 'english', lowercase='True')
    tfidf_matrix = tfidf.fit_transform(df.Submission).toarray()
    X = np.concatenate((tfidf_matrix, normalized_features), axis = 1)
    y = df.Label
    
# =============================================================================
#     y = label_binarize(y, classes=['Believe', 'Disbelieve', 'Neutral'])
#     n_classes = y.shape[1]
# =============================================================================
    
# =============================================================================
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#     execute_SVM(X_train, y_train, X_test, y_test)
#     execute_NB(X_train, y_train, X_test, y_test)
#     execute_KNN(X_train, y_train, X_test, y_test)
#     
# =============================================================================
    execute_voting_classifier(X, y)
# =============================================================================
#     loo = LeaveOneOut() 
#     classifier = BernoulliNB()
#     scores = cross_val_score(classifier, X, y, cv = loo)
#     print(np.mean(scores))
# =============================================================================
    
# =============================================================================
#     kf = StratifiedKFold(n_splits = 10, shuffle = True)
#     classifier = KNeighborsClassifier(n_neighbors = 19)
#     scores = cross_val_score(classifier, X, y, cv = kf)
#     avg_score = np.mean(scores)
#     print("%0.2f" % avg_score)
# =============================================================================
    
    #execute_OneVsRestSVM(X_train, y_train, X_test, y_test)
   #execute_OneVsRestNB(X_train, y_train, X_test, y_test)