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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

csv_path = "/home/nicolas/Documents/Scripts/Rumour Belief/csv/"
stopwords_filepath = "/home/nicolas/Documents/Scripts/Rumour Belief/stopwords.txt"

df = pd.read_csv(csv_path + "submissions_with labels.csv")
#df = utils.clean_corpus(df, stopwords_filepath)

X = df.Submission
y = df.Label

tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 3, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words = 'english', lowercase='True')
X = tfidf.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average = 'weighted'))
print(precision_score(y_test, y_pred, average = 'weighted'))
print(recall_score(y_test, y_pred, average = 'weighted'))