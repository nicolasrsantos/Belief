#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 12:53:23 2018

@author: nicolas
"""

import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from itertools import cycle
from scipy import interp
from sklearn.metrics import roc_curve, auc


def tokenize(text):
    return nltk.word_tokenize(text)

def removeTrash(text):
    return re.sub("(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|(\d+)", '', text, flags=re.MULTILINE)

def toLower(text):
    return ' '.join(word.lower() for word in text)

def removeStopwords(text, stopwordsFileName):
    stopwords = []

    with open(stopwordsFileName, 'r') as stopwordsFile:
        reader = stopwordsFile.readlines()
        stopwords = [word.strip() for word in reader]

    return ' '.join(word for word in text if word not in stopwords)

def clean_corpus(corpus, stopwords_filepath):
    cleaned_corpus = pd.DataFrame(columns=['Submission', 'Label'], index = range(0, len(corpus)))
    
    for i in range(0, len(corpus)):
        text = toLower(corpus.Submission[i].split())
        text = removeTrash(text)
        text = removeStopwords(text.split(), stopwords_filepath)

        cleaned_corpus.Submission[i] = text
        cleaned_corpus.Label[i] = corpus.Label[i]

    return cleaned_corpus

def displayScores(vectorizer, tfidf_result):
    scores = zip(vectorizer.get_feature_names(), np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    print(len(sorted_scores))
    for item in sorted_scores:
        print("{0:50} Score: {1}".format(item[0], item[1]))

def compute_metrics(y_test, y_pred):
    #print(confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1:", f1_score(y_test, y_pred, average = 'weighted'))
    print("Precision:", precision_score(y_test, y_pred, average = 'weighted'))
    print("Recall:", recall_score(y_test, y_pred, average = 'weighted'))

def convert_label(i):
    if (i == 0):
        return "Believe"
    elif (i == 1):
        return "Disbelieve"
    else:
        return "Neutral"
    
def compute_ROC(n_classes, y_test, y_score):
    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    
    # Finally average it and compute AUC
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Plot all ROC curves
    plt.figure()
# =============================================================================
#     plt.plot(fpr["micro"], tpr["micro"],
#              label='micro-average ROC curve (area = {0:0.2f})'
#                    ''.format(roc_auc["micro"]),
#              color='deeppink', linestyle=':', linewidth=4)
#     
#     plt.plot(fpr["macro"], tpr["macro"],
#              label='macro-average ROC curve (area = {0:0.2f})'
#                    ''.format(roc_auc["macro"]),
#              color='navy', linestyle=':', linewidth=4)
# =============================================================================
    
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        print(i)
        plt.plot(fpr[i], tpr[i], color = color, lw = lw,
                 label = 'ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(convert_label(i), roc_auc[i]))
    
    plt.plot([0, 1], [0, 1], 'k--', lw = lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc = "lower right")
    plt.show()