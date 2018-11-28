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