#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 14:21:06 2018

@author: nicolas
"""

import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

if __name__ == "__main__":
    csv_path = "/home/nicolas/Documents/Scripts/Rumour Belief/csv/"
    dataset = pd.read_csv(csv_path + "submissions_with labels.csv", engine='python')
    dataset.head()
    
    col = ['Label', 'Submission']
    dataset = dataset[col]
    dataset = dataset[pd.notnull(dataset['Submission'])]
    
    dataset.columns = ['Label', 'Submission']
    
    dataset['category_id'] = dataset['Label'].factorize()[0]
    category_id_df = dataset[['Label', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'Label']].values)
    dataset.head()
    
    fig = plt.figure(figsize=(8,6))
    dataset.groupby('Label').Submission.count().plot.bar(ylim=0)
    plt.show()
    
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
    features = tfidf.fit_transform(dataset.Submission).toarray()
    labels = dataset.category_id
    features.shape
    
    N = 2
    for Label, category_id in sorted(category_to_id.items()):
      features_chi2 = chi2(features, labels == category_id)
      indices = np.argsort(features_chi2[0])
      feature_names = np.array(tfidf.get_feature_names())[indices]
      unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
      bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
      print("# '{}':".format(Label))
      print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
      print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
      

    X_train, X_test, y_train, y_test = train_test_split(dataset['Submission'], dataset['Label'], random_state = 0)
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    
    clf = MultinomialNB().fit(X_train_tfidf, y_train)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import LinearSVC
    
    from sklearn.model_selection import cross_val_score
    
    models = [
        RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        LinearSVC(),
        MultinomialNB(),
        LogisticRegression(random_state=0),
    ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * len(models)))
    entries = []
    for model in models:
      model_name = model.__class__.__name__
      accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
      for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    
    import seaborn as sns
    
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df, 
                  size=8, jitter=True, edgecolor="gray", linewidth=2)
    plt.show()
    
    model = LinearSVC()

    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, dataset.index, test_size=0.33, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    from sklearn.metrics import confusion_matrix
    
    conf_mat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=category_id_df.Label.values, yticklabels=category_id_df.Label.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    model.fit(features, labels)

    N = 2
    for Label, category_id in sorted(category_to_id.items()):
      indices = np.argsort(model.coef_[category_id])
      feature_names = np.array(tfidf.get_feature_names())[indices]
      unigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 1][:N]
      bigrams = [v for v in reversed(feature_names) if len(v.split(' ')) == 2][:N]
      print("# '{}':".format(Label))
      print("  . Top unigrams:\n       . {}".format('\n       . '.join(unigrams)))
      print("  . Top bigrams:\n       . {}".format('\n       . '.join(bigrams)))
      
      from sklearn import metrics
      print(metrics.classification_report(y_test, y_pred, target_names=dataset['Label'].unique()))