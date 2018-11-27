# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 04:35:05 2018

@author: nicolas
"""

import pandas as pd

csv_path = "D:\\Dropbox\\Códigos\\python\\machine learning\\Belief\\csv\\"
consumer_complaints_df = pd.read_csv(csv_path + "submissions_with labels.csv", engine = 'python')
consumer_complaints_filtered_df = consumer_complaints_df[pd.notnull(consumer_complaints_df['Submission'])]

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,6))
df = consumer_complaints_filtered_df[['Label','Submission']]
df.groupby('Label').count().plot.bar(ylim=0)
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

labels = df['Label']
text = df['Submission']

X_train, X_test, y_train, y_test = train_test_split(text, labels, random_state=0, test_size=0.3)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tf_transformer = TfidfTransformer().fit(X_train_counts)
X_train_transformed = tf_transformer.transform(X_train_counts)

X_test_counts = count_vect.transform(X_test)
X_test_transformed = tf_transformer.transform(X_test_counts)

labels = LabelEncoder()
y_train_labels_fit = labels.fit(y_train)
y_train_lables_trf = labels.transform(y_train)

print(labels.classes_)

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

linear_svc = LinearSVC()
clf = linear_svc.fit(X_train_transformed,y_train_lables_trf)

calibrated_svc = CalibratedClassifierCV(base_estimator=linear_svc,
                                        cv="prefit")

calibrated_svc.fit(X_train_transformed,y_train_lables_trf)
predicted = calibrated_svc.predict(X_test_transformed)
    
print('Average accuracy on test set={}'.format(np.mean(predicted == labels.transform(y_test))))