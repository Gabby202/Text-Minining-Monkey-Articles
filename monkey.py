# encoding: utf-8
from sklearn import preprocessing
from sklearn.naive_bayes import ComplementNB

import tm_useful_functions as t
import numpy as np
import re
import pandas as pd
import csv
import string
from matplotlib import pyplot as pp
import nltk, re
import urllib3
import nltk, re, math, pandas as pd, numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from pandas.plotting import parallel_coordinates
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split as tt

from sklearn.svm import SVC
from sklearn.naive_bayes import ComplementNB

from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import f1_score as f1
from sklearn.model_selection import cross_val_score as xval
with open('raw_data.csv') as csvfile:
    data = pd.read_csv("C:\\Users\\Gabby\\Documents\\year4_semester2\\Text Analysis\\Assignment\\raw_data.csv")

# reduce size of feature space
#transform the data
data["doc"] = data["doc"].str.lower()
# print(data)

#tokenize data
data_tokens = [nltk.word_tokenize(doc) for doc in data["doc"]]
# print(data_tokens)



#tag and remove POS before stop word removal
tagged_corpus = []
tagged_corpus.append(nltk.pos_tag(doc) for doc in data_tokens)
pos_to_remove = ['DT', 'MD', 'CC', 'IN', 'TO', 'PRP', 'PRP$']

new_tagged_corpus = []
for doc in tagged_corpus:
    new_doc = []
    for pair in doc:
        if pair[1] not in pos_to_remove:
            new_doc.append(pair)
# new_tagged_corpus.append(new_doc)
# print(len(data_tokens))
# print(len(new_doc))
# print(new_doc)


# ## remove custom stop words
stop_words_custom = ['a', 'is', 'are', 'was', 'does', 'do', 'did', 'to', 'a', 'she',
              'he', 'they','we','there', 'here', 'you', 'him', 'the','be',
              'because', 'it', 'so' , 'for', 'with', 'but', 'and', 'should', 'like', 're,aon', 'unlike', 'says'
                     'say, made', 'with', 'for', 'must']
stop_words = stopwords.words('english')
punctuation = ['.', ',', '!', ';', '\"', '\'', ':', ')', '(', '&', '’', '“', '”', "‘", "-"]

data_tokens = [[token for token in sent if token not in stop_words + stop_words_custom + punctuation]
                 for sent in data_tokens]

#stemming
data_tokens = [[nltk.PorterStemmer().stem(str(token)) for token in sent]for sent in data_tokens]


# build bow
## build the unigram model
unigram_bow = t.build_bow(data_tokens)
unigram_bow = unigram_bow.fillna(0)
bow = unigram_bow.copy()
# print(bow)

bow = t.vectorized_build_tfidf_bow(bow)

# print(bow)
# top 3 attributes frequencies
# three_attributes = bow[['chimpanze', 'cultur', 'disappear']]
# print(three_attributes.corr())
# plt.figure()
# three_attributes.plot()
# three_attributes.plot().bar(1, 1)
# plt.show()

#attribute means plot
# stats = bow.describe()
# stats = stats.T
# means = stats['mean']
# means.plot()
# plt.show()

# bow_scaled = preprocessing.scale(bow)
# print(bow_scaled)

#clustering
# cluster_labels = t.clustering_agglomerative(bow)
# print(cluster_labels)
# t.clustering_scatter_plot(bow, 'monkey', 'non_monkey')

labels = data['label'].copy()
# text = data['doc'].copy()


## split both the input/feature set and the output/label set
## at around 3/4 for training and 1/4 for testing
## this function returns 4 datasets:
## 2 for features and labels training set
## 2 for features and labels testing set
labels = list(labels[:142].values) + list(labels[143:].values)
x_train, x_test, y_train, y_test = tt(bow, labels,
                                       test_size=0.25)
## train an NB classifier
nb_model = ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)
nb_model.fit(x_train, y_train)
## use cross validation first on the training data
nb_xval_acc_scores = xval(nb_model, x_train, y_train, cv=5)
nb_xval_f1_scores = xval(nb_model, x_train, y_train, cv=5,scoring='f1_macro')
## average the scores across the 5 folds and get the standard deviation
print("NB X-validation Accuracy: %0.2f (+/- %0.2f)" % (nb_xval_acc_scores.mean(),
                                       nb_xval_acc_scores.std() * 2))
print("NB X-validation F1: %0.2f (+/- %0.2f)" % (nb_xval_f1_scores.mean(),
                                       nb_xval_f1_scores.std() * 2))
## predict the labels of the test_bow row using the trained model
nb_predicted_labels =nb_model.predict(x_test)
print('Labels predicted by the NB model:', nb_predicted_labels)
nb_confus_matr = cm(y_true=y_test, y_pred=nb_predicted_labels)
print('Confusion matrix for the NB model:\n', nb_confus_matr)
nb_acc_score = acc(y_true=y_test, y_pred=nb_predicted_labels)
nb_f1_score = f1(y_true=y_test, y_pred=nb_predicted_labels, pos_label='monkey')
print("NB split-validation Accuracy: %0.2f" % nb_acc_score)
print("NB split-validation F1: %0.2f" % nb_f1_score)

## train an SVM classifier
svm_model = SVC(kernel='linear', C=1.0, random_state=1)
svm_model.fit(x_train, y_train)
## use cross validation first on the training data
svm_xval_acc_scores = xval(svm_model, x_train, y_train, cv=5)
svm_xval_f1_scores = xval(svm_model, x_train, y_train, cv=5,scoring='f1_macro')

## average the scores across the 5 folds and get the standard deviation
print("SVM X-validation Accuracy: %0.2f (+/- %0.2f)" % (svm_xval_acc_scores.mean(),
                                       svm_xval_acc_scores.std() * 2))
print("SVM X-validation F1: %0.2f (+/- %0.2f)" % (svm_xval_f1_scores.mean(),
                                       svm_xval_f1_scores.std() * 2))
## predict the labels of the test_bow row using the trained model
svm_predicted_labels = svm_model.predict(x_test)
print('Labels predicted by the SVM model:', svm_predicted_labels)
svm_confus_matr = cm(y_true=y_test, y_pred=svm_predicted_labels)
print('Confusion matrix for the NB model:\n', svm_confus_matr)
svm_acc_score = acc(y_true=y_test, y_pred=svm_predicted_labels)
svm_f1_score = f1(y_true=y_test, y_pred=svm_predicted_labels, pos_label='monkey')
print("SVM split-validation Accuracy: %0.2f" % svm_acc_score)
print("SVM split-validation F1: %0.2f" % svm_f1_score)

t.classification_visualizaion(x_scores, xval_labels, split_scores,split_labels,
                                false_vals, false_labels, true_vals,true_labels)
