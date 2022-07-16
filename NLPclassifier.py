from pprint import PrettyPrinter
import nltk
from nltk.classify.naivebayes import NaiveBayesClassifier
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sqlite3 import Error
from sklearn.ensemble import RandomForestClassifier
import pickle
import glob
from collections import defaultdict
from pathlib import Path
import pandas as df
import re
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.sparsefuncs import min_max_axis
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import os

class NLPclassifier:

    RandomForestClassifier = RandomForestClassifier()
    LinearSVC = LinearSVC()
    KNeighborsClassifier = KNeighborsClassifier()
    SGDClassifier = SGDClassifier()

    def classify_v1(train):
        vectorizer = TfidfVectorizer(min_df= 10, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))
        final_features = vectorizer.fit_transform(train['cleaned']).toarray()
        final_features.shape

        #first we split our dataset into testing and training set:
        # this block is to split the dataset into training and testing set 
        X = train['cleaned']
        Y = train['NumberCatergory']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.35)

        # instead of doing these steps one at a time, we can use a pipeline to complete them all at once
        pipeline = Pipeline([('vect', vectorizer),
                            ('chi',  SelectKBest(chi2, k='all')),
                            ('clf', RandomForestClassifier())])

        # fitting our model and save it in a pickle for later use
        model = pipeline.fit(X_train, y_train)

        # with open('RandomForest.pickle', 'wb') as f:
        #     pickle.dump(model, f)
        
        ytest = np.array(y_test)

        # confusion matrix and classification report(precision, recall, F1-score)
        print(classification_report(ytest, model.predict(X_test)))
        print(confusion_matrix(ytest, model.predict(X_test)))

        print("accuracy score: " + str(model.score(X_test, y_test)))

    def returnPredictions(train, all_data):

        vectorizer = TfidfVectorizer(min_df= 5, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))

        #first we split our dataset into testing and training set:
        # this block is to split the dataset into training and testing set 
        X = train['cleaned']
        Y = train['NumberCatergory']
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.001)

        # instead of doing these steps one at a time, we can use a pipeline to complete them all at once
        pipeline = Pipeline([('vect', vectorizer),
                            ('chi',  SelectKBest(chi2, k='all')),
                            ('clf', RandomForestClassifier())])

        # fitting our model and save it in a pickle for later use
        model = pipeline.fit(X_train, y_train)
        results = model.predict( all_data["cleaned"] )
        all_data["prediction"] = results
        return all_data

    def classify_v3(train):

        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

        model = LinearSVC()
        features = tfidf.fit_transform(train.cleaned).toarray()
        labels = train.NumberCatergory
        features.shape

        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, train.index, test_size=0.42, random_state=0)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        X = train['cleaned']
        Y = train['NumberCatergory']

        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, train.index, test_size=0.42, random_state=0)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    def classify_v4(train_data, selectedModel, testSlice):

        train_data = train_data.dropna()
        print('\n\nTRAIN DATA VALUE COUNTS: ')
        print(train_data['NameCategory'].value_counts())
        print("\n\n")

        category_id_df = train_data[['NameCategory', 'NumberCatergory']].drop_duplicates().sort_values('NumberCatergory')
        category_to_id = dict(category_id_df.values)

        tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

        features = tfidf.fit_transform(train_data.cleaned).toarray()
        labels = train_data.NumberCatergory
        features.shape

        # N = 3
        #for NameCat, NumCat in sorted(category_to_id.items()):
        #    features_chi2 = chi2(features, labels == NumCat)
    #
        #    indices = np.argsort(features_chi2[0])
        #    feature_names = np.array(tfidf.get_feature_names())[indices]
    #
        #    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        #    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        #    print(" > '{}':".format(NameCat))
        #    print("  . Unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        #    print("  . Bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))
        #    print("   ")

        model = selectedModel
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, train_data.index, test_size=testSlice, random_state=0)
        y_train = y_train.astype('int')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        from sklearn.metrics import confusion_matrix
        conf_mat = confusion_matrix(list(y_test.values), y_pred)

        #import seaborn as sns

        #fig, ax = plt.subplots(figsize=(15,15))
        #sns.heatmap(conf_mat, annot=True, fmt='d',
        #        xticklabels=category_id_df.NameCategory.values, yticklabels=category_id_df.NameCategory.values)
        #import textwrap
        #f = lambda x: textwrap.fill(x.get_text(), 12) 
        #ax.set_yticklabels(map(f, ax.get_yticklabels()))
        #ax.set_xticklabels(map(f, ax.get_xticklabels()))

        #plt.ylabel('Actual')
        #plt.xlabel('Predicted')
        #plt.show()

        from sklearn import metrics
        print(metrics.classification_report(list(y_test.values), y_pred, target_names=train_data['NameCategory'].unique()))
