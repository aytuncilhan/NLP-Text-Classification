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

def main():

    # generate_v2()
    # all_data = pd.read_pickle("/Users/aytuncilhan/Projects/Clea/alldata_V2.pkl")
    # print(all_data)

    generateTrainData_V2()
    # train_data = pd.read_pickle("/Users/aytuncilhan/Projects/Clea/traindata.pkl")

    clean = pd.read_pickle("/Users/aytuncilhan/Projects/Clea/traindata_V3.pkl")

    # print(clean)

    trainModel(clean)
    # alternative(clean)
    # results = predict(clean, all_data)
    # results.to_pickle("/Users/aytuncilhan/Projects/Clea/final_results.pkl")
    # results = pd.read_pickle("/Users/aytuncilhan/Projects/Clea/final_results.pkl")

    # results.hist(column='prediction')
    # results.plot(kind='hist')
    # results.plot.hist(bins=9)

    # plt.yscale('log', nonposy='clip')
    # plt.show()

    # import openpyxl
    # results.to_excel(r'/Users/aytuncilhan/Projects/Clea/Predictions_13D.xlsx', index = False)

    # print(results)

# NOTE Below are the custom functions used in main NOTE 

# Read all "first_stage" .txt files and extract the "item 4" descriptions to "~/Clea/alldata.pkl"
def generateAllData():

    my_dir_path = "/Users/aytuncilhan/Projects/Clea/first_stage/"
    results = defaultdict(list)
    index = 0
    for file in Path(my_dir_path).iterdir():
        index = index+1
        print("PROGRESS:  " + str(index) + " / 53,799 ")
        with open(file, "r") as f:
            # Accession Number
            size = len(file.name)
            results["AccessionNumber"].append(file.name[:size-4])

            # Item 4 Descrition
            data=[]
            flag=False
            entry=0
            for line in f:
                trystr = line.strip()
                trystr = " ".join(trystr.split())
                trystr = trystr.lower()

                if (('item 4' in trystr or 'item iv' in trystr or 'itemiv' in trystr or 'item4' in trystr) and 
                    ('in item 4' not in trystr and 'of item 4' not in trystr and 'at item 4' not in trystr and 'on item 4' not in trystr and 'to item 4' not in trystr and 'this item 4' not in trystr and 'and item 4' not in trystr and 'also item 4' not in trystr and 'item 4 of' not in trystr and 'see item 4' not in trystr and 'under item 4' not in trystr and 'item for are' not in trystr and 'by item 4' not in trystr and 'item 4(' not in trystr and 'item 4 and' not in trystr and 'date item 4' not in trystr and 'item 4 below' not in trystr and 'item 4 is' not in trystr and 'item 4 hereof' not in trystr and r'item 4\d' not in trystr) ):
                    flag=True
                if ('item 5' in trystr or 'item V' in trystr or 'item5' in trystr or 'itemV' in trystr or
                'item 6' in trystr or 'item VI' in trystr or 'item6' in trystr or 'itemVI' in trystr or
                'item 7' in trystr or 'item VII' in trystr or 'item7' in trystr or 'itemVII' in trystr or
                'item 8' in trystr or 'item VIII' in trystr or 'item8' in trystr or 'itemVIII' in trystr or
                'item 9' in trystr or 'item IX' in trystr or 'item9' in trystr or 'itemIX' in trystr):
                    flag=False
                if flag:
                    entry=entry+1
                    if entry>1: data.append(line)

            clean = ''.join(data)
            results["Description"].append(clean)

    alldata = pd.DataFrame(results)

    # TODO : Preprocess all_data
    alldata['Description'] = alldata['Description'].str.replace('Purpose of Transaction', '')
    alldata['Description'] = alldata['Description'].str.replace('Purpose of the Transaction', '')
    alldata['Description'] = alldata['Description'].str.replace('PURPOSE OF TRANSACTION', '')
    alldata['Description'] = alldata['Description'].str.replace('Purpose of\n  Transaction', '')

    alldata.to_pickle("/Users/aytuncilhan/Projects/Clea/alldata.pkl")

def generate_v2():

    my_dir_path = "/Users/aytuncilhan/Projects/Clea/first_stage/"

    column_names = ["AccessionNumber", "cleaned"]
    results = pd.DataFrame(columns = column_names)
    stemmer = PorterStemmer()
    words = stopwords.words("english")
    index = 0
    for file in Path(my_dir_path).iterdir():
        if os.path.getsize(my_dir_path + '/' + file.name) < 1000000:
            index = index+1
            with open(file, "r") as f:
                temp = f.read()
                temp2 = ''
                for i in re.sub("[^a-zA-Z]", " ", temp).split():
                    if i not in words:
                        temp2 = temp2 + " " + " ".join([stemmer.stem(i)])
                
                start_string1 = 'item purpos of transact'
                start_string2 = 'item purpos transact'
                end_string1 = 'item interest in secur'
                end_string2 = 'item interest secur'

                temp2 = temp2.replace(start_string1 + ' ' + end_string1, '')
                temp2 = temp2.replace(start_string1 + ' ' + end_string2, '')
                temp2 = temp2.replace(start_string2 + ' ' + end_string1, '')
                temp2 = temp2.replace(start_string2 + ' ' + end_string2, '')

                st_index = max( temp2.find(start_string1) , temp2.find(start_string2) )

                if st_index> 0:
                    start_index = st_index + len(start_string2)
                    end_index = max( temp2.find(end_string1) , temp2.find(end_string2) )

                    temp2 = temp2[start_index:end_index]
                    if (index % 200 == 0):
                        print("STEP (" +file.name[:len(file.name)-4] + "): " + str(index))
                else:
                    temp2 = ""
                    if (index % 200 == 0):
                        print("STEP (" +file.name[:len(file.name)-4] + "): " + str(index) + " /// BUT NO MATCH ///")

                results.loc[index] = [file.name[:len(file.name)-4] , temp2]

        else: 
            if (index % 200 == 0):
                print("STEP (" +file.name[:len(file.name)-4] + "): " + str(index) + 
                    ' /// BUT File size limit exceeded (max 1 MB)! File size is: ' + str(os.path.getsize(my_dir_path + '/' + file.name)) + 'KB')

    results.to_csv(r'/Users/aytuncilhan/Projects/Clea/alldata_V2.csv')
    results.to_pickle("/Users/aytuncilhan/Projects/Clea/alldata_V2.pkl")

def generateTrainData():
    
    all_data = pd.read_pickle("/Users/aytuncilhan/Projects/Clea/alldata.pkl")
    train_data = pd.read_excel (r'/Users/aytuncilhan/Projects/Clea/NLP_TrainingData.xlsx', sheet_name='Training Data')

    train_data.drop('Type of Fund', 1, inplace = True)
    train_data.drop('SubjectCompanyName', 1, inplace = True)
    train_data.drop('FilingsCompanyName', 1, inplace = True)
    train_data.drop('How sure? 1-3', 1, inplace = True)

    train = pd.merge(all_data,
                    train_data,
                    on='AccessionNumber',
                    how='inner')

    # Remove "Purpose of the Transaction"
    train['Description'] = train['Description'].str.replace('Purpose of Transaction', '')
    train['Description'] = train['Description'].str.replace('Purpose of the Transaction', '')
    train['Description'] = train['Description'].str.replace('PURPOSE OF TRANSACTION', '')
    train['Description'] = train['Description'].str.replace('Purpose of\n  Transaction', '')

    train['NumberCatergory'].replace({"-": 0}, inplace=True)
    train['NumberCatergory'] = train['NumberCatergory'].astype('int')

    train.to_pickle("/Users/aytuncilhan/Projects/Clea/traindata.pkl")

def generateTrainData_V2():
    
    all_data = pd.read_pickle("/Users/aytuncilhan/Projects/Clea/alldata_V2.pkl")
    train_data = pd.read_excel (r'/Users/aytuncilhan/Projects/Clea/NLP_TrainingData.xlsx', sheet_name='Training Data')

    train_data.drop('Type of Fund', 1, inplace = True)
    train_data.drop('SubjectCompanyName', 1, inplace = True)
    train_data.drop('FilingsCompanyName', 1, inplace = True)
    train_data.drop('How sure? 1-3', 1, inplace = True)

    train = pd.merge(all_data,
                    train_data,
                    on='AccessionNumber',
                    how='inner')

    # train['NumberCatergory'].replace({"-": 0}, inplace=True)
    # train['NumberCatergory'] = train['NumberCatergory'].astype('int')

    train = train[train['NumberCatergory'] != '-']

    train.to_pickle("/Users/aytuncilhan/Projects/Clea/traindata_V3.pkl")

def trainModel(train):
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

def alternative(train_data):

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

    N = 3
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

    model = LinearSVC()
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, train_data.index, test_size=0.42, random_state=0)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    from sklearn.metrics import confusion_matrix
    conf_mat = confusion_matrix(y_test, y_pred)

    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=category_id_df.NameCategory.values, yticklabels=category_id_df.NameCategory.values)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    # plt.show()

    from sklearn import metrics
    print(metrics.classification_report(y_test, y_pred, target_names=train_data['NameCategory'].unique()))


def predict(train, all_data):

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

def predict_v2(train, all_data):

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

    return all_data

if __name__ == '__main__':
    main()