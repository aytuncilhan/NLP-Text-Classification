import nltk
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sqlite3 import Error
import pickle
import glob
from collections import defaultdict
from pathlib import Path
import pandas as df
import re
import os

class DataEngine:

    # Read all "first_stage" .txt files and extract the "item 4" descriptions to "/Users/aytuncilhan/Documents/Documents/Projects/VC_Analysis/alldata.pkl"
    def generateAllData():

        # TODO: The Data for 13D filing smus
        my_dir_path = "/Users/aytuncilhan/Documents/Documents/Projects/VC_Analysis/DATA_TO_BE_ADDED/"
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

        alldata.to_pickle("/Users/aytuncilhan/Documents/Documents/Projects/VC_Analysis/alldata.pkl")

    def generate_v2():

        my_dir_path = "/Users/aytuncilhan/Documents/Documents/Projects/VC_Analysis/"

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

        results.to_csv(r'/Users/aytuncilhan/Documents/Documents/Projects/VC_Analysis/alldata_V2.csv')
        results.to_pickle("/Users/aytuncilhan/Documents/Documents/Projects/VC_Analysis/alldata_V2.pkl")



    ## TRAINING DATA GENERATION ##
    def generateTrainData():
        
        all_data = pd.read_pickle("/Users/aytuncilhan/Documents/Documents/Projects/VC_Analysis/alldata.pkl")
        train_data = pd.read_excel (r'/Users/aytuncilhan/Documents/Documents/Projects/VC_Analysis/NLP_TrainingData.xlsx', sheet_name='Training Data')

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

        train.to_pickle("/Users/aytuncilhan/Documents/Documents/Projects/VC_Analysis/traindata.pkl")

    def generateTrainData_V2():
        
        all_data = pd.read_pickle("/Users/aytuncilhan/Documents/Documents/Projects/VC_Analysis/alldata_V2.pkl")
        train_data = pd.read_excel (r'/Users/aytuncilhan/Documents/Documents/Projects/VC_Analysis/NLP_TrainingData.xlsx', sheet_name='Training Data')

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

        train.to_pickle("/Users/aytuncilhan/Documents/Documents/Projects/VC_Analysis/traindata_V3.pkl")