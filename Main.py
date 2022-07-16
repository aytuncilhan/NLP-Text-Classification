import pandas as pd
import numpy as np
import pickle
import pandas as df
from NLPclassifier import NLPclassifier
from DataEngine import DataEngine

class Main:
    def main():

        classifier = NLPclassifier

        ## NOTE: Parse 13D filing text data and create a dataset usable in Pyhton.
        ## NOTE: Already done and data extracted as pickle, no need to run every time so it's commented out.
        # DataEngine.generate_v2()
        # all_data = pd.read_pickle("/Users/aytuncilhan/Documents/Documents/Projects/VC_Analysis/alldata_V2.pkl")

        ## NOTE: Create training dataset. 
        ## NOTE: Already done and data extracted as pickle, no need to run every time so it's commented out.
        # DataEngine.generateTrainData_V2()
        # train_data = pd.read_pickle("/Users/aytuncilhan/Documents/Documents/Projects/VC_Analysis/traindata.pkl")

        # Read the clean training (and testing) dataset
        clean_data = pd.read_pickle("/Users/aytuncilhan/Documents/Documents/Projects/VC_Analysis/traindata_V3.pkl")

        # Run the NLP training and classifier
        classifier.classify_v4(clean_data, classifier.RandomForestClassifier, 0.2)

        ## NOTE: For future use
        # results = NLPclassifier.returnPredictions(clean_data, all_data)
        # results.to_pickle("/Users/aytuncilhan/Documents/Documents/Projects/VC_Analysis/final_results.pkl")

    if __name__ == '__main__':
        main()