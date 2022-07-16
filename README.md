# Venture Capital Investment Activity Classifier

## 1. Introduction

The aim for this project is to automate the process of classifying the purpose of Venture Capital investments under the 8 predefined categories (listed in the table below).

The [Natural Language Toolkit (nltk)](https://www.nltk.org) and [Scikit-learn](https://scikit-learn.org/stable/) library is used to implement the Natural Language Processing model to accurately classify investment purposes, utilizing the extracted data from [13D filings](https://en.wikipedia.org/wiki/Schedule_13D). As classifier models, the following are used and their performances are analyzed:

**1. Random Forest**, <br/>
**2. Stochastic Gradient Descent**, <br/>
**3. Multinomial Naive Bayesian**, <br/>
**4. Linear Support Vector Classifier (LinearSVC)**, <br/>
**5. K-nearest Neighbor.**

In the next section, the acquired and processed dataset used in this project is introfuced. Then the classifier performances are presented. The document ends with Leassons Learned and Conslusions.

## 2. About the Data

<img align="right" src="https://raw.githubusercontent.com/aytuncilhan/VC-Investment-Analysis/main/AnalysisResults/DatasetOccurence.png" alt="My Image" width="500">
The 13D filings were initially manually labeled and the training (and testing) dataset was created. Due to data privacy reasons, the respective datasets are not presented in the repository. However, data can be shared upon request with the permission of the originator(s).
<br/><br/>
On the right, you can see the distribution of the labels in the training dataset.

<br/><br/><br/><br/><br/><br/><br/><br/>

## 3. Classifiers

Having allocated 20% of the dataset at random for testing, each classifier output is analysed with precision, recall, f1-score and support Key Performance Indicators. In addition, the confusion matrix for each classifier is presented in a heatmap format.

### 3.1. Random Forest Classifier
<img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/RandomForrest/Report_RF_20.png" width="500"><br/><img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/RandomForrest/Heatmap_RF_20.png" width="500">

The Random Forest Classifier is the highest performing one among all others.

<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

### 3.2. Stochastic Gradient Descent
<img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/StochasticGradientDescent/Report_SGD_20.png" width="500"><br/><img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/StochasticGradientDescent/Heatmap_SGD_20.png" width="500">

The Random Forest Classifier is the highest performing one among all others.

<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

### 3.3. Multinomial Naive Bayesian
<img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/RandomForrest/Report_RF_20.png" width="500"><br/><img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/RandomForrest/Heatmap_RF_20.png" width="500">

The Random Forest Classifier is the highest performing one among all others.

<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

### 3.4. Linear Support Vector Classifier (LinearSVC)
<img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/LinearSVC/Report_LSVC_20.png" width="500"><br/><img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/LinearSVC/Heatmap_LSVC_20.png" width="500">

The Random Forest Classifier is the highest performing one among all others.

<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

### 3.5. K-nearest Neighbor (Support Vector Classifier)
<img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/KNearestNeighbor/Report_KNN_20.png" width="500"><br/><img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/KNearestNeighbor/Heatmap_KNN_20.png" width="500">

The Random Forest Classifier is the highest performing one among all others.

<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

## Analysis and Conclusion

Lorem ipsum dolor sit amet
