# Venture Capital Investment Activity Classifier

## 1. Introduction

The aim for this project is to automate the process of classifying the purpose of Venture Capital investments under the 8 predefined categories (listed in the table below).

The [Natural Language Toolkit (nltk)](https://www.nltk.org) and [Scikit-learn](https://scikit-learn.org/stable/) library is used to implement the Natural Language Processing model to accurately classify investment purposes, utilizing the extracted data from [13D filings](https://en.wikipedia.org/wiki/Schedule_13D). As classifier models, the following are used and their performances are analyzed:

**1. Random Forest**, <br/>
**2. Linear Support Vector Classifier (LinearSVC)**, <br/>
**3. Stochastic Gradient Descent**, <br/>
**4. K-nearest Neighbor**,<br/>
**5. Multinomial Naive Bayesian**.<br/>

In the next section, the acquired and processed dataset used in this project is introfuced. Then the classifier performances are presented. The document ends with Leassons Learned and Conslusions.

## 2. About the Data

<img align="right" src="https://raw.githubusercontent.com/aytuncilhan/VC-Investment-Analysis/main/AnalysisResults/DatasetOccurence.png" alt="My Image" width="400">
The 13D filings were manually labeled to create the training (and testing) dataset. *Due to data privacy reasons, the respective datasets are not presented in the repository. However, data can be shared upon request with the permission of the originator(s).*
<br/><br/>
On the right, you can see the distribution of the labels in the training dataset.
<br/><br/>
In the project, *DataEngine.py* provides the pre-processing of raw data until it gets ready for the training models. After reading the .txt files into a pandas dataframe, we use NLTK stemmer to get stems of each word and use stopwords to remove unnecessary words from the dataset.

```
stemmer = PorterStemmer()
words = stopwords.words("english")

for i in re.sub("[^a-zA-Z]", " ", temp).split():
    if i not in words:
        temp2 = temp2 + " " + " ".join([stemmer.stem(i)])
```        

## 3. Classifiers

Having allocated 20% of the dataset at random for testing, each classifier output is analysed with precision, recall, f1-score and support Key Performance Indicators. In addition, the confusion matrix for each classifier is presented in a heatmap format.
<br/><br/>

### 3.1. Random Forest Classifier
<img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/RandomForrest/Report_RF_20.png" width="500"><img align="right"  align="top" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/RandomForrest/Heatmap_RF_20.png" width="500">
* Together with LinearSVC, the Random Forest Classifier performed very well one among other classifiers in this project. 
* 68% overall accuracy is a really good achievement. With no classifier, the success rate would be 12,5% as there are 8 categories. Hence, the Rabdom Forest Classifier achieved more than 5 times improvement compared to baseline.

* As mentioned in this [Towards AI article](https://towardsai.net/p/machine-learning/why-choose-random-forest-and-not-decision-trees), pros of Random Forest include,
  * Robust to outliers: Since the text data is highly noisy, this is an important feature for this project.
  * Works well with non-linear data.
  * Lower risk of overfitting: No overfitting was observed.
  * Better accuracy than other classification algorithms: This was practically proved in this project (LinearSVC has also done a good job but has lower precision).

* Cons of Random Forest include,
  * Random forests are found to be biased while dealing with categorical variables: No bias is examined for this case.
  * Slow Training: This was not the case since our dataset was relatively small.

<br/><br/>

### 3.2. Linear Support Vector Classifier (LinearSVC)
<img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/LinearSVC/Report_LSVC_20.png" width="500"><br/><img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/LinearSVC/Heatmap_LSVC_20.png" width="500">

The Random Forest Classifier is the highest performing one among all others.

<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

### 3.3. Stochastic Gradient Descent
<img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/StochasticGradientDescent/Report_SGD_20.png" width="500"><br/><img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/StochasticGradientDescent/Heatmap_SGD_20.png" width="500">

The Random Forest Classifier is the highest performing one among all others.

<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

### 3.4. K-nearest Neighbor (Support Vector Classifier)
<img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/KNearestNeighbor/Report_KNN_20.png" width="500"><br/><img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/KNearestNeighbor/Heatmap_KNN_20.png" width="500">

The Random Forest Classifier is the highest performing one among all others.

<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

### 3.5. Multinomial Naive Bayesian
<img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/MultinomialNaiveBayes/Report_MNB_20.png" width="500"><br/><img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/MultinomialNaiveBayes/Heatmap_MNB_20.png" width="500">

The Random Forest Classifier is the highest performing one among all others.

<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

## Analysis and Conclusion

Lorem ipsum dolor sit amet
