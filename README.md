# Venture Capital Investment Activity Classifier

## 1. Introduction

The aim for this project is to automate the process of classifying the purpose of Venture Capital investments under the 8 predefined categories (listed in the table below).

The [Natural Language Toolkit (nltk)](https://www.nltk.org) and [Scikit-learn](https://scikit-learn.org/stable/) library is used to implement a Natural Language Processing model and run Supervised Machine Learning to accurately classify investment purposes, utilizing the extracted data from [13D filings](https://en.wikipedia.org/wiki/Schedule_13D). As classifier models, the following are used and their performances are analyzed:

**1. Random Forest**, <br/>
**2. Linear Support Vector Classifier (LinearSVC)**, <br/>
**3. Stochastic Gradient Descent**, <br/>
**4. K-nearest Neighbor**,<br/>
**5. Multinomial Naive Bayesian**.<br/>

In the next section, the acquired and processed dataset used in this project is introfuced. Then the classifier performances are presented. The document ends with Leassons Learned and Conslusions.

## 2. About the Data

<img align="right" src="https://raw.githubusercontent.com/aytuncilhan/VC-Investment-Analysis/main/AnalysisResults/DatasetOccurence.png" alt="My Image" width="400">
The 13D filings were manually labeled to create the training (and testing) dataset. Due to data privacy reasons, the datasets are not provided in the repository but they can be shared upon request with the permission of the originator(s).
<br/><br/>
On the right, you can see the distribution of the labels in the training dataset. The dataset is an imbalanced dataset and this fact is reflected upon in the upcoming section, especially when analysing the results of Linear SVC.
<br/><br/>
In the project, the DataEngine class provides the pre-processing of raw data until it gets ready for the training models. After reading the .txt files into a pandas dataframe, we use NLTK stemmer to get stems of each word and use stopwords to remove unnecessary words from the dataset.

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
* It is important to highlight the differences between [Support Vector Machine (SVM)](https://scikit-learn.org/stable/modules/svm.html#svm-classification) and Linear SVC even though Linear SVC is documents under SVM in Scikit-learn documentation. [This stackoverflow article](https://stackoverflow.com/questions/33843981/under-what-parameters-are-svc-and-linearsvc-in-scikit-learn-equivalent) explains some of the differnces in contrast to the documentation.
* Also, note that while SVM performs better for small datasets ([e.g. <100k samples](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)) Linear SVC [performs well](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html) for large datasets.
* Looking at the test results, LinearSVC achieved 69% accuuracy, about the same as Random Forest Classifier which makes the two methods best fit for this project.

<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

### 3.3. Stochastic Gradient Descent (SGD)
<img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/StochasticGradientDescent/Report_SGD_20.png" width="500"><br/><img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/StochasticGradientDescent/Heatmap_SGD_20.png" width="500">
* As mentioned in the [Scikit-learn documentation](https://scikit-learn.org/stable/modules/sgd.html), Stochastic Gradient Descent (SGD) is a simple yet very efficient approach to fitting linear classifiers and regressors under convex loss functions such as (linear) Support Vector Machines and Logistic Regression. Considering the (linear) SVM relation of SGD and having made the point that SVM is different than LinearSVC in the previous section, it is interesting to see the similarity between Linear SVC and SGD.
* Again from the [Scikit-learn documentation](https://scikit-learn.org/stable/modules/sgd.html), the advantages of Stochastic Gradient Descent are:
  * Efficiency.
  * Ease of implementation (lots of opportunities for code tuning).
* And the disadvantages of Stochastic Gradient Descent include:
  * SGD requires a number of hyperparameters such as the regularization parameter and the number of iterations.
  * SGD is sensitive to feature scaling.

<br/><br/><br/><br/><br/><br/>

### 3.4. K-nearest Neighbor Classifier
<img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/KNearestNeighbor/Report_KNN_20.png" width="500"><br/><img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/KNearestNeighbor/Heatmap_KNN_20.png" width="500">
* The K-nearest Neighbor Classifier is one of the simplest algorithms to implement. The lsimplicity is well explained in this [g2 article](https://learn.g2.com/k-nearest-neighbor):
  * It's called a lazy learning algorithm or lazy learner because it doesn't perform any training when you supply the training data. Instead, it just stores the data during the training time and doesn't perform any calculations. It doesn't build a model until a query is performed on the dataset. This makes KNN ideal for data mining.
  * It's considered a non-parametric method because it doesn’t make any assumptions about the underlying data distribution. Simply put, KNN tries to determine what group a data point belongs to by looking at the data points around it.
* The simplicity of the KNN algorithm fails to cope with the text data extracted from the filings and hence the accuracy seems to be lower compared to other more complex classifiers (e.g. that use kernels as LinearSVC or set of Decision Trees as Random Forest).
* Nonetheless, approximately 50% accuracy is reached which is around 3 times better than baseline random selection of 12.5%.
* Just like the previous three classifiers, Improve Governance was the most confused cluster but the poor result is amplified in KNN.


### 3.5. Multinomial Naive Bayesian
<img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/MultinomialNaiveBayes/Report_MNB_20.png" width="500"><br/><img align="right" src="https://github.com/aytuncilhan/VC-Investment-Analysis/blob/main/AnalysisResults/MultinomialNaiveBayes/Heatmap_MNB_20.png" width="500">

* Known as one of the most simple, straightforward Machine Learning Algorithm.
* As explained [here](https://towardsdatascience.com/naive-bayes-classifier-explained-50f9723571ed), the key difference of Naive Bayes Classifier is that it assumes that features are independent of each other and there is no correlation between features. However, this is not the case in real life. This naive assumption of features being uncorrelated is the reason why this algorithm is called “naive”.
* The low accuracy rate is probably the result of the assumption that the features are uncorrelated (provided that the categories are various motivations of financial investments and it's naive to assume uncorrelated features from the extarcted text.
* Just like KNN, about 50% accuracy is reached which is approximately still 3 times better than baseline random selection  of 12.5%.
* Unlike KNN, from the heatmaps, Multinomial Naive Bayesian has done interestingly well in identifying "Improve Governance" cluster (this was the section all other classifiers had trouble with in their confusion matrix) but has done terribly wrong in identifying "Other" cluster.

<br/><br/><br/><br/>

## Conclusion and Future Work

This was a great project to work with unstructured data, preprocess data, implement Supervised Machine Learning, and assess their performances in the given context.

The heatmaps provide great insights as to how the classifier fails. Further analysis can be done by looking at the vectors and identified features to have deeper insights on the performance of the classifiers for specific clusters.

Maximum accuracy was just under 70% for this project (for 8 clusters) but with a more balanced and high-quality dataset would yield much higher accuracy.
