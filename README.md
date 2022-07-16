# Venture Capital Investment Activity Classifier
The aim for this project is to automate the process of classifying the purpose of Venture Capital investments under the 8 predefined categories (listed in the table below).

[Scikit-learn](https://scikit-learn.org/stable/) library is used to implement a Natural Language Processing model to accurately classify investment purposes, utilizing the extracted data from [13D filings](https://en.wikipedia.org/wiki/Schedule_13D). As classifier models, **Random Forest**, **K-nearest Neighbors**, and **Linear Support Vector Classifier (LinearSVC)** are used. Their performances are analysed depending on a variety of KPIs.

## About the Dataset
Training dataset labels value count of investment purpose: 
 
    ALL M&A                         426
    Improve Governance              420
    Alter Business Strategy         144
    Change Capital Structure        143
    Trading Strategy by Activist     68
    Sell Target Company              67
    Bankruptcy                       18
    Other                           850

| Purpose       | Occurence Frequency |
| :-------------: |:-------------:|
| ALL M&A                       | 426 |
| Improve Governance            | 420 |
| Alter Business Strategy       | 144 |
| Change Capital Structure      | 143 |
| Trading Strategy by Activist  | 68 |
| Sell Target Company           | 67 |
| Bankruptcy                    | 18 |
| Other                         | 850 |


## 1. Random Forest Classifier

Results when the test datset comrpise 20% of all data: 

                              precision    recall  f1-score   support

                     ALL M&A       0.50      0.07      0.12        28
                       Other       0.83      0.19      0.31        26
    Alter Business Strategy        0.87      0.84      0.86        89
          Improve Governance       1.00      0.21      0.35        19
        Sell Target Company        0.50      0.75      0.60        93
                  Bankruptcy       0.00      0.00      0.00         1
    Change Capital Structure       0.74      0.82      0.78       159

                    accuracy                           0.69       428
                   macro avg       0.65      0.43      0.46       428
                weighted avg       0.71      0.69      0.66       428


Results when the test datset comrpise 30% of all data: 


                              precision    recall  f1-score   support

                     ALL M&A       0.75      0.15      0.24        41
                       Other       0.50      0.05      0.09        41
    Alter Business Strategy        0.84      0.80      0.82       126
          Improve Governance       1.00      0.25      0.40        28
        Sell Target Company        0.47      0.68      0.56       136
                  Bankruptcy       0.00      0.00      0.00         4
    Change Capital Structure       0.69      0.81      0.74       245

                    accuracy                           0.65       641
                   macro avg       0.64      0.41      0.44       641
                weighted avg       0.68      0.65      0.62       641
                
                
### Heat Maps



| 20%        | 30%           |
| :-------------: |:-------------:|
| <img src="https://raw.githubusercontent.com/aytuncilhan/VC-Investment-Analysis/main/Assests/20Percent.png" width="600"> | <img src="https://raw.githubusercontent.com/aytuncilhan/VC-Investment-Analysis/main/Assests/30percent.png" width="600"> |
| col 2 is      | centered      |
| zebra stripes | are neat      |


## 2. LinerSVC (Support Vector Classifier)



## Analysis and Conclusion

Lorem ipsum dolor sit amet
