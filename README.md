# Venture Capital Investment Activity Classifier
The aim for this project is to automate the process of classifying the purpose of Venture Capital investments under the 8 predefined categories (listed in the table below).

The [Natural Language Toolkit (nltk)](https://www.nltk.org) and [Scikit-learn](https://scikit-learn.org/stable/) library is used to implement the Natural Language Processing model to accurately classify investment purposes, utilizing the extracted data from [13D filings](https://en.wikipedia.org/wiki/Schedule_13D). As classifier models, **Random Forest**, **K-nearest Neighbors**, and **Linear Support Vector Classifier (LinearSVC)** are used. Their performances are analysed depending on a variety of KPIs.

## About the Dataset

Training dataset labels value count of investment purpose: 

| Purpose       | Occurence Frequency |
| :-------------|:-------------:|
| ALL M&A                       | 426 |
| Improve Governance            | 420 |
| Alter Business Strategy       | 144 |
| Change Capital Structure      | 143 |
| Trading Strategy by Activist  | 68 |
| Sell Target Company           | 67 |
| Bankruptcy                    | 18 |
| Other                         | 850 |

Due to data privacy reasons, the respective datasets are not presented in the repository.

## 1. Random Forest Classifier

                
### Heat Maps



| 20%        | 30%           |
| :-------------: |:-------------:|
| <img src="https://raw.githubusercontent.com/aytuncilhan/VC-Investment-Analysis/main/Assests/20Percent.png" width="600"> | <img src="https://raw.githubusercontent.com/aytuncilhan/VC-Investment-Analysis/main/Assests/30percent.png" width="600"> |
| col 2 is      | centered      |
| zebra stripes | are neat      |


## 2. LinerSVC (Support Vector Classifier)



## Analysis and Conclusion

Lorem ipsum dolor sit amet
