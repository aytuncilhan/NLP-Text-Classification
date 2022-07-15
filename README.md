# Venture Capital Investments Classifier
Natural Language Processing model to analyze Venture Capital Investment activity purpose.

TRAIN DATA VALUE COUNTS: 
 
    ALL M&A                         426
    Improve Governance              420
    Alter Business Strategy         144
    Change Capital Structure        143
    Trading Strategy by Activist     68
    Sell Target Company              67
    Bankruptcy                       18
    Other                           850

Random Forest Classifier is used.

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
    
