# credit-risk-classification
Challenge for Module 20 - Supervised Machine Learning

This challenge utilizes different Supervised Machine Learning tactics to create and evaluate a model based on loan default risk, using data from a peer-to-peer lending services company to build a model that can identify the creditworthiness of loan borrowers. Supervised Machine Learning differs from Unsupervised Machine Learning as we are providing the model with an expected output (Y) and asking the machine to create predictions, whereas with Unsupervised Machine Learning we are not defining the output in the data and are asking the machine to describe what is going on within the data. 

Libraries used in this analysis include: 
- import numpy as np
- import pandas as pd
- from pathlib import Path
- from sklearn.metrics import confusion_matrix, classification_report
- from sklearn.model_selection import train_test_split
- from sklearn.linear_model import LogisticRegression
- from sklearn.metrics import accuracy_score

# Loading and Preparing the Data:
We start by reading in the "Lending_Data" CSV into Jupyter notebook and converting it to a Dataframe using Pandas. When previewing the dataframe we can see that the variables consist of Loan Size, Interest Rate, Borrower Income, Debt-to-Income ratio, Number of Accounts, Derogatory Marks, Total Debt, Loan Status. We can also see (using .describe()) that there are 77,536 rows, with the loan size ranging from $5,000 to $23,800 and an average of $9,805. 

We then take steps to separate the X and Y variables; The Y variable lives in the "loan_status" column, which identifies a borrower as either 0 ("healthy loan") or 1 ("high risk loan"). We separate the Y variable from the remainder of the columns, which are then identified as X variables in a separate dataframe.
From there, we use the **"train_test_split"** function from the sklearn.linear_model library to split the X and Y variables into Training and Testing groups. This is important as the model is built and trained using the Training dataset, but tested for accuracy using data it has never seen before in the Testing dataset. 

# Creating & Testing the Logistic Regression Model:
To create the Logistic Regression Model, we start by importing the **LogisticRegression** library from SkLearn and create the loan_log_classifier to instantiate the model, using a random state of 1 and increasing the max iterations. We then fit the model using the X_Train and Y_Train data.
Once fitted, we can use the accuracy score (F1 score) to determine the efficacy of the model. For this, we use X_Test data rather than Training data, and use the SKlearn **.predict()** formula to get the Y_Pred value. From here we can generate an array of predicted y values and a topline accuracy score, which in this case is very high at 99.3%.

The predicted Y values (y_pred) and actual Y values (y_test) are compared using a confusion matrix, which shows the number of Y values that were predicted accurately as either a 0 or a 1, those who were predicted to be 0 but were actually 1 (false negative) and those that were predicted to be a 1 but were actually a 0. We can see that the model accurately predicted 19,240 values, predicted 107 false positives and 37 false negatives. The total F1 score is 99.3%, but the F1 score for a false negative was 89%. In this case, false negatives are more important to look at as the bank can accidentally give someone a loan based on the model telling them that they were healthy, but then having the person default on the loan. While the F1 score for high risk loan predictions is lower than healthy predictions, it is still above an 80% and can therefore be considered highly accurate. However, this is up to each individual banks' discretion based on how much risk they are comfortable with. 

![image](https://github.com/user-attachments/assets/843bb5cf-9022-4f09-b255-5a3896f63219)


