# credit-risk-classification
Challenge for Module 20 - Supervised Machine Learning

This challenge utilizes different Supervised Machine Learning tactics to create and evaluate a model based on loan default risk, using data from a peer-to-peer lending services company to build a model that can identify the creditworthiness of loan borrowers. 

Libraries used in this analysis include: 
- import numpy as np
- import pandas as pd
- from pathlib import Path
- from sklearn.metrics import confusion_matrix, classification_report
- from sklearn.model_selection import train_test_split
- from sklearn.linear_model import LogisticRegression

# Loading and Preparing the Data:
We start by reading in the "Lending_Data" CSV into Jupyter notebook and converting it to a Dataframe using Pandas. When previewing the dataframe we can see that the variables consist of Loan Size, Interest Rate, Borrower Income, Debt-to-Income ratio, Number of Accounts, Derogatory Marks, Total Debt, Loan Status. We can also see (using .describe()) that there are 77,536 rows, with the loan size ranging from $5,000 to $23,800 and an average of $9,805. 

We then take steps to separate the X and Y variables; The Y variable lives in the "loan_status" column, which identifies a borrower as either 0 ("healthy loan") or 1 ("high risk loan"). We separate the Y variable from the remainder of the columns, which are then identified as X variables in a separate dataframe.
From there, we use the **"train_test_split"** function from the sklearn.linear_model library to split the X and Y variables into Training and Testing groups. This is important as ............
