# credit-card-fraud-detection
Credit cards are becoming popular in purchases and transactions nowadays. Therefore credit card-related fraud is also increasing. Fraudsters steal credit cards or credit card information and make fraudulent transactions in the name of their owner and individuals as well as credit card companies have to suffer losses. In this project, machine learning models are built and the best model is selected after comparing their performances.

In this project, Kaggle's four datasets are used. After checking them for errors, we did exploratory data analysis to find out which feature is important in predicting fraudulent transactions. Then we did feature engineering and created more features based on the existing features so that the models understand the patterns easily. We applied various resampling techniques such as Random Under sampling, SMOTE and ADASYN to make the dataset balanced and then applied feature scaling to scale the dataset. After scaling, the datasets were divided into two groups for training and testing purposes. Then different machine-learning models were deployed on the training datasets. 

Once the models were trained, they were deployed on the training datasets to predict the fraudulent transactions. In this project, data is checked using Logistic Regression, Decision Tree, XGBoost, Random Forest and K-Nearest Neighbors models. The performance was evaluated using accuracy, precision, recall, F1 score, AUC ROC curve and confusion matrix.

# Steps:

Get the code from the repository 

git clone: https://github.com/mohitbt66/Credit-Card-Fraud-Detection-using-Machine-Learning.git

Download the dataset and unzip it.

Dataset 1 URL: https://www.kaggle.com/datasets/kartik2112/fraud-detection

Dataset 2 URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Dataset 3 URL: https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023

Dataset 4 URL: https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud

Install required Python packages.

Run on Jupyter Notebook.
