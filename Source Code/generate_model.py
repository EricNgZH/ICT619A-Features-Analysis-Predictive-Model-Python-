from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
import numpy as np
import pandas as pd
import dataframe_image as dfi
import pickle
import os

# Define a function to show the progress bar
with tqdm(total=4, position=1, leave=True) as progress:
    
    resultMap = {'1': 'Yes', '0': "No"}
    
    # Create the result folder
    dtDirectory = "Decision Tree Output"
    if not os.path.exists(dtDirectory):
        os.makedirs(dtDirectory)
        
    rfDirectory = "Random Forest Output"
    if not os.path.exists(rfDirectory):
        os.makedirs(rfDirectory)
        
    lrDirectory = "Linear Regression Output"
    if not os.path.exists(lrDirectory):
        os.makedirs(lrDirectory)   
        
    # Read the data set
    customerDf = pd.read_csv("Data Source/CustomerAcqusition.csv")
    spendDf= pd.read_csv("Data Source/Spend.csv")
    repaymentDf = pd.read_csv("Data Source/Repayment.csv")
    progress.update(1)

    # Merged the customer and spending df
    mergedCustomerSpendingDf = pd.merge(customerDf, spendDf, on="Customer")

    # Pivot the customer spending df to get the total spending amount for each customer
    pivotCustomerSpendingDf = pd.pivot_table(mergedCustomerSpendingDf, values="Amount", index="Customer", aggfunc='mean')

    # Merged the customer and repayment df
    mergedCustomerRepaymentDf = pd.merge(customerDf, repaymentDf, on="Customer")

    # Pivot the customer repayment df to get the total repayment amount for each customer
    pivotCustomerRepaymentDf = pd.pivot_table(mergedCustomerRepaymentDf, values="Amount", index="Customer", aggfunc='mean')

    # Combine the customer total spending and customer repayment amount into a single df
    newMergedDf = pd.merge(customerDf, pivotCustomerSpendingDf, on="Customer")
    newMergedDf.rename(columns={'Amount': 'Average Spending Amount'}, inplace=True)
    newMergedDf = pd.merge(newMergedDf, pivotCustomerRepaymentDf, on="Customer")
    newMergedDf.rename(columns={'Amount': 'Average Repayment Amount'}, inplace=True)

    # Card spend ratio = customer's spending amount / limit
    newMergedDf['Card Spend Ratio'] = newMergedDf['Average Spending Amount'] / newMergedDf['Limit']

    # Assumption of high spender is when the card spend ration is more than 0.8 
    newMergedDf['High Spender'] = np.where(newMergedDf['Card Spend Ratio'] >= 0.8, '1', '0')

    # Avg repayment rate 
    newMergedDf['Average Repayment Rate'] = newMergedDf['Average Repayment Amount'] / newMergedDf['Average Spending Amount']

    # Mapping city
    cityValueToNumber = {'BANGALORE': 1, 'CALCUTTA': 2, 'COCHIN': 3, 'BOMBAY': 4, 'DELHI': 5, 'PATNA': 6, 'CHENNAI': 7, 'TRIVANDRUM': 8}
    newMergedDf['City'] = newMergedDf['City'].map(cityValueToNumber)

    # Mapping product
    productValueToNumber = {'Gold': 1, 'Silver': 2, 'Platimum': 3}
    newMergedDf['Product'] = newMergedDf['Product'].map(productValueToNumber)

    # Mapping company
    companyValueToNumber = {'C1':1, 'C2':2, 'C3':3, 'C4':4, 'C5':5, 'C6':6, 'C7':7, 'C8':8, 'C9':9, 'C10':10, 'C11':11, 'C12':12, 'C13':13,
    'C14':14, 'C15':15, 'C16':16, 'C17':17, 'C18':18, 'C19':19, 'C20':20, 'C21':21, 'C22':22, 'C23':23, 'C24':24, 'C25':25,
    'C26':26, 'C27':27, 'C28':28, 'C29':29, 'C30':30, 'C31':31, 'C32':32, 'C33':33, 'C34':34, 'C35':35, 'C36':36, 'C37':37,
    'C38':38, 'C39':39, 'C40':40, 'C41':41}
    newMergedDf['Company'] = newMergedDf['Company'].map(companyValueToNumber)

    # Mapping segment
    segmentValueToNumber = {'Self Employed':1, 'Salaried_MNC':2, 'Salaried_Pvt':3, 'Govt':4, 'Normal Salary':5}
    newMergedDf['Segment'] = newMergedDf['Segment'].map(segmentValueToNumber)

    # Defined independent column name
    # Independent variable: Age, City, Product, Company, Segment, Average Repayment Rate
    independentVarColumnName = ["Age", "City", "Product", "Company", "Segment", "Average Repayment Rate"]

    # Defined dependent column name
    # Dependent variable: High Spender
    dependentVarColumnName = ["High Spender"]
    
    # Split the data into train and test data
    trainDf, testDf =  train_test_split(newMergedDf, test_size=0.2, random_state=1)

    # Get the train df based on the independent and dependent column name
    independentTrainDf = trainDf.loc[:, independentVarColumnName].values
    dependentTrainDf = trainDf.loc[:, dependentVarColumnName].values

    # Get the test df based on the independent and dependent column name
    independentTestDf = testDf.loc[:, independentVarColumnName].values
    dependentTestDf = testDf.loc[:, dependentVarColumnName].values

    # Preparing the n-fold cross validation with 5 splits
    nSplits = 5
    kf = KFold(n_splits=nSplits, shuffle=True)
    
    print("----------------------")
    progress.update(1)
    
    # RF
    rfData = {
        'RF': ['Accuracy', 'AUC', 'MSE', 'Precision Rate', 'Recall Rate'],
    }
    
    maxDepth = [-1,20,15,10,7,6,5]
    for num in maxDepth:
        # Train the model using train data
        rfClassifierModel = RandomForestClassifier() if num == -1 else RandomForestClassifier(max_depth=num)
        
        rfAccuracyTestScores = np.zeros(nSplits)
        rfAucTestScores = np.zeros(nSplits)
        rfMseTestScores = np.zeros(nSplits)
        rfPrecisionTestScores = np.zeros(nSplits)
        rfRecallTestScores = np.zeros(nSplits)
        
        for i, (trainIdx, testIdx) in enumerate(kf.split(newMergedDf)):
            trainDf = newMergedDf.loc[trainIdx]
            rfTestDf = newMergedDf.loc[testIdx]
            
            independentTrainDf = trainDf.loc[:, independentVarColumnName].values
            dependentTrainDf = trainDf.loc[:, dependentVarColumnName].values
            
            independentTestDf = rfTestDf.loc[:, independentVarColumnName].values
            dependentTestDf = rfTestDf.loc[:, dependentVarColumnName].values
            
            rfClassifierModel.fit(independentTrainDf, dependentTrainDf.ravel())
            
            # Predict using the trained model
            rfPredictionResult = rfClassifierModel.predict(independentTestDf)

            # Print out the accuracy score
            rfAccuracyTestScores[i] = accuracy_score(dependentTestDf, rfPredictionResult)

            # Print out the AUC score
            rfAucTestScores[i] = roc_auc_score(dependentTestDf, rfPredictionResult)

            # Print out the MSE score
            rfMseTestScores[i] = mean_squared_error(dependentTestDf, rfPredictionResult)
            rfPrecisionTestScores[i] = precision_score(dependentTestDf, rfPredictionResult, pos_label='0')
            rfRecallTestScores[i] = recall_score(dependentTestDf, rfPredictionResult, pos_label='0')
        
        rfTitle = "Default" if num == -1 else ("MaxDepth=" + str(num))
        rfData[rfTitle] = [
            np.mean(rfAccuracyTestScores), 
            np.mean(rfAucTestScores), 
            np.mean(rfMseTestScores), 
            np.mean(rfPrecisionTestScores), 
            np.mean(rfRecallTestScores)
        ]
        
        # Save your model to a file
        if num == 10:
            with open('model.pkl', 'wb') as f:
                pickle.dump(rfClassifierModel, f)

    rfDf = pd.DataFrame(rfData)
    dfi.export(rfDf,"Random Forest Output/RF pruning result comparison.png")

    print("----------------------")
    progress.update(1)

    # DT
    dtData = {
        'DT': ['Accuracy', 'AUC', 'MSE', 'Precision Rate', 'Recall Rate'],
    }

    maxDepth = [-1,6,5,4,3,2,1]
    for num in maxDepth:
        # Train the model using train data
        dtClassifierModel = DecisionTreeClassifier() if num == -1 else DecisionTreeClassifier(max_depth=num)
        
        # Create the array to contains the scores
        dtAccuracyTestScores = np.zeros(nSplits)
        dtAucTestScores = np.zeros(nSplits)
        dtMseTestScores = np.zeros(nSplits)
        dtPrecisionTestScores = np.zeros(nSplits)
        dtRecallTestScores = np.zeros(nSplits)
        
        for i, (trainIdx, testIdx) in enumerate(kf.split(newMergedDf)):
            trainDf = newMergedDf.loc[trainIdx]
            dtTestDf = newMergedDf.loc[testIdx]
            
            independentTrainDf = trainDf.loc[:, independentVarColumnName].values
            dependentTrainDf = trainDf.loc[:, dependentVarColumnName].values
            
            independentTestDf = dtTestDf.loc[:, independentVarColumnName].values
            dependentTestDf = dtTestDf.loc[:, dependentVarColumnName].values
            
            dtClassifierModel.fit(independentTrainDf, dependentTrainDf)
            
            # Predict using the trained model
            dtPredictionResult = dtClassifierModel.predict(independentTestDf)

            # Print out the accuracy score
            dtAccuracyTestScores[i] = accuracy_score(dependentTestDf, dtPredictionResult)

            # Print out the AUC score
            dtAucTestScores[i] = roc_auc_score(dependentTestDf, dtPredictionResult)

            # Print out the MSE score
            dtMseTestScores[i] = mean_squared_error(dependentTestDf, dtPredictionResult)
            dtPrecisionTestScores[i] = precision_score(dependentTestDf, dtPredictionResult, pos_label='0')
            dtRecallTestScores[i] = recall_score(dependentTestDf, dtPredictionResult, pos_label='0')
        
        dtTitle = "Default" if num == -1 else ("MaxDepth=" + str(num))
        dtData[dtTitle] = [
            np.mean(dtAccuracyTestScores), 
            np.mean(dtAucTestScores), 
            np.mean(dtMseTestScores), 
            np.mean(dtPrecisionTestScores), 
            np.mean(dtRecallTestScores)
        ]

    dtDf = pd.DataFrame(dtData)
    dfi.export(dtDf,"Decision Tree Output/DT pruning result comparison.png")

    print("----------------------")
    progress.update(1)
    
    # LR
    # Train the model using train data
    lrClassifierModel = LogisticRegression()

    lrAccuracyTestScores = np.zeros(nSplits)
    lrAucTestScores = np.zeros(nSplits)
    lrMseTestScores = np.zeros(nSplits)
    lrPrecisionTestScores = np.zeros(nSplits)
    lrRecallTestScores = np.zeros(nSplits)

    for i, (trainIdx, testIdx) in enumerate(kf.split(newMergedDf)):
        lrTrainDf = newMergedDf.loc[trainIdx]
        lrTestDf = newMergedDf.loc[testIdx]
        
        lrIndependentTrainDf = lrTrainDf.loc[:, independentVarColumnName].values
        lrDependentTrainDf = lrTrainDf.loc[:, dependentVarColumnName].values
        
        lrIndependentTestDf = lrTestDf.loc[:, independentVarColumnName].values
        lrDependentTestDf = lrTestDf.loc[:, dependentVarColumnName].values
        
        lrClassifierModel.fit(lrIndependentTrainDf, lrDependentTrainDf.ravel())
        
        lrPredictionResult = lrClassifierModel.predict(lrIndependentTestDf)
        
        lrAccuracyTestScores[i] = accuracy_score(lrDependentTestDf, lrPredictionResult)
        lrAucTestScores[i] = roc_auc_score(dependentTestDf, lrPredictionResult)
        lrMseTestScores[i] = mean_squared_error(dependentTestDf, lrPredictionResult)
        lrPrecisionTestScores[i] = precision_score(dependentTestDf, lrPredictionResult, pos_label='0')
        lrRecallTestScores[i] = recall_score(dependentTestDf, lrPredictionResult, pos_label='0')
        
        
    lrData = {
        'LR': ['Accuracy', 'AUC', 'MSE', 'Precision Rate', 'Recall Rate'],
    }

    lrData['Default'] = [
        np.mean(lrAccuracyTestScores), 
        np.mean(lrAucTestScores), 
        np.mean(lrMseTestScores), 
        np.mean(lrPrecisionTestScores), 
        np.mean(lrRecallTestScores)
    ]

    lrDf = pd.DataFrame(lrData)
    dfi.export(lrDf,"Linear Regression Output/LR result.png")
    
    print("----------------------")
    
    newMergedDf = newMergedDf.drop('High Spender', axis=1)
    newMergedDf.to_csv("Data Source/dataset.csv")

    print("Done")