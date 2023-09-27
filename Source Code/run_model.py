from sklearn import tree
import pickle
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
    
dtDirectory = "Final Output"
if not os.path.exists(dtDirectory):
    os.makedirs(dtDirectory)
    
# Read the dataset
df = pd.read_csv("Data Source/dataset.csv")

# Features
features = ["Age", "City", "Product", "Company", "Segment", "Average Repayment Rate"]

# Update the DataFrame with the features only
features_df = df.loc[:, features].values

# Predict using the model
result = model.predict(features_df)

# Result map
resultMap = {'1': "Yes", '0': "No"}

# Put the result into DataFrame
df["High Spender Prediction Result"] =  np.vectorize(resultMap.get)(result)

# Generate the output with the prediction
df.to_csv("Final Output/output.csv")

print(df.head())

rfFig, rfAxes = plt.subplots(nrows=1, ncols=5, figsize=(10,5), dpi=900)

# generate the first 5 tree from random forest
for index in range(0, 5):
    tree.plot_tree(model.estimators_[index],
        feature_names = features, 
        class_names=['0', '1'],
        filled = True,
        ax = rfAxes[index]);
    
    rfAxes[index].set_title('Estimator: ' + str(index), fontsize = 9)
    
# save the trees plot into an image
rfFig.savefig('Final Output/trees.png')