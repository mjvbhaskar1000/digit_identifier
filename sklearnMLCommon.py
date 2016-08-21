import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

trainData = pd.read_csv("train.csv")
features = trainData.columns[1:]
x = trainData[features]
y = trainData['label']

xtrain, xtest, ytrain, ytest = cross_validation.train_test_split(x/255., y, test_size=0.1, random_state=0)

