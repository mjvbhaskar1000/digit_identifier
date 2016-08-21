#importing the required modules.
import sklearnMLCommon as cm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

#Building the RandomForestClassifier with 200 trees and 4 parallel runs. 
rfc = RandomForestClassifier(n_estimators=200, n_jobs = 4)

#training the model.
rfc.fit(cm.xtrain, cm.ytrain)

#testing the model.
predictedVal = rfc.predict(cm.xtest)

#calculating and printing the accuracy of the model.
accuracy = cm.accuracy_score(cm.ytest, predictedVal)
print "Accuracy is: ", accuracy
