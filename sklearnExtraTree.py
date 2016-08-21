#importing the required modules.
import sklearnMLCommon as cm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib

#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
#defining the ExtraTreesClassifier with 200 trees and 4 parallel jobs.
#n_jobs value has to be modified based on the cores. 
etc = ExtraTreesClassifier(n_estimators=200, n_jobs = 4)

#training the model.
etc.fit(cm.xtrain, cm.ytrain)

#testing the model
predictedVal = etc.predict(cm.xtest)

#calculating and printing the accuracy of the model.
accuracy = cm.accuracy_score(cm.ytest, predictedVal)
print "Accuracy is: ", accuracy
