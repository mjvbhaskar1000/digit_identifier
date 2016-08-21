#importing the required modules.
import sklearnMLCommon as cm
from sklearn.neighbors import KNeighborsClassifier

#http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
#defining a KNeighborsClassifier with n_jobs = 4. 
#value of n_jobs much be modified depending the number of cores.
knnc = KNeighborsClassifier(n_jobs=4)

#training the model.
knnc.fit(cm.xtrain, cm.ytrain)

#testing the model.
predictedVal = knnc.predict(cm.xtest)

#calculating the accuracy and printing it.
accuracy = cm.accuracy_score(cm.ytest, predictedVal)
print "accuracy: ", accuracy
