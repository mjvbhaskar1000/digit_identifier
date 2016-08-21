#importing the required modules.
import sklearnMLCommon as cm
from sklearn.svm import LinearSVC

#http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
#building a model.
svmc = LinearSVC()

#training the model.
svmc.fit(cm.xtrain, cm.ytrain)

#testing the model.
predictedVal = svmc.predict(cm.xtest)

#calculating the accuracy and printing it.
accuracy = cm.accuracy_score(cm.ytest, predictedVal)
print "Accuracy: ", accuracy

 
