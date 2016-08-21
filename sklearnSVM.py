import sklearnMLCommon as cm
from sklearn.svm import LinearSVC

svmc = LinearSVC()
svmc.fit(cm.xtrain, cm.ytrain)
predictedVal = svmc.predict(cm.xtest)
accuracy = cm.accuracy_score(cm.ytest, predictedVal)
print "Accuracy: ", accuracy

 
