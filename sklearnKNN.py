import sklearnMLCommon as cm
from sklearn.neighbors import KNeighborsClassifier

knnc = KNeighborsClassifier(n_jobs=4)
knnc.fit(cm.xtrain, cm.ytrain)
predictedVal = knnc.predict(cm.xtest)
accuracy = cm.accuracy_score(cm.ytest, predictedVal)
print "accuracy: ", accuracy
