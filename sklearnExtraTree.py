import sklearnMLCommon as cm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib

etc = ExtraTreesClassifier(n_estimators=200, n_jobs = 4)
etc.fit(cm.xtrain, cm.ytrain)
predictedVal = etc.predict(cm.xtest)
accuracy = cm.accuracy_score(cm.ytest, predictedVal)
#joblib.dump(etc, 'etc.pkl')
print "Accuracy is: ", accuracy
