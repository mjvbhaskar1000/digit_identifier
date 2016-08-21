import sklearnMLCommon as cm
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

rfc = RandomForestClassifier(n_estimators=200, n_jobs = 4)
rfc.fit(cm.xtrain, cm.ytrain)
predictedVal = rfc.predict(cm.xtest)
accuracy = cm.accuracy_score(cm.ytest, predictedVal)
#joblib.dump(rfc, 'rfc.pkl')
#print rfc.feature_importances_
print "Accuracy is: ", accuracy
