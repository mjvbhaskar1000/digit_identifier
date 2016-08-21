#Machine Learning based Hand Written Digit Identifier

This uses various machine learning algorithms to identify hand written digits. <br />
The dataset can be downloaded from: https://www.kaggle.com/c/digit-recognizer

Description of each algorithm and their results can be found in report.pdf. <br />
Description of each file:

* sklearnMLCommon.py: This file reads the .csv file and builds the training and testing dataset using crossvalidation. This program is used by all the other programs.<br />
* sklearnExtraTree.py: This file has the program for implementation of Extremely Randomized Trees algorithm. <br />
* sklearnKNN.py: This file has the program for implementation of K-Nearest Neighbors algorithm. <br />
* sklearnRandomForest.py: This file has the program for implementation of Random Forest algorithm. <br />
* sklearnSVM.py: This file has the program for implementation of Support Vector Machine algorithm. <br />
* tensorflownn.py: This file has the program for implementation of Neural Network using Tensorflow. <br />
* tensorflownn-opt.py: This file has the program for optimized implementation of Neural Network using Tensorflow. Using this user can easily increase or decrease the number of hidden layers in the network.

The results of execution can be visualized: <br />
![picture alt](https://github.com/mjvbhaskar1000/digit_identifier/blob/master/graph.png "Algorithm vs Time and Accuracy")
