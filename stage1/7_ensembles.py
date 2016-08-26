import pandas
import numpy
import matplotlib.pyplot as plt
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

import pickle
# b) Load dataset
url = "../Datasets/stage_1_dataset_05.csv"
names = ['a.LexicalDiversity' ,'a.MeanWordLen' , 'a.MeanSentenceLen' , 'a.MeanParagraphLen' , 'a.DocumentLen' , 'a.Commas',  'a.Semicolons' , 'a.Exclamations' , 'a.Buts' , 'a.Thats' , 'a.This' ,'b.LexicalDiversity' ,'b.MeanWordLen' , 'b.MeanSentenceLen' , 'b.MeanParagraphLen' , 'b.DocumentLen' , 'b.Commas' ,  'b.Semicolons' , 'b.Exclamations' , 'b.Buts' , 'b.Thats' , 'b.This' , 'Output']
dataset = pandas.read_csv(url, names=names, delimiter='|')


######
# 4. Evaluate Algorithms
######
# a) Split-out validation dataset
array = dataset.values
X = array[:,0:22] 	# inputs
Y = array[:,22]		# outpus
validation_size = 0.20	# Hold back 20% of data for later validation
seed = 7	# Random Seed for reproducability
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y,
    test_size=validation_size, random_state=seed) # Do the actual split!

# b) Test options and evaluation metric
num_folds = 10 # for k-fold cross validation (or 10 fold in this case)
num_instances = len(X_train) 
seed = 7 # Setting the seed so that each different model gets exactly the same data
scoring = 'accuracy' # Metric to evaluate the tests by - correct instances / total instances



# ensembles

estimators = []
model1 = LogisticRegression()
#estimators.append(( 'logistic' , model1))
model2 = DecisionTreeClassifier()
#estimators.append(( 'cart' , model2))
model3 = SVC()
#estimators.append(( 'svm' , model3))
model4 = KNeighborsClassifier()
#estimators.append(('KNN', model4)) 
model5 =GradientBoostingClassifier()
estimators.append(('GBM', model5))
model6 =SVC()
#estimators.append(('SVM', SVC()))
model7 =GradientBoostingClassifier()
estimators.append(('GBM', model7))

print "\nEnsembles\n"
ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreesClassifier()))
ensembles.append(('SVN', SVC()))
ensembles.append(('KNN', KNeighborsClassifier()))
ensembles.append(('VoCl', VotingClassifier(estimators)))
results = []
names = []
for name, model in ensembles:
	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
	cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


print "\nEnsembles through scaled pipeline"

pipelines = []
pipelines.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostClassifier())])))
pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM', GradientBoostingClassifier())])))
pipelines.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestClassifier())])))
pipelines.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET', ExtraTreesClassifier())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledVoCL', Pipeline([('Scaler', StandardScaler()),('VoCl', VotingClassifier(estimators))])))
results = []
results = []
names = []
for name, model in pipelines:
	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
	cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)




