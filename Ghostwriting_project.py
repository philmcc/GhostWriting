# Python Project Template
######
# 1. Prepare Problem
######
# a) Load libraries
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
# b) Load dataset
url = "/Users/pmcclarence/philmccgit/GhostWriting/csvs/dataset1.csv"

names = ['a.LexicalDiversity','a.MeanWordLen','a.MeanSentenceLen','a.StdevSentenceLen','a.MeanParagraphLen','a.DocumentLen','a.Commas','a.Semicolons','a.Quotes','a.Exclamations','a.Colons','a.Dashes','a.Mdashes','a.Ands','a.Buts','a.Howevers','a.Ifs','a.Thats','a.Mores','a.Musts','a.Mights','a.This','a.Verys',
'b.LexicalDiversity','b.MeanWordLen','b.MeanSentenceLen','b.StdevSentenceLen','b.MeanParagraphLen','b.DocumentLen','b.Commas','b.Semicolons','b.Quotes','b.Exclamations','b.Colons','b.Dashes','b.Mdashes','b.Ands','b.Buts','b.Howevers','b.Ifs','b.Thats','b.Mores','b.Musts','b.Mights','b.This','b.Verys', 'Output']

 
dataset = pandas.read_csv(url, names=names, delimiter='|')
######
# 2. Summarize Data
######
# a) Descriptive statistics
pandas.set_option('display.width', 180)
# shape

print(dataset.shape)
# head
#print(dataset.head(20))
# descriptions
print(dataset.describe())
# class distribution
print(dataset.groupby('Output').size())



# b) Data visualizations

######
# 3. Prepare Data
######
# a) Data Cleaning
# b) Feature Selection
# c) Data Transforms

######
# 4. Evaluate Algorithms
######
# a) Split-out validation dataset
array = dataset.values
X = array[:,0:46] 	# inputs
Y = array[:,46]		# outpus
validation_size = 0.20	# Hold back 20% of data for later validation
seed = 7	# Random Seed for reproducability
X_train, X_validation, Y_train, Y_validation = cross_validation.train_test_split(X, Y,
    test_size=validation_size, random_state=seed) # Do the actual split!

# b) Test options and evaluation metric
num_folds = 10 # for k-fold cross validation (or 10 fold in this case)
num_instances = len(X_train) 
seed = 7 # Setting the seed so that each different model gets exactly the same data
scoring = 'accuracy' # Metric to evaluate the tests by - correct instances / total instances

# c) Spot-Check Algorithms
# Quickly test a number of models to get an idea of the best performing ones for this problem
models = []
models.append(('LR', LogisticRegression())) 
models.append(('LDA', LinearDiscriminantAnalysis())) 
models.append(('KNN', KNeighborsClassifier())) 
models.append(('CART', DecisionTreeClassifier())) 
models.append(('NB', GaussianNB())) 
models.append(('SVM', SVC()))
# evaluate each model in turn
# for each model return the mean of the accuracy and the standard deviation 
results = []
names = []
for name, model in models:
  kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
  cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold,
      scoring=scoring)
  results.append(cv_results)
  names.append(name)
  msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
  print(msg)
# d) Compare Algorithms

CART = DecisionTreeClassifier()
CART.fit(X_train, Y_train)
predictions = CART.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR', LogisticRegression())])))
pipelines.append(('ScaledLDA', Pipeline([('Scaler', StandardScaler()),('LDA', LinearDiscriminantAnalysis())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsClassifier())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART', DecisionTreeClassifier())])))
pipelines.append(('ScaledNB', Pipeline([('Scaler', StandardScaler()),('NB', GaussianNB())])))
pipelines.append(('ScaledSVM', Pipeline([('Scaler', StandardScaler()),('SVM', SVC())])))
results = []
names = []
for name, model in pipelines:
	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
	cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)




######
# 5. Improve Accuracy
######
# a) Algorithm Tuning
# Tune scaled KNN
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#neighbors = [1,3,5,7,9,11,13,15,17,19,21]
#param_grid = dict(n_neighbors=neighbors)
#model = KNeighborsClassifier()
#kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
#grid_result = grid.fit(rescaledX, Y_train)
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#for params, mean_score, scores in grid_result.grid_scores_:
#    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))


# Tune scaled SVM
#scaler = StandardScaler().fit(X_train)
#rescaledX = scaler.transform(X_train)
#c_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
#kernel_values = ['linear', 'poly', 'rbf', 'sigmoid']
#param_grid = dict(C=c_values, kernel=kernel_values)
#model = SVC()
#kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
#grid_result = grid.fit(rescaledX, Y_train)
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#for params, mean_score, scores in grid_result.grid_scores_:
#    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

# Run on KNN
print "\nRunning KNN Predictions"
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(rescaledX, Y_train)
# estimate accuracy on validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

#Run on SVC
print "\nRunning SVC Predictions"
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = SVC(C=2)
model.fit(rescaledX, Y_train)
# estimate accuracy on validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# b) Ensembles
# ensembles
print "\nEnsembles\n"
ensembles = []
ensembles.append(('AB', AdaBoostClassifier()))
ensembles.append(('GBM', GradientBoostingClassifier()))
ensembles.append(('RF', RandomForestClassifier()))
ensembles.append(('ET', ExtraTreesClassifier()))
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
results = []
names = []
for name, model in pipelines:
	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
	cv_results = cross_validation.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)



######
# 6. Finalize Model
######
# a) Predictions on validation dataset
# b) Create standalone model on entire training dataset
# c) Save model for later use