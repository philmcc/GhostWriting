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
url = "/Users/pmcclarence/Vagrantboxes/GW/vagrant/training_dataset_03.csv"

names = ['a.LexicalDiversity','a.MeanWordLen','a.MeanSentenceLen','a.StdevSentenceLen','a.MeanParagraphLen','a.DocumentLen','a.Commas','a.Semicolons','a.Quotes','a.Exclamations','a.Colons','a.Dashes','a.Mdashes','a.Ands','a.Buts','a.Howevers','a.Ifs','a.Thats','a.Mores','a.Musts','a.Mights','a.This','a.Verys',
'b.LexicalDiversity','b.MeanWordLen','b.MeanSentenceLen','b.StdevSentenceLen','b.MeanParagraphLen','b.DocumentLen','b.Commas','b.Semicolons','b.Quotes','b.Exclamations','b.Colons','b.Dashes','b.Mdashes','b.Ands','b.Buts','b.Howevers','b.Ifs','b.Thats','b.Mores','b.Musts','b.Mights','b.This','b.Verys', 'Output']

 
dataset = pandas.read_csv(url, names=names, delimiter='|')


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

print "\nSplitting dataset:"
print('X_train: - length,width ' + str(X_train.shape))
print('X_validation: - length,width ' + str(X_validation.shape))
print('Y_train: - length,width ' + str(Y_train.shape))
print('Y_validation: - length,width ' + str(Y_validation.shape))


print"\ntop 10 rows of the dataset:"
print(X_validation)

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

print "\nRun some basic algorithms against the training dataset and evaluate the results"
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

print "\nCART was looking good so test this against the validation dataset: "
CART = DecisionTreeClassifier()
CART.fit(X_train, Y_train)
predictions = CART.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

print "Aprox 75 percent accuracy on unseen validation data."