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

import pickle
# b) Load dataset
url = "/Users/pmcclarence/Vagrantboxes/GW/vagrant/training_dataset_04.csv"

#url="test_data_comb_big.csv"

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

# b) Test options and evaluation metric
num_folds = 10 # for k-fold cross validation (or 10 fold in this case)
num_instances = len(X_train) 
seed = 7 # Setting the seed so that each different model gets exactly the same data
scoring = 'accuracy' # Metric to evaluate the tests by - correct instances / total instances



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


filename =  'finalized_model.sav'
pickle.dump(model, open(filename,  'wb' ))
# some time later...
# load the model from disk
loaded_model = pickle.load(open(filename,  'rb' ))

scaler = StandardScaler().fit(X_validation)
rescaledX = scaler.transform(X_validation)

result = loaded_model.score(rescaledX, Y_validation)
print(result)




