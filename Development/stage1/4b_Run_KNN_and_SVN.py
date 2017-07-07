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



# Run on KNN
print "\nRunning KNN Predictions"
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = KNeighborsClassifier(n_neighbors=211)
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
model = SVC(C=1.7)
model.fit(rescaledX, Y_train)
# estimate accuracy on validation dataset
rescaledValidationX = scaler.transform(X_validation)
predictions = model.predict(rescaledValidationX)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

