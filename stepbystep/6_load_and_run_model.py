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
url_same= "test_data_same_big.csv"
url_diff= "test_data_diff_big.csv"
url_comb = "test_data_comb_big.csv"

names = ['a.LexicalDiversity','a.MeanWordLen','a.MeanSentenceLen','a.StdevSentenceLen','a.MeanParagraphLen','a.DocumentLen','a.Commas','a.Semicolons','a.Quotes','a.Exclamations','a.Colons','a.Dashes','a.Mdashes','a.Ands','a.Buts','a.Howevers','a.Ifs','a.Thats','a.Mores','a.Musts','a.Mights','a.This','a.Verys',
'b.LexicalDiversity','b.MeanWordLen','b.MeanSentenceLen','b.StdevSentenceLen','b.MeanParagraphLen','b.DocumentLen','b.Commas','b.Semicolons','b.Quotes','b.Exclamations','b.Colons','b.Dashes','b.Mdashes','b.Ands','b.Buts','b.Howevers','b.Ifs','b.Thats','b.Mores','b.Musts','b.Mights','b.This','b.Verys', 'Output']

 
dataset_same = pandas.read_csv(url_same, names=names, delimiter='|')
dataset_diff = pandas.read_csv(url_diff, names=names, delimiter='|')
dataset_comb = pandas.read_csv(url_comb, names=names, delimiter='|')


array_same = dataset_same.values
X_same = array_same[:,0:46] 	# inputs
Y_same = array_same[:,46]		# outpus
array_diff = dataset_diff.values
X_diff = array_diff[:,0:46] 	# inputs
Y_diff = array_diff[:,46]		# outpus
array_comb = dataset_comb.values
X_comb = array_comb[:,0:46] 	# inputs
Y_comb = array_comb[:,46]		# outpus


filename =  'finalized_model.sav'
loaded_model = pickle.load(open(filename,  'rb' ))

scaler_same = StandardScaler().fit(X_same)
rescaledX_same = scaler_same.transform(X_same)

scaler_diff = StandardScaler().fit(X_diff)
rescaledX_diff = scaler_diff.transform(X_diff)

scaler_comb = StandardScaler().fit(X_comb)
rescaledX_comb = scaler_comb.transform(X_comb)

result_same = loaded_model.score(rescaledX_same, Y_same)
print("Same author dataset: " + str(result_same))
result_diff = loaded_model.score(rescaledX_diff, Y_diff)
print("Diff author dataset: " + str(result_diff))
result_comb = loaded_model.score(rescaledX_comb, Y_comb)
print("Combined dataset: " + str(result_comb))




