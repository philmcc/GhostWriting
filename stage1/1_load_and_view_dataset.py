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
url = "../Datasets/stage_1_dataset_05.csv"

names = ['a.LexicalDiversity' ,'a.MeanWordLen' , 'a.MeanSentenceLen' , 'a.MeanParagraphLen' , 'a.DocumentLen' , 'a.Commas',  'a.Semicolons' , 'a.Exclamations' , 'a.Buts' , 'a.Thats' , 'a.This' ,'b.LexicalDiversity' ,'b.MeanWordLen' , 'b.MeanSentenceLen' , 'b.MeanParagraphLen' , 'b.DocumentLen' , 'b.Commas' ,  'b.Semicolons' , 'b.Exclamations' , 'b.Buts' , 'b.Thats' , 'b.This' , 'Output']


dataset = pandas.read_csv(url, names=names, delimiter='|')

pandas.set_option('display.width', 180)
# shape
print "\nData Shape: "
print(dataset.shape)
# head
print"\ntop 10 rows of the dataset:"
print(dataset.head(20))
# descriptions
print"\nDescribe the data"
print(dataset.describe())
print"\nClass distribution"
print(dataset.groupby('Output').size())
