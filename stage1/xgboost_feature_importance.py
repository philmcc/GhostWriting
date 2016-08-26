# plot feature importance using built-in function
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
# load data
url = "../Datasets/stage_1_dataset_05.csv"
dataset = loadtxt(url, delimiter="|")
# split data into X and y
X = dataset[:,0:22]
Y = dataset[:,22]
# fit model no training data
model = XGBClassifier()
model.fit(X, Y)
# plot feature importance
plot_importance(model)
pyplot.show()
