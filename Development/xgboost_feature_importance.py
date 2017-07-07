# plot feature importance using built-in function
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
# load data
dataset = loadtxt('datasets/training_dataset_03.csv', delimiter="|")
# split data into X and y
X = dataset[:,0:46]
Y = dataset[:,46]
# fit model no training data
model = XGBClassifier()
model.fit(X, Y)
# plot feature importance
plot_importance(model)
pyplot.show()
