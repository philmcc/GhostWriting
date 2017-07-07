# Tune learning_rate
from numpy import loadtxt
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
## load data
url = "../Datasets/stage_1_dataset_05.csv"
dataset = loadtxt(url, delimiter="|")
# split data into X and y
X = dataset[:,0:22]
Y = dataset[:,22]

#dataset2 = loadtxt('Datasets/testing_dataset_reduced_04.csv', delimiter="|")
#X2 = dataset2[:,0:22]
#Y2 = dataset2[:,22]
# grid search params
#n_estimators = [50, 100, 150, 200]
#max_depth = [2, 4, 6, 8]
#learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
#subsample = [0.7,0.8,0.9]
n_estimators = [150, 200]
max_depth = [2, 4]
learning_rate = [0.01, 0.1]
subsample = [0.8,0.9]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample)

# Split dataframe
kfold = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=7)
seed=7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
    random_state=seed)


xgdmat = xgb.DMatrix(X_train, y_train) # Create our DMatrix to make XGBoost more efficient
our_params = {'eta': 0.1, 'seed':0, 'subsample': 0.9, 
            'objective': 'binary:logistic', 'max_depth':8, 'min_child_weight':1, 'n_estimators':150, 'learning_rate':0.1} 

final_gb = xgb.train(our_params, xgdmat, num_boost_round = 432,  verbose_eval=False)

testdmat = xgb.DMatrix(X_test)

y_pred = final_gb.predict(testdmat) # Predict using our testdmat

y_pred[y_pred > 0.5] = 1
y_pred[y_pred <= 0.5] = 0
print y_pred

print accuracy_score(y_pred, y_test), 1-accuracy_score(y_pred, y_test)


#testdmat = xgb.DMatrix(X2)

#y_pred = final_gb.predict(testdmat) # Predict using our testdmat

#y_pred[y_pred > 0.5] = 1
#y_pred[y_pred <= 0.5] = 0
#print y_pred

#print accuracy_score(y_pred, Y2), 1-accuracy_score(y_pred, Y2)

