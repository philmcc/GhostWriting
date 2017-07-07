# Tune learning_rate
from numpy import loadtxt
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
# load data
url = "../Datasets/stage_1_dataset_05.csv"
dataset = loadtxt(url, delimiter="|")
# split data into X and y
X = dataset[:,0:22]
Y = dataset[:,22]

# grid search params
n_estimators = [50, 100, 150, 200]
max_depth = [2, 4, 6, 8]
learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
subsample = [0.7,0.8,0.9]
#n_estimators = [150, 200]
#max_depth = [2, 4]
#learning_rate = [0.01, 0.1]
#subsample = [0.8,0.9]
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, subsample=subsample)

# Split dataframe
kfold = StratifiedKFold(Y, n_folds=10, shuffle=True, random_state=7)
seed=7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
    random_state=seed)
#################
#################

# Grid search
cv_params = param_grid
#cv_params = {'learning_rate': [0.1, 0.01], 'subsample': [0.7,0.8,0.9]}

ind_params = {'seed':0, 'colsample_bytree': 0.8, 
             'objective': 'binary:logistic', 'min_child_weight': 1}
optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
                            cv_params, 
                             scoring = 'accuracy', cv = 5, n_jobs = -1)
optimized_GBM.fit(X_train, y_train)

print("Best: %f using %s" % (optimized_GBM.best_score_, optimized_GBM.best_params_))

################
################






