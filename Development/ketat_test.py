# Use scikit-learn to grid search the batch size and epochs
import numpy
from sklearn.grid_search import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD

seed = 7
numpy.random.seed(seed)
# Load the dataset
dataset = numpy.loadtxt('../Datasets/training_dataset_03.csv', delimiter="|")
# split data into X and y
X = dataset[:,0:46]
Y = dataset[:,46]

def create_model(learn_rate=0.1, momentum=0.4):
	# Define and Compile
	model = Sequential()
	model.add(Dense(46, input_dim=46, init='uniform', activation='relu'))
	model.add(Dense(150, init='uniform', activation='relu'))
	#model.add(Dense(23, init='uniform', activation='relu'))
	model.add(Dense(1, init='uniform', activation='sigmoid'))
	optimizer = SGD(lr=learn_rate, momentum=momentum)

	model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])
	return model 



########################
# Tuning the network training parameters Batch Size and Number of Epochs
# define the grid search parameters
# create model
#model = KerasClassifier(build_fn=create_model, verbose=0)
#batch_size = [10, 20, 40, 60, 80, 100]
#epochs = [10, 50, 100]
#param_grid = dict(batch_size=batch_size, nb_epoch=epochs)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(X, Y)
# summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#for params, mean_score, scores in grid_result.grid_scores_:
#    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
#########################


#########################
#Tuning Training Optimization Algorithm
# create model
#model = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=40, verbose=0)
# define the grid search parameters
#optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
#param_grid = dict(optimizer=optimizer)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(X, Y)
# summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#for params, mean_score, scores in grid_result.grid_scores_:
#    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
#########################

#########################
# Tuning Learning Rate and momentum
# create model
#model = KerasClassifier(build_fn=create_model, nb_epoch=100, batch_size=10, verbose=0)
# define the grid search parameters
#learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
#momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
#param_grid = dict(learn_rate=learn_rate, momentum=momentum)
#grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
#grid_result = grid.fit(X, Y)
# summarize results
#print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#for params, mean_score, scores in grid_result.grid_scores_:
#    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))
#########################


model = Sequential()
model.add(Dense(12, input_dim=44, init=✬uniform✬, activation=✬relu✬))
model.add(Dense(8, init=✬uniform✬, activation=✬relu✬))
model.add(Dense(1, init=✬uniform✬, activation=✬sigmoid✬))
model.compile(loss=✬binary_crossentropy✬ , optimizer=✬adam✬, metrics=[✬accuracy✬])
# Fit the model
model.fit(X, Y, nb_epoch=150, batch_size=10)
# Evaluate the model
scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
#########################
# Train using specific parameters
#model1=create_model()
# Fit the model
#model1.fit(X, Y, nb_epoch=100, batch_size=40)
# Evaluate the model
#scores = model1.evaluate(X, Y)
#print("%s: %.2f%%" % (model1.metrics_names[1], scores[1]*100))

#########################
# Run predictions
# create model
#model = KerasClassifier(build_fn=create_model, verbose=0)
#predictions = model.predict(X)
# round predictions
#rounded = [round(x) for x in predictions]
#print(rounded)
##########################