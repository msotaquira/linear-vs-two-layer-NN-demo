import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import np_utils

## Function definitions
def plot_class_distribution(X,Y,title):
	plt.scatter(X[:,0],X[:,1],c=Y, s=40, cmap=plt.cm.Set1, edgecolors='k')
	plt.title(title)
	plt.show()
	

def plot_decision_boundary(X,Y,model,title):
	# Min and max values, and padding
	x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
	y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
	h = 0.01

	# Generate grid of points
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

	# Predict the function value for the whole grid
	Z = model.predict_classes(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	fig = plt.figure()
	plt.contourf(xx, yy, Z, cmap=plt.cm.Set1, alpha=0.8)
	plt.scatter(X[:, 0], X[:, 1], c=Y, s=40, cmap=plt.cm.Set1, edgecolor='k')
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.title(title)
	plt.show()

## The spiral dataset
N = 500					# Number of points per class
D = 2					# Number of features/example in the dataset
K = 3 					# Number of classes

X = np.zeros((N*K,D)) 				# Input
Y = np.zeros(N*K, dtype='uint8')	# Labels

for j in range(K):
	ix = range(N*j, N*(j+1))
	r = np.linspace(0.0,1,N)									# radius
	t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2		# theta
	X[ix] = np.c_[r*np.sin(t),r*np.cos(t)]
	Y[ix] = j
plot_class_distribution(X,Y,'Original spiral dataset with {} classes'.format(K))

# Training and test datasets (80% and 20%)
permute_indices = np.random.permutation(X.shape[0])
X = X[permute_indices,:]
Y = Y[permute_indices]

idx = int(np.round(0.8*X.shape[0]))
X_train = X[0:idx,:]
Y_train = Y[0:idx]
X_test = X[idx:,:]
Y_test = Y[idx:]

nclasses = 3
Y_train = np_utils.to_categorical(Y_train,nclasses)
Y_test = np_utils.to_categorical(Y_test,nclasses)

## Multiclass logistic regression (no hidden layers) 
# - Input layer: 2 nodes (2 features/example)
# - Output layer: 3 nodes (one out of three possible classes)
np.random.seed(1)

input_dim = X_train.shape[1]
output_dim = 3

model = Sequential()
model.add(Dense(output_dim, input_dim = input_dim, activation='softmax'))

sgd = SGD(lr=0.025)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

nepochs = 50
batch_size = X_train.shape[1]
model.fit(X_train, Y_train, epochs=nepochs, batch_size=batch_size, verbose=2)

score = model.evaluate(X_test, Y_test, verbose=0) 
title = 'Decision boundary for simple logistic regression (1 neuron, no hidden layers).\n Accuracy on test dataset: {:.1f}%'.format(100*score[1])
plot_decision_boundary(X_test,np.argmax(Y_test,axis=1),model,title)

## Neural network classification

# - Input layer: 2 nodes (2 features/example)
# - Hidden layer: 100 nodes, 'ReLU' activation
# - Output layer: 3 nodes (one out of three possible classes), 'softmax' activation

input_dim = X_train.shape[1]
hidden_dim = 100
output_dim = 3

model = Sequential()
model.add(Dense(hidden_dim, input_dim=input_dim, activation='relu'))
model.add(Dense(output_dim, activation='softmax'))

sgd = SGD(lr=0.025)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

nepochs = 100
batch_size = X_train.shape[1]
model.fit(X_train, Y_train, epochs=nepochs, batch_size=batch_size, verbose=2)

score = model.evaluate(X_test, Y_test, verbose=0) 
title = 'Decision boundary for neural network classification(hidden layer with 100 neurons).\n Accuracy on test dataset: {:.1f}%'.format(100*score[1])
plot_decision_boundary(X_test,np.argmax(Y_test,axis=1),model,title)