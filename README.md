# Logistic multiclass regression vs. Neural Network: demo

## Overview
Comparison between simple logistic multiclass regression (1 neuron, no hidden layers) and a neural network with one hidden layer.

## Description
First, a spiral dataset is created and then classified using two approaches:

1. *Multiclass logistic regression*, containing only one neuron and using *softmax* activation. The resulting decision boundary is linear and accuracy for the test dataset is close to only **50%**.
2. *Neural network*: containing one hidden layer with 100 nodes and *ReLU* activation, followed by an output *softmax* layer. The resulting decision boundary is non-linear, and accuracy for the test dataset is close to **99%**.

## Conclusion
The use of a hidden layer provides better accuracy since the model can better learn the non-linearities of the decision boundaries in the dataset.

## Dependencies
numpy==1.14.0, matplotlib==2.0.0, Keras==2.1.3, Tensorflow==1.4.0

## Credits
Credit for most of the code here goes to [Andrej Karpathy](http://cs231n.github.io/neural-networks-case-study/). I've merely implemented his algorithms for use with the Keras library.