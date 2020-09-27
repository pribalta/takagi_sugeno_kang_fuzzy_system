# Takagi-Sugeno-Kang Fuzzy System

## Overview

This repository contains the implementation of a fuzzy classifier following the formulation of a Takagi-Sugeno-Kang Fuzzy system.

The classifier is built around two components:
* C-Means clustering with the purpose of computing the firing levels relative to each input
* A logistic regressor that performs the fitting adapted to the conditioned firing levels

The implementation of the C-Means clustering method can be found in [tsk/clustering.py](tsk/clustering.py)
The implementation of the fuzzy classifier can be found in [tsk/classifier.py](tsk/classifier.py)

## Data

The data used for training is available in [data/iris.csv](data/iris.csv). The samples used for training the classifer
in this repository come from the famous [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set).

The Iris flower data set is a multivariate data set consisting of 50 samples from each of three species of Iris 
(Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: The length and the width
 of the sepals and petals, in centimeters. Based on the combination of these four features, it is possible to develop a 
 discriminant model to distinguish the species from each other.

## Setup

There are two ways to begin using the code in this repository:

1. Build the docker container that is provided as defined in the Dockerfile (**Preferred**)
2. Ensure you have PyThon 3.6 installed in your system and fetch the additional requiremens specified in ``requirement.txt``

```
$ pip3 install -r requirements.txt
```


## Quickstart guide 

Once you have completed the setup, you should be good to go and can start playing with the code.

When you run the classifier, the following operations are taking place:

1. Data is loaded from storage
2. Data is shuffled
3. Data is split in train and test sets
4. Train samples are fit by the classifier
5. Accuracy is predicted over the unseen test data

You can launch the example by running:

```bash
$ python main.py --dataset data/iris.csv [--n_cluster n] 
```

The output displays the fitting process and shows the achieved accuracy:

```bash
$ python main.py --dataset data/iris.csv --n_cluster 2
Loaded dataset from: data/iris.csv
Number of training samples: 125
Number of test samples: 25
Fitting classifier to data:
	iter: 0 - loss: 218.6259
	iter: 1 - loss: 116.7424
	iter: 2 - loss: 89.3245
	iter: 3 - loss: 67.0880
	iter: 4 - loss: 63.8012
	iter: 5 - loss: 63.6859
	iter: 6 - loss: 63.6811
	iter: 7 - loss: 63.6807
	iter: 8 - loss: 63.6806
	iter: 9 - loss: 63.6806
	iter: 10 - loss: 63.6806
	iter: 11 - loss: 63.6806
Predicting unseen data:
	accuracy: 0.88
```

You can experiment with the ``--n_cluster`` flag in order to achieve a better accuracy.

```bash
$ python main.py --dataset data/iris.csv --n_cluster 5
Loaded dataset from: data/iris.csv
Number of training samples: 125
Number of test samples: 25
Fitting classifier to data:
	iter: 0 - loss: 223.7359
	iter: 1 - loss: 140.9750
	iter: 2 - loss: 140.1536
	iter: 3 - loss: 135.9419
	iter: 4 - loss: 119.5171
	iter: 5 - loss: 87.6509
	iter: 6 - loss: 70.6772
	iter: 7 - loss: 69.0612
	iter: 8 - loss: 68.9887
	iter: 9 - loss: 68.9825
	iter: 10 - loss: 68.9817
	iter: 11 - loss: 68.9816
	iter: 12 - loss: 68.9816
	iter: 13 - loss: 68.9816
	iter: 14 - loss: 68.9816
Predicting unseen data:
	accuracy: 1.0
```
