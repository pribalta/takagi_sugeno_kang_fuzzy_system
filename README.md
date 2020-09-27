# Takagi-Sugeno Fuzzy System

## Overview

This repository contains the implementation of a fuzzy classifier following the formulation of a Takagi-Sugeno Fuzzy system.

The classifier is built around two components:
* C-Means clustering with the purpose of computing the firing levels relative to each input
* A logistic regressor that performs the fitting adapted to the conditioned firing levels

The implementation of the C-Means clustering method can be found in [tsfs/clustering.py](tsfs/clustering.py)
The implementation of the fuzzy classifier can be found in [tsfs/clustering.py](tsfs/classifier.py)

## Data

The data used for training is available in [data/iris.csv](data/iris.csv). The samples used for training the classifer
in this repository come from the famous [Iris flower dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set).

The Iris flower data set is a multivariate data set consisting of 50 samples from each of three species of Iris 
(Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width
 of the sepals and petals, in centimeters. Based on the combination of these four features, it is possible to develop a 
 discriminant model to distinguish the species from each other.

## Setup

There are two ways to begin using the code in this repository:

1. Build the docker container that is provided as defined in the Dockerfile (**Preferred**)
2. Ensure you have PyThon 3.6 installed in your system and fetch the additional requiremens specified in ``requirement.txt``


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
Iter: 0 - Loss: 245.5040
Iter: 1 - Loss: 144.6684
Iter: 2 - Loss: 140.6356
Iter: 3 - Loss: 123.1583
Iter: 4 - Loss: 87.8052
Iter: 5 - Loss: 70.4500
Iter: 6 - Loss: 68.9995
Iter: 7 - Loss: 68.9192
Iter: 8 - Loss: 68.9110
Iter: 9 - Loss: 68.9099
Iter: 10 - Loss: 68.9098
Iter: 11 - Loss: 68.9097
Iter: 12 - Loss: 68.9097
Iter: 13 - Loss: 68.9097
Accuracy: 0.88
```

You can experiment with the ``--n_cluster`` flag in order to achieve a better accuracy.

```bash
$ python main.py --dataset data/iris.csv --n_cluster 5
Iter: 0 - Loss: 248.0536
Iter: 1 - Loss: 144.9909
Iter: 2 - Loss: 144.9004
Iter: 3 - Loss: 144.7798
Iter: 4 - Loss: 144.1407
Iter: 5 - Loss: 140.6302
Iter: 6 - Loss: 125.3837
Iter: 7 - Loss: 91.7900
Iter: 8 - Loss: 71.9566
Iter: 9 - Loss: 69.9073
Iter: 10 - Loss: 69.7892
Iter: 11 - Loss: 69.7763
Iter: 12 - Loss: 69.7744
Iter: 13 - Loss: 69.7741
Iter: 14 - Loss: 69.7741
Iter: 15 - Loss: 69.7741
Iter: 16 - Loss: 69.7741
Iter: 17 - Loss: 69.7741
Accuracy: 1.0
```