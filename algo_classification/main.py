#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from load_data import load 

print("Bienvenue sur notre programme de test d'algorithmes de classification")
print("Dans un premier temps choisisez la base de donnée parmi les suivantes :")
print("\t- iris")
print("\t- chiffre")
print("\t- vin")
print("\t- cancer")
bdd = input("Entrez votre choix : ")
affichage = input("Voulez vous les informations sur la base de données (o/n) : ")

data = load(bdd, (affichage == "o"))

import sys
print("Python version:", sys.version)

import pandas as pd
print("pandas version:", pd.__version__)

import matplotlib
print("matplotlib version:", matplotlib.__version__)

import numpy as np
print("NumPy version:", np.__version__)

import scipy as sp
print("SciPy version:", sp.__version__)

import IPython
print("IPython version:", IPython.__version__)

import sklearn
print("scikit-learn version:", sklearn.__version__)

from sklearn.datasets import load_iris
iris_dataset = load_iris()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0)

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),marker='o', hist_kwds={'bins': 20}, s=60,alpha=.8, cmap=mglearn.cm3)