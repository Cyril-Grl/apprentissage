#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def load(database="iris", affichage=True):
    if database == "iris":
        from sklearn.datasets import load_iris
        data = load_iris()
    elif database == "chiffre":
        from sklearn.datasets import load_digits
        data = load_digits()
    elif database == "vin":
        from sklearn.datasets import load_wine
        data = load_wine()
    elif database == "cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer()
    else :
        print("\nErreur\nLa base de donnée de base doit être :")
        print("\t- iris")
        print("\t- chiffre")
        print("\t- vin")
        print("\t- cancer")
        exit(0)

    if affichage :
        print(data['DESCR'])

    return data
