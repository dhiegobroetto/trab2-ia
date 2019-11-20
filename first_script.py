import warnings

def warn (*args, **kargs) :
    pass
warnings.warn = warn

from pandas import DataFrame
from collections import Counter
import os

from zeror import ZeroR
from oner import OneR
from sklearn.naive_bayes import GaussianNB

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import preprocessing

from seaborn import boxplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import show, title, xlabel, ylabel

# ------- Datasets ------- #

ds_iris = datasets.load_iris()
ds_digits = datasets.load_digits()
ds_wine = datasets.load_wine()
ds_breast_cancer = datasets.load_breast_cancer()

# ------- Classifiers ------- #

classifiers = dict()
classifiers['ZeroR'] =  ZeroR()
classifiers['OneR'] = OneR()
classifiers['GaussianNB'] = GaussianNB()

# ------- DataFrames with Media / DP / Score ------- #

df_iris = DataFrame(index = classifiers.keys(), columns = ['media', 'dp', 'score'])
df_digits = DataFrame(index = classifiers.keys(), columns = ['media', 'dp', 'score'])
df_wine = DataFrame(index = classifiers.keys(), columns = ['media', 'dp', 'score'])
df_breast_cancer = DataFrame(index = classifiers.keys(), columns = ['media', 'dp', 'score'])

# ------- Score from all classifiers for all datasets ------- #

for classifier_index, classifier in classifiers.items() :
    score = cross_val_score(estimator = classifier, X = ds_iris.data, y = ds_iris.target, cv = 10)
    df_iris.loc[classifier_index] = [score.mean(), score.std(), score]

    score = cross_val_score(estimator = classifier, X = ds_digits.data, y = ds_digits.target, cv = 10)
    df_digits.loc[classifier_index] = [score.mean(), score.std(), score]

    score = cross_val_score(estimator = classifier, X = ds_wine.data, y = ds_wine.target, cv = 10)
    df_wine.loc[classifier_index] = [score.mean(), score.std(), score]

    score = cross_val_score(estimator = classifier, X = ds_breast_cancer.data, y = ds_breast_cancer.target, cv = 10)
    df_breast_cancer.loc[classifier_index] = [score.mean(), score.std(), score]

# ------- Initiate boxplot dataframes ------- #

box_iris = DataFrame()
box_digits = DataFrame()
box_wine = DataFrame()
box_breast_cancer = DataFrame()

# ------- Obtains score from all classifiers to all datasets ------- #

for classifier_name, classifier in classifiers.items() :
    box_iris[classifier_name] = df_iris.at[classifier_name, 'score']
    box_digits[classifier_name] = df_digits.at[classifier_name, 'score']
    box_wine[classifier_name] = df_wine.at[classifier_name, 'score']
    box_breast_cancer[classifier_name] = df_breast_cancer.at[classifier_name, 'score']

# ------- Boxplots ------- #

boxplot(data = box_iris, showmeans = True)
title("Boxplot com scores de cada classificador no dataset Iris")
xlabel('Classificadores')
ylabel('Score')
show()
boxplot(data = box_digits, showmeans = True)
title("Boxplot com scores de cada classificador no dataset Digits")
xlabel('Classificadores')
ylabel('Score')
show()
boxplot(data = box_wine, showmeans = True)
title("Boxplot com scores de cada classificador no dataset Wine")
xlabel('Classificadores')
ylabel('Score')
show()
boxplot(data = box_breast_cancer, showmeans = True)
title("Boxplot com scores de cada classificador no dataset Breast Cancer")
xlabel('Classificadores')
ylabel('Score')
show()

# # ------- CSVs ------- #
# df_iris.drop(['score'], axis = 1).to_csv(f"results/iris_first_script.csv")
# df_digits.drop(['score'], axis = 1).to_csv(f"results/digits_first_script.csv")
# df_wine.drop(['score'], axis = 1).to_csv(f"results/wine_first_script.csv")
# df_breast_cancer.drop(['score'], axis = 1).to_csv(f"results/breast_cancer_first_script.csv")

