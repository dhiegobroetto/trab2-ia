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
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer

from seaborn import boxplot as bxpt
import matplotlib.pyplot as plt


ds_iris = datasets.load_iris()
ds_digits = datasets.load_digits()
ds_wine = datasets.load_wine()
ds_breast_cancer = datasets.load_breast_cancer()

# class_zeror =
# class_oner = OneR()
# class_oner_prob = OneRProb()
# class_gaussiannb = GaussianNB()
classifiers = dict()
classifiers['ZeroR'] =  ZeroR()
classifiers['OneR'] = OneR()
# classifiers.append(class_oner_prob)
classifiers['GaussianNB'] = GaussianNB()

df_iris = DataFrame(index=classifiers.keys(), columns=['media', 'dp', 'scores'])
df_digits = DataFrame(index=classifiers.keys(), columns=['media', 'dp', 'scores'])
df_wine = DataFrame(index=classifiers.keys(), columns=['media', 'dp', 'scores'])
df_breast_cancer = DataFrame(index=classifiers.keys(), columns=['media', 'dp', 'scores'])

for classifier_index, classifier in classifiers.items() :
    scores = cross_val_score(estimator=classifier, X=ds_iris.data, y=ds_iris.target, cv=10)
    df_iris.loc[classifier_index] = [scores.mean(), scores.std(), scores]

    # scores = cross_val_score(estimator=classifier, X=ds_digits.data, y=ds_digits.target, cv=10)
    # df_digits.loc[classifier_index] = [scores.mean(), scores.std(), scores]

    # scores = cross_val_score(estimator=classifier, X=ds_wine.data, y=ds_wine.target, cv=10)
    # df_wine.loc[classifier_index] = [scores.mean(), scores.std(), scores]

    # scores = cross_val_score(estimator=classifier, X=ds_breast_cancer.data, y=ds_breast_cancer.target, cv=10)
    # df_breast_cancer.loc[classifier_index] = [scores.mean(), scores.std(), scores]

print(df_iris.keys()[0])
# print(df_iris)
# for ds_index, dataset in datasets.items() :
#     print(f"Dataset: {ds_index}")
#     for classifier_index, classifier in classifiers.items() :
#         print(f"Classifier: {classifier_index}")
#         scores = cross_val_score(estimator=classifier, X=dataset.data, y=dataset.target, cv=10)
#         ds_frames[ds_index].loc[classifier_index] = [scores.mean(), scores.std(), scores]

# for ds_index, frame in ds_frames.items() :
#     print(f"{ds_index} (boxplots/part1/{ds_index}_part1.png)")
#     print(frame)

#     boxplot = DataFrame()

#     for classifier_name, classifier in classifiers.items() :
#         boxplot[classifier_name] = frame.at[classifier_name, 'scores']

#     bxpt(data = boxplot, showmeans = True)
    
#     plt.title(ds_index)
#     plt.ylabel('scores')
#     plt.xlabel('classificadores')

#     try :
#         os.makedirs('boxplots/part1')
#     except FileExistsError :
#         pass

#     plt.savefig(f"boxplots/part1/{ds_index}_part1.png")
    
#     plt.cla()
#     plt.clf()

# plt.close()
