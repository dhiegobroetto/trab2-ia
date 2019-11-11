import warnings

def warn (*args, **kargs) :
    pass
warnings.warn = warn

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from collections import Counter

class ZeroR() :
    def __init__(self) :
        self.class_ = 0

    def fit(self, x_train, y_train) :
        # [0, 0, 1, 2] = 0
        group_by = Counter(y_train)
        group = max(group_by.items(), key = lambda x: x[1])[0]
        self.class_ = group
    
    def predict(self, x_test, y_test) :
        return [self.class_] * len(y_test)

    def score(self, x_test, y_test) :
        pred = self.predict(x_test, y_test)
        values = 0
        for i in range(len(pred)) :
            if pred[i] == y_test[i] :
                values += 1
        return values / len(pred)

iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.4, random_state = 0)

classifier = ZeroR()

params = [0, 1, 2, 3]
best_params = []
opt = 0

classifier.fit(np.array([x_train[:,0]]).T, y_train)
print(classifier.score(np.array([x_test[:,0]]).T, y_test))