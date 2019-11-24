import warnings

def warn (*args, **kargs) :
    pass
warnings.warn = warn

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from collections import Counter
from sklearn.base import BaseEstimator, ClassifierMixin


class ZeroR(BaseEstimator, ClassifierMixin) :
    def __init__(self) :
        self.class_ = 0

    def fit(self, x_train, y_train) :
        group_by = Counter(y_train)
        group = max(group_by.items(), key = lambda x: x[1])[0]
        self.class_ = group
    
    def predict(self, x_test) :
        return [self.class_] * len(x_test)

    def score(self, x_test, y_test) :
        x_test = np.array([x_test[:,0]]).T
        pred = self.predict(x_test)
        values = 0
        for i in range(len(pred)) :
            if pred[i] == y_test[i] :
                values += 1
        return values / len(pred)

if __name__ == '__main__' :
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.4, random_state = 0)

    classifier = ZeroR()
    classifier.fit(x_train, y_train)
    
    print(f"Predict: {classifier.predict(x_test)}")
    print(f"Score: {classifier.score(x_test, y_test)}")