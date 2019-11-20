import warnings

def warn (*args, **kargs) :
    pass
warnings.warn = warn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.multiclass import unique_labels
from collections import Counter
from pandas import crosstab
from sklearn.base import BaseEstimator, ClassifierMixin

class Centroid(BaseEstimator, ClassifierMixin) :
    def __init__(self) :
        self.class_ = 0

    def fit(self, x_train, y_train) :
        print(Counter(y_train))
        val = 0
        # val += x_train[0]
        val += x_train[1]
        print(val)

    def predict(self, x_test, y_test) :
        pass

    def score(self, x_test, y_test) :
        pass

if __name__ == '__main__' :
    iris = datasets.load_digits()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.4, random_state = 0)

    classifier = Centroid()
    classifier.fit(x_train, y_train)

    # print(f"Predict: {classifier.predict(x_test, y_test)}")
    # print(f"Score: {classifier.score(x_test, y_test)}")
