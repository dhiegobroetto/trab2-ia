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
from scipy.spatial import distance
from math import inf
from sklearn.base import BaseEstimator, ClassifierMixin

class Centroid(BaseEstimator, ClassifierMixin) :
    def __init__(self) :
        self.centroids = {}

    def fit(self, x_train, y_train) :
        groups = Counter(y_train)
        
        # Soma todos os pontos
        for i in range(0, len(x_train)) :
            if y_train[i] not in self.centroids :
                self.centroids[y_train[i]] = x_train[i]
            else :    
                self.centroids[y_train[i]] += x_train[i]

        # Ponto médio
        for index, value in self.centroids.items() :
            self.centroids[index] = value / groups[index]

    def predict(self, x_test) :
        predict = []
        for i in range(0, len(x_test)) :
            best_index = 0
            best_dist = inf
            for index, value in self.centroids.items() :
                dist = distance.euclidean(x_test[i], value)
                if(dist <= best_dist) :
                    best_index = index
                    best_dist = dist
            predict.append(best_index)
        return predict

    def score(self, x_test, y_test) :
        pred = self.predict(x_test)
        equals = zip(pred, y_test)
        equals = filter(lambda x: x[0] == x[1], equals)
        return len(list(equals)) / len(list(y_test))

if __name__ == '__main__' :
    iris = datasets.load_digits()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.4, random_state = 0)

    classifier = Centroid()
    classifier.fit(x_train, y_train)

    print(f"Predict: {classifier.predict(x_test)}")
    print(f"Score: {classifier.score(x_test, y_test)}")
