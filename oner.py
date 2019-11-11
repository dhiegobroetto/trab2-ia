import warnings

def warn (*args, **kargs) :
    pass
warnings.warn = warn

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from collections import Counter

class OneR() :
    def __init__(self) :
        self.class_ = 0

    def fit(self, x_train, y_train) :
        # [0, 0, 1, 2] = 0
        group_by = Counter(y_train)
        self.class_ = max(group_by.items(), key = lambda x: x[1])[0]
    
    def predict(self, x_test, y_test) :
        return [self.class_] * len(y_test)

    # def score(self, x_test, y_test) :
    #     pred = self.predict(x_test, y_test)
    #     values = 0
    #     for i in range(len(pred)) :
    #         if pred[i] == y_test[i] :
    #             values += 1
    #     return values / len(pred)

iris = datasets.load_iris()
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.4, random_state = 0)

n_class = len(iris['target_names'])

est = KBinsDiscretizer(n_bins=[n_class, n_class, n_class, n_class], encode='ordinal').fit(x_train)
X_bin = est.transform(x_train)

print(X_bin)

oner = OneR()
oner.fit(X_bin, y_train)

equals = zip(oner.predict(x_test, y_test), y_test)