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

class OneR(BaseEstimator, ClassifierMixin) :
    def __init__(self) :
        self.class_ = 0

    def fit(self, x_train, y_train) :
        n_class = [len(unique_labels(y_train))] * len(x_train[0])
        est = KBinsDiscretizer(n_bins=n_class, encode='ordinal', strategy='uniform').fit(x_train)
        X_bin = est.transform(x_train)
        possible_values = []
        for i in range(0, len(x_train[0])) :
            rules_values = dict()
            cross_values = crosstab(X_bin[:,i], y_train)
            for df_row_index, df_row in cross_values.iterrows() :
                df_index = self.best_index(df_row)
                df_name = cross_values.columns[df_index]
                rules_values[df_row_index] = df_name
            possible_values.append(rules_values)
        self.predict_index = self.best_predict_index(X_bin, y_train, possible_values)
        self.predict_table = possible_values[self.predict_index]
    
    def predict(self, x_test, y_test) :
        predict = list()
        n_class = [len(unique_labels(y_test))] * len(x_test[0])
        est = KBinsDiscretizer(n_bins=n_class, encode='ordinal', strategy='uniform').fit(x_test)
        X_bin = est.transform(x_test)
        for index in X_bin[:,self.predict_index] :
            predict.append(self.predict_table[index])
        return predict

    def score(self, x_test, y_test) :
        pred = self.predict(x_test, y_test)
        equals = zip(pred, y_test)
        equals = filter(lambda x: x[0] == x[1], equals)
        return len(list(equals)) / len(list(y_test))
        

    def best_index(self, row) :
        best_index = 0
        max_value = 0

        for index, element in enumerate(row) :
            if element > max_value :
                max_value = element
                best_index = index

        return best_index

    def best_predict_index(self, x_train, y_train, possible_values) :
        best_possible = 0
        best_score = 0.0
        for possible_index, possible in enumerate(possible_values) :
            predict = list()
            for index in x_train[:,possible_index] :
                predict.append(possible[index])
                
            equals = zip(predict, y_train)
            equals = filter(lambda x: x[0] == x[1], equals)
            score = len(list(equals)) / len(list(y_train))

            if score >= best_score :
                best_score = score
                best_possible = possible_index

        return best_possible

if __name__ == '__main__' :
    iris = datasets.load_digits()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.4, random_state = 0)

    classifier = OneR()
    classifier.fit(x_train, y_train)

    print(f"Predict: {classifier.predict(x_test, y_test)}")
    print(f"Score: {classifier.score(x_test, y_test)}")