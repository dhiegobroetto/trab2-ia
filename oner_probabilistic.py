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
from random import seed, uniform, choice
from timeit import default_timer

class ProbabilisticOneR(BaseEstimator, ClassifierMixin) :
    def __init__(self) :
        self.class_ = 0

    def fit(self, x_train, y_train) :
        self.class_ = unique_labels(y_train)
        
        n_class = [len(unique_labels(y_train))] * len(x_train[0])
        est = KBinsDiscretizer(n_bins=n_class, encode='ordinal', strategy='uniform').fit(x_train)
        X_bin = est.transform(x_train)

        # Gerando tabela de regras através da roleta
        possible_values = []
        for i in range(0, len(x_train[0])) :
            cross_values = crosstab(X_bin[:,i], y_train)
            rules_values = dict()
            for df_row_index, df_row in cross_values.iterrows() :
                df_index = self.best_index_roulette(df_row)
                df_name = cross_values.columns[df_index]
                rules_values[df_row_index] = df_name
            possible_values.append(rules_values)

        self.predict_index = self.best_predict_index(X_bin, y_train, possible_values)
        self.predict_table = possible_values[self.predict_index]
        self.initial_predict_table = crosstab(X_bin[:,self.predict_index], y_train)

    def predict(self, x_test) :
        predict = list()
        n_class = [len(self.class_)] * len(x_test[0])

        est = KBinsDiscretizer(n_bins=n_class, encode='ordinal', strategy='uniform').fit(x_test)
        X_bin = est.transform(x_test)
        
        for index in X_bin[:,self.predict_index] :
            if index in self.predict_table:
                predict.append(self.predict_table[index])
            else :
                predict.append(choice(self.predict_table))

            # Aplicando a roleta novamente em toda tabela de regras
            rules_values = dict()
            for df_row_index, df_row in self.initial_predict_table.iterrows() :
                df_index = self.best_index_roulette(df_row)
                df_name = self.initial_predict_table.columns[df_index]
                rules_values[df_row_index] = df_name

            self.predict_table = rules_values

        return predict

    def score(self, x_test, y_test) :
        pred = self.predict(x_test)
        equals = zip(pred, y_test)
        equals = filter(lambda x: x[0] == x[1], equals)
        return len(list(equals)) / len(list(y_test))

    def best_index_roulette(self, row) :
        seed(default_timer())
        total = 0
        for index in row.keys() :
            total += row[index]

        elements_roulette = []
        for index in row.keys() :
            elements_roulette.append([index, (row[index]/total)])
        sort_list(elements_roulette)

        rand = uniform(0, 1)
        percent = 0
        for i in range(0, len(elements_roulette)) :
            if rand <= (elements_roulette[i][1] + percent) :
                return elements_roulette[i][0]
            percent += elements_roulette[i][1]
        return -1

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

def sort_list(list) :
    list.sort(key = lambda pos: pos[1], reverse = False)

if __name__ == '__main__' :
    iris = datasets.load_iris()
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size = 0.4, random_state = 0)

    classifier = ProbabilisticOneR()
    classifier.fit(x_train, y_train)

    print(f"Predict: {classifier.predict(x_test)}")
    print(f"Score: {classifier.score(x_test, y_test)}")