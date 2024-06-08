import math
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score, mean_squared_error
from scripts.machine_learning_algorithms.ML import *


class ST(Algorithm):
    def __init__(self, target, factors, df):
        super().__init__(target, factors, df)
        self.model = self.create_model()

    def get_result(self):
        return self.model.predict(self.test[self.factors])

    def get_r2_score(self):
        return r2_score(self.get_result(), self.test[self.target])

    def get_RMSE(self):
        return math.sqrt(mean_squared_error(self.get_result(), self.test[self.target]))

    def create_model(self):
        tree = DecisionTreeClassifier()
        tree.fit(self.train[self.factors], self.train[self.target])
        return tree

    def print_information(self):
        print('RMSE', self.get_RMSE())
        print('R2 score', self.get_r2_score())
        print(mean_squared_error(self.train[self.factors]))
