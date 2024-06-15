import math
from sklearn import tree

from sklearn.metrics import r2_score, mean_squared_error
from scripts.machine_learning_algorithms.ML import *
import matplotlib.pyplot as plt


class ST(Algorithm):
    def __init__(self, target, factors, df, max_depth):
        self.max_depth = max_depth
        super().__init__(target, factors, df)

    def create_model(self):
        _tree = tree.DecisionTreeRegressor(max_depth=self.max_depth)
        _tree.fit(self.train[self.factors], self.train[self.target])
        return _tree

    def print_information(self):
        print('RMSE', self.get_RMSE())
        print('R2 score', self.get_r2_score())
        tree.plot_tree(self.model, feature_names=self.factors,
               fontsize=5,
               rounded=True,
               filled=True,
               impurity=True)
        plt.show()
