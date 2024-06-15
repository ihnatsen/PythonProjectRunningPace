import numpy
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import BaggingRegressor
from scripts.machine_learning_algorithms.ML import *
from scripts.Support.format_txt import paint as p
from scripts.Support.calculate import get_coefficients
import pandas as pd


class SLR(Algorithm):
    def create_model(self):
        X = self.train[self.factors]
        y = self.train[self.target]
        model = BaggingRegressor(n_estimators=50, n_jobs=-1)
        model.fit(X, y.values.ravel())
        return model

    def print_information(self):

        super().print_information()

        point_one = [
            1,
            self.model.predict(pd.DataFrame({self.factors[0]: [1]}))[0]
        ]

        point_two = [
            self.test[self.factors].values[1][0],
            self.get_result()[1]
        ]
        coef, free = get_coefficients(point_one, point_two)
        print(f'{p(self.target[0], 'orange')} = {coef:0.4f}{p(self.factors[0], 'green')} {free:+0.4f}')

        point_one = [
            min(self.df[self.factors].values),
            min(self.df[self.factors].values) * coef + free
        ]
        point_two = [
            max(self.df[self.factors].values),
            max(self.df[self.factors].values) * coef + free
        ]

        plt.scatter(self.train[self.factors], self.train[self.target], color='blue', label='Training data')
        plt.scatter(self.test[self.factors], self.test[self.target], color='green', label='Testing data')
        plt.plot(*zip(point_one, point_two),
                 color='red',
                 label='Linear regression')

        plt.title(f'Liner Regression for \'{self.factors}\'')
        plt.xlabel(f'Values for {self.factors}')
        plt.ylabel(f'Values for {self.target}')
        plt.legend()
        plt.show()
