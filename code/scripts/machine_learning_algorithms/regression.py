import math

import pandas as pd
from matplotlib import pyplot as plt

from scripts.machine_learning_algorithms.ML import *
import sklearn.linear_model as sk
from sklearn.metrics import r2_score, mean_squared_error
from scripts.Support.format_txt import paint as p


class LinearRegression(Algorithm, MetricsR2, MetricsRMSE):

    def __init__(self, df: pd.DataFrame, factors: list[str], target: list[str]):
        self.target = target
        self.factors = factors
        self.df = df

        # 20 %
        self.test = df.sample(frac=0.2)
        # 80 %
        self.train = df.drop(self.test.index)

        self.moduls = [self.create_model(factor) for factor in factors]
        self.results = self.get_result()

    def get_result(self) -> pd.DataFrame:
        results = self.test[self.target].copy()
        for mod, factor in zip(self.moduls, self.factors):
            results[factor] = mod.predict(self.test[factor].to_numpy().reshape(-1, 1))
        return results

    def get_r2_score(self):
        return {factor: r2_score(self.results[self.target], self.results[factor]) for factor in self.factors}

    def get_RMSE(self):
        return {factor: math.sqrt(mean_squared_error(self.results[self.target], self.results[factor])) for factor in self.factors}

    def create_model(self, factor: str):
        x = self.train[factor].to_numpy().reshape(-1, 1)
        y = self.train[self.target].to_numpy()
        model = sk.LinearRegression()
        model.fit(x, y)
        return model

    def print_information(self):
        print(f'{p('RMSE', 'green')}')
        for factor, value in sorted(self.get_RMSE().items(), key=lambda v: v[1]):
            print(f'{factor} -- {round(value, 3)}')
        print()

        print(f'{p('R2 Score', 'green')} 🟥 >= 0.5, 🟦 < 0.5')
        for factor, value in sorted(self.get_r2_score().items(), key=lambda v: v[1], reverse=True):
            if value >= 0.5:
                print(f'{factor} -- {p(round(value, 3), 'red')}')

            else:
                print(f'{factor} -- {p(round(value, 3), 'blue')}')
        print()

        for factor, mod in zip(self.factors, self.moduls):
            print(f'{p(self.target[0], 'green')} = {round(mod.coef_[0][0], 2):-} * {p(factor, 'orange')} '
                  f'{round(mod.intercept_[0], 2):-}')

        for factor, mod in zip(self.factors, self.moduls):
            plt.scatter(self.train[factor], self.train[self.target], color='blue', label='Training data')
            plt.scatter(self.test[factor], self.test[self.target], color='green', label='Testing data')
            plt.plot(self.df[factor], mod.predict(self.df[factor].to_numpy().reshape(-1, 1)), color='red', label='Linear regression')
            plt.title(f'Liner Regression for \'{factor}\'')
            plt.xlabel(f'Values for {factor}')
            plt.ylabel(f'Values for {self.target}')
            plt.legend()
            plt.show()
