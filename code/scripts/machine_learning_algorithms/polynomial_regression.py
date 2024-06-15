import pandas as pd

from scripts.machine_learning_algorithms.ML import *
from sklearn.linear_model import LinearRegression


class PR(Algorithm):

    def __init__(self, target, factors, df, degree):
        self.degree = degree
        super().__init__(target, factors, df)

    def create_model(self):
        X = pd.DataFrame(index=self.train.index)
        for degree in range(1, self.degree + 1):
            X[f'{self.factors[0]}^^{degree}'] = self.train[self.factors] ** degree
        y = self.train[self.target]
        lr = LinearRegression()
        lr.fit(X, y)
        return lr

    def get_result(self):
        X = pd.DataFrame(index=self.test.index)
        for degree in range(1, self.degree + 1):
            X[f'{self.factors[0]}^^{degree}'] = self.test[self.factors] ** degree
        return self.model.predict(X)

    def print_information(self):
        print(f'Factor: {self.factors}[{p(self.degree, 'orange')}]:')
        super().print_information()
        print(f'Model:')
        print(f'{(self.model.coef_).tolist()[0]} {self.model.intercept_}')
