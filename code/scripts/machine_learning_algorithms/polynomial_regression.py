import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scripts.machine_learning_algorithms.multiple_linear_regression import *
from scripts.machine_learning_algorithms.ML import *


class PR(Algorithm):

    def __init__(self, target, factors, df, degree):
        super().__init__(target, factors, df)
        self.degree = degree
        self.models = self.create_model(factors)
        self.results = self.get_result()

    def get_df(self):
        return self.df

    def create_model(self, factor):
        X = pd.DataFrame(index=self.train.index)
        for degree in range(1, self.degree + 1):
            X[f'{factor}^^{degree}'] = self.train[factor] ** degree
        y = self.train[self.target]
        lr = LinearRegression()
        lr.fit(X, y)
        return lr

    def get_result(self):
        X = pd.DataFrame(index=self.test.index)
        for degree in range(1, self.degree + 1):
            X[f'{self.factors}^^{degree}'] = self.test[self.factors] ** degree
        return self.models.predict(X)

    def get_RMSE(self):
        return math.sqrt(mean_squared_error(self.get_result(), self.test[self.target]))

    def get_r2_score(self):
        return r2_score(self.get_result(), self.test[self.target])

    def print_information(self):
        print(f'Factor: {self.factors}[{p(self.degree, 'orange')}]:')
        print(f'{p('RMSE', 'green')} [{round(self.get_RMSE(), 2)}]')
        print(f'{p('R2 Score', 'green')} [{round(self.get_r2_score(), 2)}]')
        print(f'Model:')
        print(f'{(self.models.coef_).tolist()[0]} {self.models.intercept_}')
