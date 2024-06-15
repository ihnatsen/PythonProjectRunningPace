
from scripts.Support.combinatorics import *
from sklearn.linear_model import LinearRegression
from scripts.machine_learning_algorithms.ML import *
from scripts.Support.format_txt import paint as p


class MLR(Algorithm):
    def create_model(self):
        model = LinearRegression()
        model.fit(self.train[self.factors], self.train[self.target])
        return model

    def print_information(self):
        super().print_information()
        factors = [f'{weight:+0.3f}{p(name, 'orange'):>18}' for name, weight in zip(self.factors,
                                                                                    self.model.coef_.ravel().tolist())]
        factors.append(f'{self.model.intercept_.ravel().tolist()[0]:+0.3f}')
        print(f'{p(self.target[0], 'orange')} = {' '.join(factors)}')

