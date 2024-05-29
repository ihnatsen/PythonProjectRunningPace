import math
from sklearn.metrics import r2_score, mean_squared_error
from scripts.Support.combinatorics import *
from sklearn.linear_model import LinearRegression
from scripts.machine_learning_algorithms.ML import *
from scripts.Support.format_txt import paint as p


class MLR(Algorithm, MetricsR2, MetricsRMSE):

    def __init__(self, target, factors, df):
        super().__init__(target, factors, df)

    def create_model(self):
        return {factors: LinearRegression().fit(self.train[list(factors)], self.train[self.target])
                for factors in get_all_combinations(self.factors) if len(factors) > 1}

    def get_result(self):
        results = self.test[self.target].copy()
        for factors, model in self.create_model().items():
            results[','.join(factors)] = model.predict(self.test[list(factors)])

        return results

    def get_r2_score(self):
        results = self.get_result()
        return {labels: r2_score(results[labels], results[self.target]) for labels in list(results)}

    def get_RMSE(self):
        results = self.get_result()
        return {labels: math.sqrt(mean_squared_error(results[labels], results[self.target])) for labels in list(results)}

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

        features, mods = self.create_model().keys(), list(self.create_model().values())

        features = [[f'{p(value, 'orange'):<30}' for value in values_features] for values_features in features]
        free = [f'{list(values)[0]:+0.3f}' for values in [mod.intercept_ for mod in mods]]
        mods = [[f'{values:+0.3f}' for values in list(mod.coef_[0])] for mod in mods]

        for values_features, values_mod, free_ in zip(features, mods, free):
            print(f'{p(*self.target, 'green')} = ', end='')
            for feature, value in zip(values_features, values_mod, ):
                print(f'{value}*{feature}', end=' ')
            print(free_)


