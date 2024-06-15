from sklearn.svm import SVR as SVR_
from scripts.machine_learning_algorithms.ML import Algorithm
from typing import Literal

class SVR(Algorithm):
    def __init__(self, target, factors, df,
                 core: Literal['linear', 'pole', 'rbf', 'sigmoid', 'precomputed']):
        self.core = core
        super().__init__(target, factors, df)

    def create_model(self):
        mod = SVR_(self.core)
        mod.fit(self.train[self.factors], self.train[self.target])
        return mod

    def print_information(self):
        super().print_information()
