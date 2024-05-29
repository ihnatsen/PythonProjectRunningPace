from abc import ABC, abstractmethod


class Algorithm(ABC):

    def __init__(self, target, factors, df):
        self.target = target
        self.factors = factors
        self.df = df
        self.test = df.sample(frac=0.2)
        self.train = df.drop(self.test.index)


    @abstractmethod
    def get_result(self):
        raise NotImplementedError




class MetricsR2(ABC):
    @abstractmethod
    def get_r2_score(self):
        raise NotImplementedError


class MetricsRMSE(ABC):
    @abstractmethod
    def get_RMSE(self):
        raise NotImplementedError
