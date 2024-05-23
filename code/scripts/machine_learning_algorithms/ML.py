from abc import ABC, abstractmethod


class Algorithm(ABC):
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
