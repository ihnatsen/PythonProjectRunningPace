from __future__ import annotations
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


class Algorithm(ABC):

    @abstractmethod
    def __init__(self, target, factors, df):
        self.target = target
        self.factors = factors
        self.df = df
        self.test = self.get_df().sample(frac=0.2, random_state=1)
        self.train = self.get_df().drop(self.test.index)

    @abstractmethod
    def get_result(self):
        raise NotImplementedError

    @abstractmethod
    def get_RMSE(self):
        raise NotImplementedError

    @abstractmethod
    def get_r2_score(self):
        raise NotImplementedError

    @abstractmethod
    def get_df(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def print_information(self):
        raise NotImplementedError


class MMScaler:
    @staticmethod
    def set_scaler_df(df: pd.DataFrame) -> pd.DataFrame:

        df_for_normal, other = (df.select_dtypes(include=['int', 'float']),
                                df.select_dtypes(include=['object']))
        mms = MinMaxScaler()
        mms.fit(df_for_normal)

        normal_data = mms.transform(df_for_normal)
        normal_data = pd.DataFrame(normal_data, columns=df_for_normal.columns)

        return pd.concat([normal_data, other], axis=1)


class SScaler:
    @staticmethod
    def set_scaler_df(df: pd.DataFrame) -> pd.DataFrame:
        df_for_normal, other = (df.select_dtypes(include=['int', 'float']),
                                df.select_dtypes(include=['object']))
        ss = StandardScaler()
        ss.fit(df_for_normal)

        normal_data = ss.transform(df_for_normal)
        normal_data = pd.DataFrame(normal_data, columns=df_for_normal.columns)

        return pd.concat([normal_data, other], axis=1)
