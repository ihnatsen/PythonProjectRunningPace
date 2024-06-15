import math

import pandas as pd
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scripts.Support.format_txt import paint as p


class Algorithm(ABC):

    def __init__(self, target, factors, df):
        self.target = target
        self.factors = factors
        self.df = df
        self.test = self.df.sample(frac=0.2)
        self.train = self.df.drop(self.test.index)
        self.model = self.create_model()

    def get_result(self):
        return self.model.predict(self.test[self.factors])

    def get_r2_score(self):
        return f'{r2_score(self.get_result(), self.test[self.target]):0.3f}'

    def get_RMSE(self):
        return f'{math.sqrt(mean_squared_error(self.get_result(), self.test[self.target])):0.3}'

    def print_information(self):
        print(f'{p('RMSE', 'green')}:')
        print(f'{self.get_RMSE()}')
        print(f'{p('R2', 'green')}:')
        print(f'{self.get_r2_score()}')

    @abstractmethod
    def create_model(self):
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
