import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_normal_data(df):
    mms = MinMaxScaler()
    types = ('float64', 'int64')


def main():
    df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['34', '4', '2', 'o']})

    df_for_normal, other = (df.select_dtypes(include=['int', 'float']),
                            df.select_dtypes(include=['object']))
    mms = MinMaxScaler()
    mms.fit(df_for_normal)

    normal_data = mms.transform(df_for_normal)
    normal_data = pd.DataFrame(normal_data, columns=df_for_normal.columns)
    return pd.concat([normal_data, other], axis=1)


if __name__ == '__main__':
    main()
