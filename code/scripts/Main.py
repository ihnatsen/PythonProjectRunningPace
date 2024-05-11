import pandas as pd
from Support.path import get_path_df
from Support.data_transformation import unix_to_datetime


def main():
    df = pd.read_csv(get_path_df('mi_fitness', 'all_data.csv'))
    time = [unix_to_datetime(unix) for unix in tuple(df['Time'])]
    print(f'The first date is {min(time)}.')
    print(f'The last date is {max(time)}.')


if __name__ == '__main__':
    main()
