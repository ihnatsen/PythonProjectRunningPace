import pandas as pd
from Support.path import get_path_df
from Support.data_transformation import unix_to_datetime, json_string_to_dict, Value
from Support.format_txt import paint

df = pd.read_csv(get_path_df('mi_fitness', 'all_data.csv'))


def search_first_last_date():
    first_date, last_date = list(df['Time'])[0], list(df['Time'])[-1]
    print(f'The first date is {paint(unix_to_datetime(first_date), 'red')}')
    print(f'The last date is {paint(unix_to_datetime(first_date), 'blue')}')


def main():
    search_first_last_date()

    ...


if __name__ == '__main__':
    main()
