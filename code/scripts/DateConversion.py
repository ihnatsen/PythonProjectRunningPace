import pandas as pd
from scripts.Support.path import *


def get_df_weather() -> pd.DataFrame:
    # name, datetime, temp, feelslike, dew, humidity, precip, precipprob,
    # preciptype, snow, snowdepth, windgust, windspeed, winddir, sealevelpressure,
    # cloudcover, visibility, solarradiation, solarenergy, uvindex, severerisk,
    # conditions, icon, stations

    weather = pd.read_csv(get_path_df('weather', 'Warsaw2023-04-15to2024-05-08.csv'))

    # Only the 9336th record is valid and break algorithm
    records_str = list(weather['name'].head(9335)) + list(weather['name'].tail(24))

    values = [record.split(',')[:-5] for record in records_str]
    values = list(zip(*values))
    labels = list(weather.keys())[:-5]
    df_weather = pd.DataFrame(data={label: value for label, value in zip(labels, values)})

    labels = ['datetime', 'temp', 'feelslike',
              'humidity', 'precip', 'preciptype',
              'snow', 'windgust', 'windspeed', 'sealevelpressure',
              'cloudcover']

    df_weather = df_weather[labels]
    return df_weather


def filter_one(df: pd.DataFrame) -> pd.DataFrame:
    def foo(label, error, valid):
        indexes_error = df[df[label] == error].index
        for index in indexes_error:
            df.at[index, label] = valid
        return df

    df = foo('preciptype', '\"rain', 'rain')
    df = foo('snow', 'snow\"', 'snow')

    return df


def main():
    weather = get_df_weather()
    weather = filter_one(weather)
    weather.to_csv(get_path_to_new_file('dataset', 'weather.csv'), index=False)


if __name__ == '__main__':
    main()
