from datetime import timedelta

import pandas as pd
from scripts.Support.path import *

training = pd.read_csv(get_path_df([''], 'training.csv'))
weather = pd.read_csv(get_path_df([''], 'weather.csv'))


def calculate_weight(start_time: timedelta, duration: int) -> tuple[int, int]:
    roof = timedelta(hours=start_time.seconds//3600 + 1)
    delta = duration - (roof.seconds - start_time.seconds)
    return (1, 0) if delta < 0 else (1 - (delta/duration), delta/duration)


def get_weights():

    # start_time contain value type of "09:05:08".
    duration = list(training['duration'])

    start_time = [tuple(map(int, record.split(':'))) for record in list(training['start_time'])]
    start_time = [timedelta(hours=record[0], minutes=record[1], seconds=record[2]) for record in start_time]
    return [calculate_weight(st, dt) for st, dt in zip(start_time, duration)]


def join_weather_and_training():
    date, hour = slice(0, 10), slice(11, 13)
    key_weather = [(record[date], record[hour]) for record in list(weather['datetime'])]

    hour = slice(0, 2)
    key_training = [(date, time[hour]) for date, time in zip(training['data_training'], training['start_time'])]

    indexes = ([key_weather.index(key) for key in key_training])

    temp = [weight[0]*weather['temp'].iloc[index] + weight[1]*weather['temp'].iloc[index+1]
            for index, weight in zip(indexes, get_weights())]

    pressure = [weight[0]*weather['sealevelpressure'].iloc[index] + weight[1]*weather['sealevelpressure'].iloc[index+1]
            for index, weight in zip(indexes, get_weights())]

    training['temp'] = [round(t, 2) for t in temp]

    training['pressure'] = [round(p, 2) for p in pressure]

    return training


def main():
    training = join_weather_and_training()
    training.to_csv(get_path_to_new_file('dataset', 'bob.csv'), index=False)

if __name__ == '__main__':
    main()
