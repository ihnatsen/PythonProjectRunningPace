import pandas as pd
from Support.data_transformation import *
from Support.format_txt import paint
from Support.path import *


df = pd.read_csv(get_path_df('mi_fitness', 'all_data.csv'))
date_format = "%m/%d/%Y, %H:%M:%S"


def value_to_dict(df):
    # head[:168] : ['time', 'timezone', 'version', 'start_time', 'end_time', 'proto_type', 'sport_type', 'duration',
    # 'distance', 'calories', 'max_pace', 'min_pace', 'avg_pace', 'max_speed', 'steps', 'max_cadence', 'avg_cadence',
    # 'avg_hrm', 'max_hrm', 'min_hrm', 'rise_height', 'fall_height', 'avg_height', 'max_height', 'min_height']

    # tail [168:]: ['avg_height', 'calories', 'distance', 'duration', 'end_time', 'forefoot_landing_duration',
    # 'golpe_landing_duration', 'heel_landing_duration', 'hrm_aerobic_duration', 'hrm_anaerobic_duration',
    # 'hrm_extreme_duration', 'hrm_fat_burning_duration', 'hrm_warm_up_duration', 'max_cadence', 'max_height',
    # 'max_hrm', 'max_speed', 'min_height', 'min_hrm', 'proto_type', 'sport_type', 'start_time', 'steps', 'time',
    # 'training_experience', 'timezone', 'version']

    # labels: ['calories', 'distance', 'duration', 'start_time', 'end_time', 'max_temp']

    data: list[dict[str:float]]
    labels: list[str]

    labels = ['calories', 'distance', 'duration', 'start_time', 'end_time', 'steps']
    data = [json_string_to_dict(record) for record in df["Value"]]

    values = [{key: record[key] for key in labels} for record in data]
    return values


def replacement(record_one: dict, record_two: dict) -> dict:
    return {'calories': record_one['calories'] + record_two['calories'],
            'distance': record_one['distance'] + record_two['distance'],
            'duration': record_one['duration'] + record_two['duration'],
            'start_time': record_one['start_time'],
            'end_time': record_two['end_time'],
            'steps': record_one['steps'] + record_two['steps']}


def filter_one(df) -> pd.DataFrame:
    value = value_to_dict(df)

    date_time = [time['start_time'] for time in value]

    # It's command search for pair records, which have difference less than hour (3600sec)
    # and return pair indexes.
    data_for_formatting = filter(lambda pair_date: pair_date[1] - pair_date[0] < 3600 and pair_date[1] != pair_date[0],
                                 zip(date_time, date_time[1:]))

    indexes = [date_time.index(first) for first, _ in data_for_formatting]

    for index in indexes:
        value[index+1] = replacement(value[index], value[index+1])
        value[index] = None

    value = [item for item in value if item is not None]

    print(f'The number of records that have a training pause is {paint(len(df) - len(value), 'red')}')
    print(pd.DataFrame(value_to_dict(df)).iloc[[index for index in indexes]])
    print()
    return pd.DataFrame(value)


def filter_two(data: pd.DataFrame) -> pd.DataFrame:
    print(f'The number of records that contain a distance less than {paint(3000, 'red')} meters '
          f'is {paint(len(data[data["distance"] < 3000]), 'red')}')
    print(data[data["distance"] < 3000])
    print()
    return data[data["distance"] > 3000]


def filter_three(data: pd.DataFrame) -> pd.DataFrame:
    filter_data = data[data['duration'] < 3700]
    print(f'The number of records that contain a duration training more than {paint(3700, 'red')} sec '
          f'is {paint(len(data)-len(filter_data), 'red')}')
    print(data[data['duration'] > 3700])
    print()
    return filter_data


def filter_four(data: pd.DataFrame) -> pd.DataFrame:
    print(f'The number of records that contain a avg_pace more than {paint(6, 'red')} min/km '
          f'is {paint(len(data)-len(data[data['avg_pace'] < 6.5]), 'red')}')
    print(data[data['avg_pace'] > 6.5])
    print()
    return data[data['avg_pace'] <= 6.5]


def filter_five(data: pd.DataFrame) -> pd.DataFrame:
    print(f'The number of records that contain steps less than {paint(1400, 'red')} '
          f'is {paint(len(data[data['steps'] < 1400]), 'red')}')
    print(data[data['steps'] < 1400])

    # avg columns which have more 1400 step

    avg = int(data[data['steps'] >= 1400]['steps'].mean())
    data.loc[data['steps'] < 1400, 'steps'] = avg
    return data


def split_datatime(df: pd.DataFrame) -> pd.DataFrame:
    df['start_time'] = pd.to_datetime(df['start_time'], unit='s', utc=True).dt.tz_convert('Europe/Warsaw')
    df['end_time'] = pd.to_datetime(df['end_time'], unit='s', utc=True).dt.tz_convert('Europe/Warsaw')

    df['data_training'] = pd.to_datetime(df['start_time'], unit='s').dt.date
    df['start_time'] = pd.to_datetime(df['start_time'], unit='s').dt.time
    df['end_time'] = pd.to_datetime(df['end_time'], unit='s').dt.time

    return df


def main():
    avg_pace: pd.Series

    # filter data
    data = filter_one(df)
    data = filter_two(data)
    data = filter_three(data)

    # add pd.Series avg_pace to data
    minutes = data['duration']
    kilometres = data['distance']
    avg_pace = minutes/kilometres
    data['avg_pace'] = avg_pace.round(2)

    # filter avg data
    data = filter_four(data)
    # split end and start time to date and hours
    data = split_datatime(data)
    data = filter_five(data)

    # write a ready data to directory dataset
    data.to_csv(get_path_to_new_file('dataset', 'training.csv'), index=False)


if __name__ == '__main__':
    main()
