from datetime import datetime, timezone, timedelta
import json


class Value:
    def __init__(self, value):
        self.__value = value

    def __getitem__(self, key) -> list[int] | list[float]:
        return [record[key] for record in self.__value]


def unix_to_datetime(unix_time):
    dt_utc = datetime.fromtimestamp(unix_time, tz=timezone.utc)
    warsaw_tz = dt_utc.astimezone(timezone(timedelta(hours=2)))
    return warsaw_tz


def json_string_to_dict(string):
    return json.loads(string)
