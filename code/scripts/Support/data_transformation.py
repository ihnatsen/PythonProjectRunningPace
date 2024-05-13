from datetime import datetime, timezone, timedelta
import json


def unix_to_datetime(unix_time):
    dt_utc = datetime.fromtimestamp(unix_time, tz=timezone.utc)
    warsaw_tz = dt_utc.astimezone(timezone(timedelta(hours=2)))
    return warsaw_tz


def json_string_to_dict(string: str) -> dict:
    return json.loads(string)
