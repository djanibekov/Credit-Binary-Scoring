import os
from datetime import date
from datetime import datetime

import dotenv


def calculate_age_by_days(born_date):
    dotenv_file = dotenv.find_dotenv()
    dotenv.load_dotenv(dotenv_file)

    born_date = datetime.strptime(born_date, "%Y-%m-%d").date()
    today = datetime.today()
    birth_date = date(born_date.year, born_date.month, born_date.day)
    today_date = date(today.year, today.month, today.day)

    time_difference = today_date - birth_date
    print(os.environ["MEAN"])
    standardized_days = (time_difference.days - float(os.environ["MEAN"])) / float(os.environ["STD"])
    print(standardized_days)
    return [standardized_days]


def marital_status_arr(mrt_status):
    arr = [0] * 4
    arr[mrt_status] = 1
    return arr


def gender_arr(gender):
    arr = [0] * 2
    arr[gender] = 1
    return arr


