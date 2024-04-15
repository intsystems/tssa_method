import numpy as np
import pandas as pd


def GetBerlinData(test_ratio: float):
    """returns datetime grids train/test data on weather conditions in Berlin 1980-1990
    """
    weather_dataframe = pd.read_parquet('../../../data/weather/daily_weather.parquet')

    city_data = weather_dataframe[weather_dataframe['city_name'] == 'Berlin']
    city_data = city_data[
        (city_data['date'] > pd.to_datetime('1980')) & (city_data['date'] < pd.to_datetime('1990'))
        ]
    city_data = city_data[['date', 'avg_temp_c', 'precipitation_mm', 'avg_sea_level_pres_hpa']]

    # split on train and test
    train_data = city_data.iloc[:int(city_data.shape[0] * (1 - test_ratio))].drop(columns='date').values
    test_data = city_data.iloc[int(city_data.shape[0] * (1 - test_ratio)):].drop(columns='date').values

    # get time grids for train and test signals
    time_grid_train = city_data['date'][:int(city_data.shape[0] * (1 - test_ratio))].values
    time_grid_test = city_data['date'][int(city_data.shape[0] * (1 - test_ratio)):].values

    return (train_data, time_grid_train), (test_data, time_grid_test)