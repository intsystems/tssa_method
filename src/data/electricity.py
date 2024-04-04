import numpy as np
import pandas as pd


def GetSeriesFullLen():
    """
    Returns:
        int: length of full dataset
    """
    return pd.read_csv("../../data/turkish_electricity_consumption/TurkElectricityConsumption.csv", 
                             names=["date"]).shape[0]

def GetTrainTestData(series_len: int, test_ratio: float):
    """
    Args:
        series_len (int): the length of desired series, will be taken from the begining of the dataset
        test_ratio (float): 

    Returns:
        tuple: (train_data, time_grid_train), (test_data, time_grid_test)
    """
    data_frame = pd.read_csv("../../../data/turkish_electricity_consumption/TurkElectricityConsumption.csv", 
                         names=["date", 'time', 'DDPP', 'proj_rem_load', 'get_load', 'load_shed', 'price'])
    
    electr_data = data_frame[['DDPP', 'price']].values[::-1]
    # shrink data
    electr_data = electr_data[:series_len]

    # split on train and test
    train_data = electr_data[:int(electr_data.shape[0] * (1 - test_ratio))]
    test_data = electr_data[int(electr_data.shape[0] * (1 - test_ratio)):]

    # get time grids for train and test signals
    time_grid_train = np.arange(electr_data.shape[0])[:int(electr_data.shape[0] * (1 - test_ratio))]
    time_grid_test = np.arange(electr_data.shape[0])[int(electr_data.shape[0] * (1 - test_ratio)):]

    # eliminate one outlier element
    el_outlier = np.min(test_data.T[0])
    el_outlier_num = np.argmin(test_data.T[0])

    test_data = np.delete(test_data, el_outlier_num, axis=0)
    time_grid_test = np.delete(time_grid_test, el_outlier_num, axis=0)

    return (train_data, time_grid_train), (test_data, time_grid_test)