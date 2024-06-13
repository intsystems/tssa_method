""" complementary module with data extraction methods from raw data files
"""
import pandas as pd
import numpy as np


def GetWalkData():
    """return train/test data and corresponding time grids
    """
    raw_data = pd.read_csv("../../../data/motion2/11.Walk/1004_L_2.csv").values

    time_grid = raw_data[:, 0]
    # remove time grids
    data = raw_data[:, [1, 2, 3, 5, 6, 7]]

    # split on train/test
    train_bound = int((1 - 0.18) * data.shape[0])

    return (data[:train_bound], time_grid[:train_bound]), \
            (data[train_bound:], time_grid[train_bound:])


