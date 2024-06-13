import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


def GetTrainDataloader(
        train_time_series: np.ndarray,
        len_in_batch: int,
        num_samples: int,
        batch_size: int
) -> DataLoader:
    """build slices of given time series and form a dataset from it. Slices may overspan

    Args:
        train_time_series (np.ndarray): ]
        len_in_batch (int): length of each slice
        num_samples (int): size of the dataset
        batch_size (int): for torch.Dataloader

    Returns:
        torch DataLoader
    """
    # sample starting points in train time series to obtain a slice
    possible_starting_positions = np.arange(0, train_time_series.shape[0] - len_in_batch)
    if possible_starting_positions.size >= num_samples:
        starting_points = np.random.choice(possible_starting_positions, num_samples, replace=False)
    else:
        starting_points = np.random.choice(possible_starting_positions, num_samples, replace=True)

    # build samples
    dataset = np.empty((num_samples, len_in_batch, train_time_series.shape[1]))
    for i in range(num_samples):
        dataset[i] = train_time_series[starting_points[i]:starting_points[i] + len_in_batch].copy()

    dataset = TensorDataset(torch.from_numpy(dataset.astype(np.float32)))

    return DataLoader(dataset, batch_size, shuffle=False)
