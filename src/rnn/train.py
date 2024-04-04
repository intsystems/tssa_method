import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

import pickle
import pathlib

from .model import RnnSignalPredictor

def BackupModel(
        model: nn.Module,
        path: str
):
    # assume backup folder does exist
    with open(path, 'wb') as f:
        pickle.dump(model, f)

    # with open(f'./backup/experiment_{experiment_num}/rnn_predictor.pkl', 'wb') as f:
    #     pickle.dump(model, f)


def RestoreModel(
        path: str
):
    if pathlib.Path(path).exists():
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError('No model for given experiment')

    # if pathlib.Path(f'./backup/experiment_{experiment_num}/rnn_predictor.pkl').exists():
    #     with open(f'./backup/experiment_{experiment_num}/rnn_predictor.pkl', 'rb') as f:
    #         return pickle.load(f)
    # else:
    #     raise ValueError('No model for given experiment')


def EvaluatePredictor(
        context: torch.Tensor,
        test_data: torch.Tensor,
        model: RnnSignalPredictor,
        device: torch.device
) -> tuple:
    test_len = test_data.shape[0]

    mse_metric = 0
    mape_metric = 0

    model.eval()

    with torch.no_grad():
        # add batch dim to context and put it on device
        context = context.unsqueeze(0).to(device=device)

        # obtain predictions
        model_predictions = model.predict(test_len, context)
        # remove batch dimension and put it on kernal
        model_predictions = model_predictions.squeeze(0).to(device='cpu')

        mse_metric = torch.mean((model_predictions - test_data) ** 2).item()
        mape_metric = torch.mean(torch.abs((model_predictions - test_data) / test_data)).item()

    return (mse_metric, mape_metric)


def TrainEpoch(
        batch_generator: DataLoader,
        model: RnnSignalPredictor, 
        optimizer: torch.optim.Optimizer, # оптимизатор уже включает в себя параметры моделей
        batches_per_epoch: int = None, # по умолчанию проходим по всему датасету
        device: torch.device = torch.device('cpu') # на этом устройстве уже лежат модели и loss-fuction
) -> torch.Tensor:    
    # container for losses in current epoch
    if batches_per_epoch is None:
        loss_array = torch.empty(len(batch_generator))
    else:
        loss_array = torch.empty(batches_per_epoch)

    # loss function - MSE
    loss_func = nn.MSELoss()

    model.train()

    for batch_num, data_batch in enumerate(batch_generator):
        if batch_num == batches_per_epoch:
            break

        # put series slice to device
        data_batch = data_batch[0].to(device=device)

        # obtain model predictions, discard last input (model predicts up to this value)
        predictions = model(data_batch[:, :-1, :])

        # compute MSE on predictions
        cur_loss = loss_func(predictions, data_batch[:, 1:, :])
        loss_array[batch_num] = cur_loss.detach()

        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()

    return loss_array
