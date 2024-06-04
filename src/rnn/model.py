import torch
import torch.nn as nn


class RnnSignalPredictor(nn.Module):
    def __init__(
            self, 
            input_dim: int, 
            hidden_dim:int,
            num_layers: int = 1
        ):
        super().__init__()

        # transformation from [h_t, x_t] to our prediction x_{t+1}
        # simple FF network
        self._hid_to_target = nn.Linear(hidden_dim, input_dim)

        self._h_0 = torch.nn.Parameter(torch.ones((num_layers, 1, hidden_dim)), requires_grad=False)
        #self._c_0 = 5 * torch.ones((num_layers, 1, hidden_dim))
        
        self._rnn_unit = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    
    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]

        # obtain outputs of rnn last layer
        output, h_n = self._rnn_unit(
            x, 
            self._h_0.repeat(1, batch_size, 1)
        )
        # transform them to input space. These are our perdictions for the next value in time series
        output = self._hid_to_target(output)
        
        return output
    
    
    def predict(
            self, 
            horizon: int,
            context: torch.Tensor
        ):
        """predict next series values given the initial context

        Args:
            horizon (int): num steps to predict after the context
            context (torch.Tensor): initial series values (if given), shape=(N, L, H_in). 
                                    Should be at least one value

        Returns:
            _type_: predictions. If there is no context, prediction will not have batch dimension
        """
        batch_size = context.shape[0]
        input_size = context.shape[2]

        # container for predicted values
        forecast = torch.empty((batch_size, horizon, input_size))

        with torch.no_grad():
            # obtain last hidden vectors from rnn
            output_after_context, h_last = self._rnn_unit(context, self._h_0.repeat(1, batch_size, 1))
            # obtain last rnn output
            last_prediction = self._hid_to_target(output_after_context[:, -1, :])

            for i in range(horizon):
                # safe last prediction
                forecast[:, i, :] = last_prediction

                # add 'length' dimension
                last_prediction = last_prediction.reshape((batch_size, 1, input_size))

                # obtain next model output from last prediction
                new_output, h_last = self._rnn_unit(last_prediction, h_last)
                last_prediction = self._hid_to_target(new_output[:, -1, :])

        return forecast

        
