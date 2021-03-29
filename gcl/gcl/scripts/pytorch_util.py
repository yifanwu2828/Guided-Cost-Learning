from typing import Union

import numpy as np
import torch
from torch import nn

Activation = Union[str, nn.Module]


_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'leaky_relu': nn.LeakyReLU(),
    'sigmoid': nn.Sigmoid(),
    'selu': nn.SELU(),
    'softplus': nn.Softplus(),
    'identity': nn.Identity(),
}


class PMILayer(nn.Module):
    """ Custom Identity and negative identity layer """
    def __init__(self, size_in):
        super().__init__()
        self.size_in, self.size_out = size_in, size_in*2
        eye = torch.eye(size_in)
        weights = torch.vstack((eye, -eye))
        self.weights = nn.Parameter(weights, requires_grad=True)  # nn.Parameter is a Tensor that's a module parameter.


    def forward(self, x):
        w_times_x= torch.matmul(x, self.weights.t())
        return w_times_x


def build_mlp(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'tanh',
        output_activation: Activation = 'identity',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network

            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)


def build_mlp_yt(
        input_size: int,
        output_size: int,
        n_layers: int,
        size: int,
        activation: Activation = 'identity',
        output_activation: Activation = 'relu',
) -> nn.Module:
    """
        Builds a feedforward neural network

        arguments:
            input_placeholder: placeholder variable for the state (batch_size, input_size)
            scope: variable scope of the network

            n_layers: number of hidden layers
            size: dimension of each hidden layer
            activation: activation of each hidden layer

            input_size: size of the input layer
            output_size: size of the output layer
            output_activation: activation of the output layer

        returns:
            output_placeholder: the result of a forward pass through the hidden layers + the output layer
    """
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]
    layers = []
    in_size = input_size

    pmi_layer = PMILayer(in_size)
    layers.append(pmi_layer)
    layers.append(activation)
    in_size = in_size*2

    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)


device = None


def init_gpu(use_gpu=True, gpu_id=0) -> None:
    """ init device('cuda:0' or 'cpu') """
    global device
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:" + str(gpu_id))
        print("Using GPU id {}".format(gpu_id))
    else:
        device = torch.device("cpu")
        print("GPU not detected. Defaulting to CPU.")


def set_device(device: Union[torch.device, int]) -> None:
    torch.cuda.set_device(device)


def from_numpy(*args, **kwargs):
    """ Convert numpy array to torch tensor  and send to device('cuda:0' or 'cpu') """
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor: torch.FloatTensor) -> np.ndarray:
    """ Convert torch tensor to numpy array and send to CPU """
    return tensor.to('cpu').detach().numpy()


def exp_normalize(x):
    b = x.max()
    y = torch.exp(x - b)
    return y / y.sum()
