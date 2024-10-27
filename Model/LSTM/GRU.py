# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/10/27 17:11
from typing import Tuple

import torch
from torch import nn, Tensor
import math

from transformers import set_seed


class GRUCell(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int
    ):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ir = nn.Parameter(Tensor(input_size, hidden_size))
        self.weight_hr = nn.Parameter(Tensor(hidden_size, hidden_size))
        self.weight_iz = nn.Parameter(Tensor(input_size, hidden_size))
        self.weight_hz = nn.Parameter(Tensor(hidden_size, hidden_size))
        self.weight_in = nn.Parameter(Tensor(input_size, hidden_size))
        self.weight_hn = nn.Parameter(Tensor(hidden_size, hidden_size))
        self.bias_ir = nn.Parameter(Tensor(hidden_size))
        self.bias_hr = nn.Parameter(Tensor(hidden_size))
        self.bias_iz = nn.Parameter(Tensor(hidden_size))
        self.bias_hz = nn.Parameter(Tensor(hidden_size))
        self.bias_in = nn.Parameter(Tensor(hidden_size))
        self.bias_hn = nn.Parameter(Tensor(hidden_size))
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(
            self,
            x: Tensor,
            h_t_minus_1: Tensor
    ) -> Tensor:
        r = torch.sigmoid(torch.mm(x, self.weight_ir) + self.bias_ir + torch.mm(h_t_minus_1, self.weight_hr) + self.bias_hr)
        z = torch.sigmoid(torch.mm(x, self.weight_iz) + self.bias_iz + torch.mm(h_t_minus_1, self.weight_hz) + self.bias_hz)
        n = torch.tanh(torch.mm(x, self.weight_in) + self.bias_in + r * (torch.mm(h_t_minus_1, self.weight_hn) + self.bias_hn))
        h = (1 - z) * n + z * h_t_minus_1

        return h


class GRU(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            batch_first: bool = False,
    ):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        layers = [GRUCell(input_size, hidden_size)]
        for _ in range(self.num_layers - 1):
            layers += [GRUCell(hidden_size, hidden_size)]
        self.net = nn.Sequential(*layers)

        self.h = None

    def forward(
            self,
            x: Tensor,
            init_states: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.batch_first:
            x = x.transpose(0, 1)

        self.h = torch.zeros(x.size(0), self.num_layers, x.size(1), self.hidden_size).to(x.device)
        if init_states is not None:
            self.h[0] = init_states

        inputs = x
        for i, cell in enumerate(self.net):  # Layers
            h_t = self.h[0, i].clone()
            for t in range(x.size(0)):  # Sequences
                h_t = cell(inputs[t], h_t)
                self.h[t, i] = h_t
            inputs = self.h[:, i].clone()

        if self.batch_first:
            return self.h[:, -1].transpose(0, 1), self.h[-1]

        return self.h[:, -1], self.h[-1]


class GRUModel(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int
    ):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.gru(x, None)
        out = self.fc(out)

        return out


def main():
    set_seed(2024)

    batch_size = 5
    seq_length = 3
    input_size = 10
    inputs = torch.randn(batch_size, seq_length, input_size)

    hidden_size = 5
    num_layers = 2
    output_size = 5
    model = GRUModel(input_size, hidden_size, num_layers, output_size)

    outputs = model(inputs)
    print(outputs.size())


if __name__ == '__main__':
    main()
