# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/10/27 17:21
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


class BiGRU(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            batch_first: bool = False,
    ):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.forward_net = nn.Sequential(*[GRUCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.backward_net = nn.Sequential(*[GRUCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

        self.h_forward = None
        self.h_backward = None

        self.h = None

    def forward(
            self,
            x: Tensor,
            init_states: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if self.batch_first:
            x = x.transpose(0, 1)

        self.h_forward = torch.zeros(x.size(0), self.num_layers, x.size(1), self.hidden_size).to(x.device)
        self.h_backward = torch.zeros(x.size(0), self.num_layers, x.size(1), self.hidden_size).to(x.device)
        if init_states is not None:
            self.h[0] = init_states

        inputs = x
        for i, (forward_cell, backward_cell) in enumerate(zip(self.forward_net, self.backward_net)):  # Layers
            # Forward
            h_t = self.h_forward[0, i].clone()
            for t in range(x.size(0)):  # Sequences
                h_t = forward_cell(inputs[t], h_t)
                self.h_forward[t, i] = h_t
            # Backward
            h_t = self.h_backward[0, i].clone()
            for t in range(x.size(0)):  # Sequences
                h_t = backward_cell(inputs[t], h_t)
                self.h_backward[t, i] = h_t
            inputs = torch.cat((self.h_forward[:, i], self.h_backward[:, i]), dim=-1).clone()

        self.h = torch.cat((self.h_forward, self.h_backward), dim=-1)

        if self.batch_first:
            return self.h[:, -1].transpose(0, 1), self.h[-1]

        return self.h[:, -1], self.h[-1]


class BiGRUModel(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int
    ):
        super(BiGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bigru = BiGRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.bigru(x, None)
        out = self.fc(out)

        return out


def main():
    set_seed(2024)

    batch_size = 5
    seq_length = 3
    input_size = 10
    inputs = torch.randn(batch_size, seq_length, input_size)

    hidden_size = 10
    num_layers = 1
    output_size = 10
    model = BiGRUModel(input_size, hidden_size, num_layers, output_size)

    outputs = model(inputs)
    print(outputs.size())


if __name__ == '__main__':
    main()
