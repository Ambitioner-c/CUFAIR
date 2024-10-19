# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/10/19 10:09
from typing import Tuple

import torch
from torch import nn, Tensor
import math

from transformers import set_seed


class LSTMCell(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int
    ):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(Tensor(input_size, hidden_size * 4))
        self.weight_hh = nn.Parameter(Tensor(hidden_size, hidden_size * 4))
        self.bias_ih = nn.Parameter(Tensor(hidden_size * 4))
        self.bias_hh = nn.Parameter(Tensor(hidden_size * 4))
        self.init_parameters()

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(
            self,
            x: Tensor,
            init_states: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        h_t_minus_1, c_t_minus_1 = init_states
        gates = torch.mm(x, self.weight_ih) + self.bias_ih + torch.mm(h_t_minus_1, self.weight_hh) + self.bias_hh
        input_gate, forget_gate, cell, output_gate = gates.chunk(4, dim=1)
        c = torch.sigmoid(forget_gate) * c_t_minus_1 + torch.sigmoid(input_gate) * torch.tanh(cell)
        h = torch.sigmoid(output_gate) * torch.tanh(c)

        return h, c


class BiLSTM(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            batch_first: bool = False,
    ):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.forward_net = nn.Sequential(*[LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])
        self.backward_net = nn.Sequential(*[LSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

        self.h_forward = None
        self.c_forward = None
        self.h_backward = None
        self.c_backward = None

        self.h = None
        self.c = None

    def forward(
            self,
            x: Tensor,
            init_states: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # Input and output size : (batch_size, seq_length, input_size)
        # States size: (num_layers, batch_size, hidden_size)
        if self.batch_first:
            x = x.transpose(0, 1)

        self.h_forward = torch.zeros(x.size(0), self.num_layers, x.size(1), self.hidden_size).to(x.device)
        self.c_forward = torch.zeros(x.size(0), self.num_layers, x.size(1), self.hidden_size).to(x.device)
        self.h_backward = torch.zeros(x.size(0), self.num_layers, x.size(1), self.hidden_size).to(x.device)
        self.c_backward = torch.zeros(x.size(0), self.num_layers, x.size(1), self.hidden_size).to(x.device)
        if init_states is not None:
            self.h[0], self.c[0] = init_states

        inputs = x
        for i, (forward_cell, backward_cell) in enumerate(zip(self.forward_net, self.backward_net)):  # Layers
            # Forward
            h_t, c_t = self.h_forward[0, i].clone(), self.c_forward[0, i].clone()
            for t in range(x.size(0)):  # Sequences
                h_t, c_t = forward_cell(inputs[t], (h_t, c_t))
                self.h_forward[t, i], self.c_forward[t, i] = h_t, c_t
            # Backward
            h_t, c_t = self.h_backward[0, i].clone(), self.c_backward[0, i].clone()
            for t in range(x.size(0)):  # Sequences
                h_t, c_t = backward_cell(inputs[t], (h_t, c_t))
                self.h_backward[t, i], self.c_backward[t, i] = h_t, c_t
            inputs = torch.cat((self.h_forward[:, i], self.h_backward[:, i]), dim=-1).clone()

        self.h = torch.cat((self.h_forward, self.h_backward), dim=-1)
        self.c = torch.cat((self.c_forward, self.c_backward), dim=-1)

        if self.batch_first:
            return self.h[:, -1].transpose(0, 1), (self.h[-1], self.c[-1])

        return self.h[:, -1], (self.h[-1], self.c[-1])


class BiLSTMModel(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int
    ):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bilstm = BiLSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: Tensor) -> Tensor:
        out, _ = self.bilstm(x, None)
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
    model = BiLSTMModel(input_size, hidden_size, num_layers, output_size)

    outputs = model(inputs)
    print(outputs.size())


if __name__ == '__main__':
    main()
