# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/6/28 20:19
import torch
from torch import nn
from torch import Tensor
import math
from typing import Tuple
import torch.nn.functional as F
from transformers import set_seed


class LSTMCell(nn.Module):
    """
    Long Short-Term Memory (HeartLSTM) cell.
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        """

        :param input_size:
        :param hidden_size:
        """
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(Tensor(input_size, hidden_size * 4))
        self.weight_hh = nn.Parameter(Tensor(hidden_size, hidden_size * 4))
        self.bias_ih = nn.Parameter(Tensor(hidden_size * 4))
        self.bias_hh = nn.Parameter(Tensor(hidden_size * 4))
        self.init_parameters()

    def init_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x: Tensor, init_states: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """

        :param x:
        :param init_states:
        :return:
        """
        h_t_minus_1, c_t_minus_1 = init_states
        gates = torch.mm(x, self.weight_ih) + self.bias_ih + torch.mm(h_t_minus_1, self.weight_hh) + self.bias_hh
        input_gate, forget_gate, cell, output_gate = gates.chunk(4, dim=1)
        c = torch.sigmoid(forget_gate) * c_t_minus_1 + torch.sigmoid(input_gate) * torch.tanh(cell)
        h = torch.sigmoid(output_gate) * torch.tanh(c)

        return h, c


class LSTM(nn.Module):
    """
    Heart Long Short-Term Memory (HeartLSTM).
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 batch_first: bool = False) -> None:
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        layers = [LSTMCell(input_size, hidden_size)]
        for i in range(self.num_layers - 1):
            layers += [LSTMCell(hidden_size, hidden_size)]
        self.net = nn.Sequential(*layers)

        self.h = None
        self.c = None

    def forward(self, x: Tensor, init_states: Tuple[Tensor, Tensor] = None):
        # Input and output size: (seq_length, batch_size, input_size)
        # States size: (num_layers, batch_size, hidden_size)
        if self.batch_first:
            x = x.transpose(0, 1)

        self.h = torch.zeros(x.size(0), self.num_layers, x.size(1), self.hidden_size).to(x.device)
        self.c = torch.zeros(x.size(0), self.num_layers, x.size(1), self.hidden_size).to(x.device)
        if init_states is not None:
            self.h[0], self.c[0] = init_states

        inputs = x
        for i, cell in enumerate(self.net):  # Layers
            h_t, c_t = self.h[0, i].clone(), self.c[0, i].clone()
            for t in range(x.size(0)):  # Sequences
                h_t, c_t = cell(inputs[t], (h_t, c_t))
                self.h[t, i], self.c[t, i] = h_t, c_t
            inputs = self.h[:, i].clone()

        if self.batch_first:
            return self.h[:, -1].transpose(0, 1), (self.h[-1], self.c[-1])

        return self.h[:, -1], (self.h[-1], self.c[-1])


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.heartlstm = LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: Tensor):
        out, _ = self.heartlstm(x, None)
        # out = self.fc(out[:, -1, :])
        out = self.fc(out)
        return out


def main():
    set_seed(612)
    _inputs = torch.randn(10, 3, 10)
    # _hearts = torch.Tensor([[[1, 2, 3]], [[1, 2, 3]]])
    # _inputs = torch.Tensor([[[1, 2, 3], [2, 1, 3], ], [[1, 2, 3], [2, 1, 3], ]])
    # _inputs = torch.Tensor([[[2, 1, 3], [1, 2, 3]], [[2, 1, 3], [1, 2, 3]]])
    print(_inputs.size())

    _model = LSTMModel(10, 5, 1, 5)
    # _model = HeartLSTMModel(3, 3, 5, 1, 5)

    _outputs = _model(_inputs)
    print(_outputs.size())
    _outputs = F.adaptive_max_pool2d(_outputs, (1, 5))
    print(_outputs)


if __name__ == '__main__':
    main()
