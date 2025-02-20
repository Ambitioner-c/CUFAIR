# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/10/20 11:30
from typing import Tuple

import torch
from torch import nn, Tensor
import math

from transformers import set_seed


class SQALSTMCell(nn.Module):
    def __init__(
            self,
            question_size: int,
            answer_size: int,
            input_size: int,
            hidden_size: int,
            is_peephole: bool = False,
    ):
        super(SQALSTMCell, self).__init__()
        self.question_size = question_size
        self.answer_size = answer_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_qh = nn.Parameter(Tensor(question_size, hidden_size * 4))
        self.weight_ah = nn.Parameter(Tensor(answer_size, hidden_size * 4))
        self.weight_ih = nn.Parameter(Tensor(input_size, hidden_size * 4))
        self.weight_hh = nn.Parameter(Tensor(hidden_size, hidden_size * 4))
        self.bias_qh = nn.Parameter(Tensor(hidden_size * 4))
        self.bias_ah = nn.Parameter(Tensor(hidden_size * 4))
        self.bias_ih = nn.Parameter(Tensor(hidden_size * 4))
        self.bias_hh = nn.Parameter(Tensor(hidden_size * 4))
        self.init_parameters()

        self.is_peephole = is_peephole

    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(
            self,
            question: Tensor,
            answer: Tensor,
            x: Tensor,
            init_states: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        h_t_minus_1, c_t_minus_1 = init_states

        q_temp = torch.mm(question, self.weight_qh) + self.bias_qh
        a_temp = torch.mm(answer, self.weight_ah) + self.bias_ah
        if self.is_peephole:
            q_temp[:, self.hidden_size * 2: self.hidden_size * 3] = 0
            a_temp[:, self.hidden_size * 2: self.hidden_size * 3] = 0
        gates = q_temp + a_temp + torch.mm(x, self.weight_ih) + self.bias_ih + torch.mm(h_t_minus_1, self.weight_hh) + self.bias_hh
        input_gate, forget_gate, cell, output_gate = gates.chunk(4, dim=1)
        c = torch.sigmoid(forget_gate) * c_t_minus_1 + torch.sigmoid(input_gate) * torch.tanh(cell)
        h = torch.sigmoid(output_gate) * torch.tanh(c)

        return h, c


class SQALSTM(nn.Module):
    def __init__(
            self,
            question_size: int,
            answer_size: int,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            batch_first: bool = False,
            is_peephole: bool = False,
    ):
        super(SQALSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        layers = [SQALSTMCell(question_size, answer_size, input_size, hidden_size, is_peephole)]
        for _ in range(self.num_layers - 1):
            layers += [SQALSTMCell(question_size, answer_size, hidden_size, hidden_size, is_peephole)]
        self.net = nn.Sequential(*layers)

        self.h = None
        self.c = None

    def forward(
            self,
            question: Tensor,
            answer: Tensor,
            x: Tensor,
            init_states: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # Input and output size: (batch_size, seq_length, input_size)
        # States size: (num_layers, batch_size, hidden_size)
        if self.batch_first:
            question = question.transpose(0, 1)
            answer = answer.transpose(0, 1)
            x = x.transpose(0, 1)

        self.h = torch.zeros(x.size(0), self.num_layers, x.size(1), self.hidden_size).to(x.device)
        self.c = torch.zeros(x.size(0), self.num_layers, x.size(1), self.hidden_size).to(x.device)
        if init_states is not None:
            self.h[0], self.c[0] = init_states

        inputs = x
        for i, cell in enumerate(self.net):  # Layers
            h_t, c_t = self.h[0, i].clone(), self.c[0, i].clone()
            for t in range(x.size(0)):  # Sequences
                h_t, c_t = cell(question[0], answer[0], inputs[t], (h_t, c_t))
                self.h[t, i], self.c[t, i] = h_t, c_t
            inputs = self.h[:, i].clone()

        if self.batch_first:
            return self.h[:, -1].transpose(0, 1), (self.h[-1], self.c[-1])

        return self.h[:, -1], (self.h[-1], self.c[-1])


class SQALSTMModel(nn.Module):
    def __init__(
            self,
            question_size: int,
            answer_size: int,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            output_size: int,
            is_peephole: bool,
    ):
        super(SQALSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sqalstm = SQALSTM(question_size, answer_size, input_size, hidden_size, num_layers, batch_first=True, is_peephole=is_peephole)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, question: Tensor, answer: Tensor, x: Tensor) -> Tensor:
        out, _ = self.sqalstm(question, answer, x, None)
        out = self.fc(out)

        return out


def main():
    set_seed(2024)

    batch_size = 5
    seq_length = 3
    input_size = 12
    questions = torch.randn(batch_size, 1, input_size)
    answers = torch.randn(batch_size, 1, input_size)
    inputs = torch.randn(batch_size, seq_length, input_size)

    hidden_size = 12
    num_layers = 2
    output_size = 5
    is_peephole = False
    model = SQALSTMModel(input_size, input_size, input_size, hidden_size, num_layers, output_size, is_peephole)
    outputs = model(questions, answers, inputs)
    print(outputs.size())


if __name__ == '__main__':
    main()
