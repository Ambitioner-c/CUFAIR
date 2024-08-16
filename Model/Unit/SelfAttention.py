# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/6/29 9:52
import math

import torch
import torch.nn as nn
from torch.functional import F


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, attention_size):
        """

        :param hidden_size:
        :param attention_size:
        """
        super(SelfAttention, self).__init__()
        self.Query = nn.Linear(hidden_size, attention_size)
        self.Key = nn.Linear(hidden_size, attention_size)
        self.Value = nn.Linear(hidden_size, attention_size)

        self.HiddenSize = hidden_size
        self.AttentionSize = attention_size

        self.Outputs = None

        self.Activation = nn.Tanh()

    def forward(self, inputs):
        """

        :param inputs:
        :return:
        """
        query = self.Activation(self.Query(inputs))
        key = self.Activation(self.Key(inputs))
        value = self.Activation(self.Value(inputs))

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.AttentionSize)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value)

        self.Outputs = (self.Activation(context_layer), attention_probs)

        return self.Outputs

    def __str__(self):
        info = 'This is Generalized Attention.'
        info += '\n\tHidden Size: ' + str(self.HiddenSize)
        info += '\n\tAttention Size: ' + str(self.AttentionSize)
        info += '\n\tContext Layer: ' + str(self.Outputs[0].size())
        info += '\n\tAttention Probs: ' + str(self.Outputs[1].size())

        return info


def main():
    inputs = torch.randn(10, 5, 8)
    generalized_attention = SelfAttention(8, 4)
    outputs = generalized_attention(inputs)
    # print(outputs[1])
    print(generalized_attention)


if __name__ == '__main__':
    main()
