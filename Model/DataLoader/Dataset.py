# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/6/29 8:55
import math
import sys
from typing import *

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizer, AutoTokenizer,
)
from torch.utils.data.dataset import Dataset
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


try:
    import xml.etree.cElementTree as ElementTree
except ImportError:
    import xml.etree.ElementTree as ElementTree

from Model.DataLoader.DataProcessor import HeartQAProcessor

import pandas as pd


class HeartQADataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer, data_dir, mode="train", k=10):
        """

        :param tokenizer:
        :param data_dir:
        :param mode:
        :param k:
        """
        self.K = k
        self.processor = HeartQAProcessor(k)

        self.Mode = mode
        if mode == "dev":
            df = self.processor.get_dev_examples(data_dir)
            self.QuestionFeatures, self.AnswersFeatures, self.AnswerFeatures, self.PairFeatures = self.convert_examples_to_features(df, tokenizer, max_length=128)
        elif mode == "test":
            df = self.processor.get_test_examples(data_dir)
            self.QuestionFeatures, self.AnswersFeatures, self.AnswerFeatures, self.PairFeatures = self.convert_examples_to_features(df, tokenizer, max_length=128)
        else:
            df = self.processor.get_train_examples(data_dir)
            self.QuestionFeatures, self.AnswersFeatures, self.AnswerFeatures, self.PairFeatures = self.convert_examples_to_features(df, tokenizer, max_length=128)

        self.QuestionID = Tensor(df['question_id']).long()
        self.Label = Tensor(df['label']).long()
        self.AnswerIDs = Tensor(df['answer_ids']).long()
        self.Labels = Tensor(df['labels']).long()
        self.Length = len(self.QuestionFeatures)

    def convert_examples_to_features(self,
                                     samples: pd.DataFrame,
                                     tokenizer: PreTrainedTokenizer,
                                     max_length: Optional[int] = None,
                                     ):
        """

        :param samples:
        :param tokenizer:
        :param max_length:
        """
        if max_length is None:
            max_length = tokenizer.max_len

        question_features = tokenizer(
            samples['question'].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )['input_ids']

        answers = []
        answers_list = samples['answers'].tolist()
        for j in answers_list:
            answers.extend(j)
        answers_features = tokenizer(
            answers,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )['input_ids']
        answers_features_list = answers_features.tolist()
        new_answers_features_list = []
        temp = []
        n = 1
        for j in answers_features_list:
            if n % self.K == 0:
                temp.append(j)
                new_answers_features_list.append(temp)
                temp = []
                n += 1
            else:
                temp.append(j)
                n += 1
        answers_features = Tensor(new_answers_features_list).int()

        answer_features = tokenizer(
            samples['answer'].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )['input_ids']

        pair_features = tokenizer(
            samples['question'].tolist(),
            samples['answer'].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )['input_ids']

        return question_features, answers_features, answer_features, pair_features

    def __len__(self):
        """
        #questions
        :return:
        """
        return self.Length

    def __getitem__(self, i):
        sample = {
            'question': self.QuestionFeatures[i],
            'question_id': self.QuestionID[i],
            'answers': self.AnswersFeatures[i],
            'answer': self.AnswerFeatures[i],
            'pair': self.PairFeatures[i],
            'label': self.Label[i],
            'labels': self.Labels[i],
            'answer_ids': self.AnswerIDs[i]
        }

        return sample


def norm(x, a=0.0, b=1.0):
    return 1 / (math.sqrt(2 * math.pi) * b) * math.exp(- (x - a) ** 2 / (2 * b ** 2))


def prior_weight(question_ids, answer_idss, labelss, b1, b2):
    weightss = []
    inquirer_weightss = []
    respondent_weightss = []

    inquirers = 0
    inquirer_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    inquirer_dict_ = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    respondents = 0
    respondent_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    respondent_dict_ = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}

    inquirer_length = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    respondent_length = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    inquirer_position = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}
    respondent_position = dict()
    for j in range(10):
        for k in range(10):
            respondent_position[str([j+1, k+1])] = 0

    labelss = labelss.tolist()

    for j in range(len(question_ids)):
        question_id = question_ids[j]
        answer_ids = answer_idss[j]

        labels = labelss[j]

        length = len(answer_ids)
        question_weights = [0] * length
        answer_weights = [0] * length
        for _, answer_id in enumerate(answer_ids):
            temp = []
            if question_id == answer_id:
                inquirers += 1
                inquirer_length[_] += 1
                inquirer_position[_ + 1] += 1

                for __ in range(_):
                    temp.append(norm(_ - __, a=0, b=b1))
                    # temp.append(norm((_ - __) * b, a, b))
                for k in range(len(temp)):
                    question_weights[k] += temp[k]

                # if _ == 9:
                if True:
                    # inquirers += 1
                    for m in range(10):
                        if _-m-1 >= 0:
                            if labels[_-m-1] == 1:
                                inquirer_dict[m+1] += 1
                            else:
                                inquirer_dict_[m + 1] += 1

            # print(weights)
            temp = []
            new_answer_ids = answer_ids[_+1:]
            new_labels = labels[_ + 1:]
            if answer_id in new_answer_ids:
                respondents += 1
                try:
                    __ = torch.nonzero(new_answer_ids == answer_id).squeeze()[0].item()
                except IndexError:
                    __ = torch.nonzero(new_answer_ids == answer_id).squeeze().item()
                try:
                    respondent_length[__] += 1
                except KeyError:
                    respondent_length[__.item()] += 1
                respondent_position[str([_+1, _+1+__+1])] += 1
                for ___ in range(__):
                    temp.append(norm(__ - ___, a=0, b=b2))
                for k in range(len(temp)):
                    answer_weights[_ + k + 1] += temp[k]
                # if __ == 5:
                if True:
                    # if question_id == answer_id:
                    #     respondents += 1
                    for m in range(9):
                        if __-m-1 >= 0:
                            if new_labels[__-m-1] == 1:
                                respondent_dict[m+1] += 1
                            else:
                                respondent_dict_[m + 1] += 1

        inquirer_weightss.append(question_weights)
        respondent_weightss.append(answer_weights)
        weights = [x + y + 1 for x, y in zip(question_weights, answer_weights)]
        weightss.append(weights)
    return inquirer_weightss, respondent_weightss, weightss, inquirers, inquirer_dict, inquirer_dict_, respondents, respondent_dict, respondent_dict_, inquirer_length, respondent_length, inquirer_position, respondent_position


def main():
    model_name_or_path = '../../PretrainedModel/bert-base-uncased'

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    dataset = HeartQADataset(tokenizer=tokenizer, data_dir='../../Data/SemEvalv3.2', mode="train", k=10)
    # sample = dataset.__getitem__(0)
    # print(sample)
    # print(sample['question_id'])
    # print(sample['answer_ids'])

    dataloader = DataLoader(dataset, batch_size=10000, shuffle=False)
    for sample in dataloader:
        question_id = sample['question_id']
        answer_ids = sample['answer_ids']
        labelss = sample['labels']
        question_weightss, answer_weightss, weightss, inquirers, inquirer_dict, inquirer_dict_, respondents, respondent_dict, respondent_dict_, inquirer_length, respondent_length, inquirer_position, respondent_position =\
            prior_weight(question_id, answer_ids, labelss, 2, 0.5)

        # print(question_weightss[0])
        # print(answer_weightss[0])
        # print(weightss[0])
        # print(question_id[0])
        # print(answer_ids[0])
        # print(weightss[0])

        print(inquirers)
        print(inquirer_dict)
        print(inquirer_dict_)
        print(respondents)
        print(respondent_dict)
        print(respondent_dict_)

        # print(inquirer_length)
        # print(respondent_length)

        # print(inquirer_position)
        # print(respondent_position)


if __name__ == '__main__':
    main()
    # print(norm(1, 0, 4))
    # print(norm(2, 0, 4))
    # print(norm(3, 0, 4))
    # print(norm(4, 0, 4))
    # print(norm(5, 0, 4))
    #
    # print(norm(1, 0, 0.25))
    # print(norm(2, 0, 0.25))
    # print(norm(3, 0, 0.25))
    # print(norm(4, 0, 0.25))
    # print(norm(5, 0, 0.25))
