# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/6/29 9:01

# 重构代码，主要是修改数据的形式
import copy
import datetime
import os
from typing import Optional

import numpy as np
import sys

import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

sys.path.append('../../Model/DataLoader')

from transformers import (
    AutoConfig,
    set_seed, PreTrainedTokenizer,
)
from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn as nn

import logging
from warnings import simplefilter

from Model.DataLoader import HeartQADataset, HeartQAProcessor

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


try:
    import xml.etree.cElementTree as ElementTree
except ImportError:
    import xml.etree.ElementTree as ElementTree
from tqdm.auto import tqdm

from torch import Tensor


from sklearn.metrics import (
    average_precision_score as map_score,
)

import transformers
transformers.logging.set_verbosity_error()


max_seq_length = 64
logger = logging.getLogger(__name__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HeartQADataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    def __init__(self, tokenizer, data_dir, mode):
        """

        :param tokenizer:
        :param data_dir:
        """
        self.processor = HeartQAProcessor()

        self.Mode = mode
        if mode == "dev":
            df = self.processor.get_dev_examples(data_dir)
            self.Features = self.convert_dev_test_examples_to_features(df, tokenizer, max_length=128)
        elif mode == "test":
            df = self.processor.get_test_examples(data_dir)
            self.Features = self.convert_dev_test_examples_to_features(df, tokenizer, max_length=128)
        else:
            df = self.processor.get_train_examples(data_dir)
            self.Features = self.convert_dev_test_examples_to_features(df, tokenizer, max_length=128)

        self.Labels = Tensor(df['label']).long()
        self.Length = len(self.Features)

    @staticmethod
    def convert_dev_test_examples_to_features(pair_samples: pd.DataFrame,
                                              tokenizer: PreTrainedTokenizer,
                                              max_length: Optional[int] = None,
                                              ):
        """

        :param pair_samples:
        :param tokenizer:
        :param max_length:
        """
        if max_length is None:
            max_length = tokenizer.max_len

        features = tokenizer(
            pair_samples['question'].tolist(),
            pair_samples['answer'].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )['input_ids']

        return features

    def __len__(self):
        """
        #questions
        :return:
        """
        return self.Length

    def __getitem__(self, i):
        dev_test_sample = {
            'feature': self.Features[i],
            'label': self.Labels[i],
        }
        return dev_test_sample


class BertModel(nn.Module):
    def __init__(self, freeze=False, model_name_or_path='bert-base-uncased'):
        super(BertModel, self).__init__()
        # Bert Module
        self.Bert = BertForSequenceClassification.from_pretrained(model_name_or_path)
        self.Config = AutoConfig.from_pretrained(model_name_or_path)
        if freeze:
            for p in self.Bert.parameters():
                p.requires_grad = False

        # self.Softmax = nn.Softmax(dim=-1)

    def forward(self, sample):
        """
        :param sample:
        :return:
        """

        feature = sample['feature'].to(device)

        # Bert Embedding
        out = self.Bert(feature).logits             # torch.Size([10, 2])
        # out = self.Softmax(out)                     # torch.Size([10, 2])

        return out


def metrics(y_true: Tensor, y_score: Tensor):
    y_true = y_true.numpy()
    y_score = y_score.numpy()

    if np.sum(y_true) < 1:
        map_ = 0.
    else:
        map_ = map_score(y_true=y_true, y_score=y_score)

    return map_


def train(task_name, model, train_dataloader, dev_dataloader, epochs, lr):
    temp_csv = f'./Result/Temp/run{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}-{task_name}-dev.csv'
    fine_tune_bin = f'./FineTuneModel/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}/bert-base-uncased'
    model.to(device)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    best_model = None
    best_map = 0.

    n = 0
    for epoch in range(epochs):
        # losses = []
        for sample in tqdm(train_dataloader):
            labels = sample['label']
            optimizer.zero_grad()
            out = model(sample)

            loss = loss_function(input=out, target=labels.view(-1).to(device))
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # losses.append(loss.item())
            # print('Loss {}'.format(np.mean(losses)))

            if n % 10 == 0:
                maps = []
                for dev_sample in dev_dataloader:
                    labels = dev_sample['label']
                    with torch.no_grad():
                        out = model(dev_sample)

                        map_ = metrics(y_true=labels, y_score=out[:, -1].cpu())
                        maps.append(map_)

                temp_map = np.mean(maps)
                temp_result = '{}/{} MAP {}'.format(epoch + 1, epochs, np.mean(maps))
                with open(temp_csv, 'a' if os.path.exists(temp_csv) else 'w') as f:
                    f.write(temp_result)
                    f.write("\n")
                print(temp_result)

                if temp_map > best_map:
                    best_map = temp_map
                    # best_model = model
                    best_model = copy.deepcopy(model)
            n += 1
    print(best_map)
    print(fine_tune_bin)
    best_model.Bert.save_pretrained(fine_tune_bin)

    return best_model, fine_tune_bin


def predict(task_name, model, test_dataloader):
    model.to(device)
    model.eval()

    maps = []
    for test_sample in tqdm(test_dataloader):
        labels = test_sample['label']
        with torch.no_grad():
            out = model(test_sample)

            map_ = metrics(y_true=labels, y_score=out[:, -1].cpu())
            maps.append(map_)

    temp_result = 'MAP || {}'.format(np.mean(maps))
    print(temp_result)


def evaluate(best_model):
    set_seed(123)

    task_name = 'bert'
    data_dir = '../../Data/SemEvalv3.2'
    model_name_or_path = '../../PretrainedModel/bert-base-uncased'

    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    test_dataset = HeartQADataset(
        tokenizer=tokenizer,
        data_dir=data_dir,
        mode='test')

    test_dataloader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Train
    predict(task_name, best_model, test_dataloader)
    # print(heart_qa_model)


def main():
    set_seed(123)

    task_name = 'bert'
    data_dir = '../../Data/SemEvalv3.2'
    model_name_or_path = '../../PretrainedModel/bert-base-uncased'

    tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
    train_dataset = HeartQADataset(
        tokenizer=tokenizer,
        data_dir=data_dir,
        mode='train')
    dev_dataset = HeartQADataset(
        tokenizer=tokenizer,
        data_dir=data_dir,
        mode='dev')

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=10, shuffle=False)

    heart_qa_model = BertModel(freeze=False, model_name_or_path=model_name_or_path)

    # Train
    best_model, fine_tune_bin = train(task_name, heart_qa_model, train_dataloader, dev_dataloader, epochs=3, lr=2e-5)
    # print(heart_qa_model)

    evaluate(best_model)


if __name__ == '__main__':
    main()
