# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/11/12 21:18
from typing import Optional

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer

from Model4AfD.DataLoader.DataProcessor import OurProcessor

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


class OurDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            data_dir: str,
            data_name: str = 'AfD',
            max_length: Optional[int] = None,
            mode: str = 'train',
            seq_length: int = 10,
    ):
        processor = OurProcessor(data_name)

        if mode == 'train':
            df = processor.get_train_examples(data_dir)
        elif mode == 'dev':
            df = processor.get_dev_examples(data_dir)
        elif mode == 'test':
            df = processor.get_test_examples(data_dir)
        else:
            raise ValueError('mode must be one of [train, dev, test]')

        self.right_features, self.comments_features, self.pings_features = self.convert_examples_to_features(df, tokenizer, max_length, seq_length)
        self.labels = Tensor(df['label']).long()

    @staticmethod
    def convert_examples_to_features(
            df: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            max_length: Optional[int] = None,
            seq_length: int = 10,
    ):
        right_features = tokenizer(
            df['text_right'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )['input_ids']

        comments_features = []
        comments_list = df['comment'].tolist()
        for comments in comments_list:
            comments = [x if x is not None else '' for x in comments]
            comments.extend([''] * seq_length)
            comments = comments[:seq_length]

            features = tokenizer(
                comments,
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )['input_ids']
            comments_features.append(features)
        comments_features = torch.stack(comments_features, dim=0)

        pings_features = []
        pings_list = df['ping'].tolist()
        for pings in pings_list:
            pings.extend([0] * seq_length)
            pings = pings[:seq_length]
            pings_features.append(torch.tensor(pings))

        return right_features, comments_features, pings_features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return {
            'right': self.right_features[item],
            'comments': self.comments_features[item],
            'label': self.labels[item],
            'ping': self.pings_features[item],
        }


def main():
    data_dir = '/home/cuifulai/Projects/CQA/Data/WikiPedia'
    data_name = 'AfD'

    pretrained_model_path = '/data/cuifulai/PretrainedModel/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    dev_dataset = OurDataset(
        tokenizer=tokenizer,
        data_dir=data_dir,
        data_name=data_name,
        max_length=128,
        mode='dev',
        seq_length=10,
    )

    dev_dataloader = DataLoader(dev_dataset, batch_size=10, shuffle=False)
    for dev_sample in dev_dataloader:
        print(dev_sample)
        break


if __name__ == '__main__':
    main()
