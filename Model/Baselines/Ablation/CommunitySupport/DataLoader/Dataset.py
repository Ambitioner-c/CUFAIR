# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/10/28 15:08
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, set_seed, AutoTokenizer

from Model.Baselines.Ablation.CommunitySupport.DataLoader.DataProcessor import CSProcessor


class CSDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            data_name: str,
            data_dir: str,
            limit: int,
            fold: int,
            mode: str = 'train',
            max_length: int = 256,
            seq_length: int = 5,
    ):
        self.processor = CSProcessor(
            data_name=data_name,
            limit=limit,
            fold=fold,
        )

        self.seq_length = seq_length

        if mode == 'train':
            df = self.processor.get_train_examples(data_dir)
        else:
            df = self.processor.get_test_examples(data_dir)

        self.left_features, self.right_features, self.other_features, self.ping_features = self.convert_examples_to_features(df, tokenizer, max_length)

        self.labels = torch.tensor(df['label'].tolist())

    def convert_examples_to_features(
            self,
            df: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            max_length: Optional[int] = None,
    ):
        left_features = tokenizer(
            df['text_left'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )['input_ids']

        right_features = tokenizer(
            [text if text is not None else '' for text in df['text_right'].tolist()],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )['input_ids']

        other_features = []
        for _ in range(df.shape[0]):
            comments = df.iloc[_]['comment']
            comments.extend([''] * self.seq_length)
            other_features.append(tokenizer(
                comments[: self.seq_length],
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )['input_ids'])
        other_features = torch.stack(other_features, dim=0)

        ping_features = []
        for _ in range(df.shape[0]):
            pings = df.iloc[_]['ping']
            pings.extend([0] * self.seq_length)
            ping_features.append(torch.tensor(pings[: self.seq_length], dtype=torch.long))
        ping_features = torch.stack(ping_features, dim=0)


        return left_features, right_features, other_features, ping_features

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {
            'text_left': self.left_features[idx],
            'text_right': self.right_features[idx],
            'comment': self.other_features[idx],
            'ping': self.ping_features[idx],
            'label': self.labels[idx],
        }


def main():
    set_seed(2024)

    data_dir = '/home/cuifulai/Projects/CQA/Data/StackExchange'
    data_name = 'meta.stackoverflow.com'

    pretrained_model_path = '/data/cuifulai/PretrainedModel/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    train_dataset = CSDataset(
        tokenizer=tokenizer,
        data_name=data_name,
        data_dir=data_dir,
        limit=0,
        fold=1,
        mode='train',
        max_length=256,
        seq_length=5,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    for batch in train_dataloader:
        print(batch['ping'])
        break


if __name__ == '__main__':
    main()
