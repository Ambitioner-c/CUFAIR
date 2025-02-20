# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/8/23 11:17
from typing import Optional

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import PreTrainedTokenizer, AutoTokenizer, set_seed

from Model.Baselines.Ablation.StackExchange.DataLoader.DataProcessor import (
    SEProcessor,
    AnnotatedSEProcessor
)

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


class SEDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            data_dir: str,
            data_name: str,
            limit: int = 0,
            threshold: float = 0.5,
            mode: str = 'All',
            max_length: Optional[int] = None
    ):
        self.processor = SEProcessor(
            data_name=data_name,
            limit=limit,
            threshold=threshold
        )

        self.mode = mode
        if self.mode == 'All':
            self.df = self.processor.get_all_examples(data_dir)
        elif self.mode == 'Train':
            self.df = self.processor.get_train_examples(data_dir)
        elif self.mode == 'Dev':
            self.df = self.processor.get_dev_examples(data_dir)
        else:
            self.df = self.processor.get_test_examples(data_dir)

        self.left_features, self.right_features, self.pair_features = self.convert_examples_to_features(self.df,tokenizer,max_length)
        self.labels = Tensor(self.df['label']).long()

    @staticmethod
    def convert_examples_to_features(
            examples: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            max_length: Optional[int] = None,
    ) -> [Tensor]:
        if max_length is None:
            max_length = tokenizer.model_max_length

        left_features = tokenizer(
            examples['left'].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )['input_ids']

        right_features = tokenizer(
            examples['right'].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )['input_ids']

        pair_features = tokenizer(
            examples['left'].tolist(),
            examples['right'].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )['input_ids']

        return left_features, right_features, pair_features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'left': self.left_features[idx],
            'right': self.right_features[idx],
            'pair': self.pair_features[idx],
            'label': self.labels[idx],
        }


class AnnotatedSEDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            data_dir: str,
            data_name: str,
            model_name: str,
            limit: int = 0,
            mode: str = 'All',
            max_length: Optional[int] = None
    ):
        self.processor = AnnotatedSEProcessor(
            data_name=data_name,
            model_name=model_name,
            limit=limit
        )

        self.mode = mode
        if self.mode == 'All':
            self.df = self.processor.get_all_examples(data_dir)
        elif self.mode == 'Train':
            self.df = self.processor.get_train_examples(data_dir)
        elif self.mode == 'Dev':
            self.df = self.processor.get_dev_examples(data_dir)
        else:
            self.df = self.processor.get_test_examples(data_dir)

        self.left_features, self.right_features, self.pair_features = self.convert_examples_to_features(self.df, tokenizer, max_length)
        self.labels = Tensor(self.df['label']).long()

    @staticmethod
    def convert_examples_to_features(
            examples: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            max_length: Optional[int] = None,
    ) -> [Tensor]:
        if max_length is None:
            max_length = tokenizer.model_max_length

        left_features = tokenizer(
            examples['left'].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )['input_ids']

        right_features = tokenizer(
            examples['right'].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )['input_ids']

        pair_features = tokenizer(
            examples['left'].tolist(),
            examples['right'].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )['input_ids']

        return left_features, right_features, pair_features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'left': self.left_features[idx],
            'right': self.right_features[idx],
            'pair': self.pair_features[idx],
            'label': self.labels[idx],
        }


def main():
    set_seed(2024)

    data_dir = '/home/cuifulai/Projects/CQA/Data/StackExchange'
    data_name = 'meta.stackoverflow.com'
    model_name = 'gpt-4o-2024-08-06'

    pretrained_model_path = '/data/cuifulai/PretrainedModel/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    # all_dataset = SEDataset(
    #     tokenizer,
    #     data_dir,
    #     data_name,
    #     limit=0,
    #     threshold=0.5,
    #     mode='All',
    #     max_length=256
    # )
    all_dataset = AnnotatedSEDataset(
        tokenizer,
        data_dir,
        data_name,
        model_name,
        limit=0,
        mode='All',
        max_length=256
    )
    all_dataloader = DataLoader(all_dataset, batch_size=32, shuffle=False)
    for example in all_dataloader:
        left_feature = example['left']
        right_feature = example['right']
        pair_feature = example['pair']
        label = example['label']
        print(left_feature)
        print(right_feature)
        print(pair_feature)
        print(label)

        break


    train_dataset, dev_dataset, test_dataset = random_split(all_dataset, [0.8, 0.1, 0.1])

    print('Train: ', all_dataset.df.iloc[train_dataset.indices]['label'].value_counts())
    print('Dev: ', all_dataset.df.iloc[dev_dataset.indices]['label'].value_counts())
    print('Test: ', all_dataset.df.iloc[test_dataset.indices]['label'].value_counts())


if __name__ == '__main__':
    main()
