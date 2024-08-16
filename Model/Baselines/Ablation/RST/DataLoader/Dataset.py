# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/8/16 19:56
from typing import Optional

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer

from Model.Baselines.Ablation.RST.DataLoader.DataProcessor import RSTProcessor

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)


class RSTDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            data_dir: str,
            mode: str='Train',
            max_length: Optional[int]=None
    ):
        self.processor = RSTProcessor()

        self.mode = mode
        if self.mode == 'Train':
            df = self.processor.get_train_examples(data_dir)
        elif self.mode == 'Dev':
            df = self.processor.get_dev_examples(data_dir)
        else:
            df = self.processor.get_test_examples(data_dir)

        self.left_features, self.right_features = self.convert_examples_to_features(df, tokenizer, max_length)
        self.labels = Tensor(df['label'])

    @staticmethod
    def convert_examples_to_features(
            examples: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            max_length: Optional[int]=None,
    ) -> [Tensor]:
        if max_length is None:
            max_length = tokenizer.model_max_length

        left_features = tokenizer(
            examples['left'].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )['input_ids']

        right_features = tokenizer(
            examples['right'].tolist(),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        )['input_ids']

        return left_features, right_features

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'left': self.left_features[idx],
            'right': self.right_features[idx],
            'label': self.labels[idx],
        }


def main():
    data_dir = '/home/cuifulai/Projects/CQA/Data/RST/GUM'

    pretrained_model_path = '/data/cuifulai/PretrainedModel'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    train_dataset = RSTDataset(tokenizer, data_dir, mode='Train', max_length=128)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for example in train_dataloader:
        left = example['left']
        right = example['right']
        label = example['label']
        print(left)
        print(right)
        print(label)

        break


if __name__ == '__main__':
    main()
