# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/9 19:36
from typing import Optional

import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, set_seed, AutoTokenizer

from DataLoader.DataProcessor import OurProcessor


class OurDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            data_dir: str,
            data_name: str,
            limit: int = 0,
            mode: str = 'All',
            max_length: Optional[int] = None
    ):
        self.processor = OurProcessor(
            data_name=data_name,
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

    @staticmethod
    def convert_examples_to_features(
            examples: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            max_length: Optional[int] = None,
    ) -> [Tensor]:
        if max_length is None:
            max_length = tokenizer.model_max_length

        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


def main():
    set_seed(2024)

    data_dir = '/home/cuifulai/Projects/CQA/Data/StackExchange'
    data_name = 'meta.stackoverflow.com'

    pretrained_model_path = '/data/cuifulai/PretrainedModel/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    add_dataset = OurDataset(
        tokenizer=tokenizer,
        data_dir=data_dir,
        data_name=data_name,
        limit=0,
        mode='All',
        max_length=None
    )
    all_dataloader = DataLoader(dataset=add_dataset, batch_size=32, shuffle=True)
    for example in all_dataloader:
        print(example)
        exit()


if __name__ == '__main__':
    main()
