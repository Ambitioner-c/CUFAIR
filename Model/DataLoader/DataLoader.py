# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/10/1 17:11
import typing

import numpy as np
import spacy
import torch
from torch.utils import data
from transformers import set_seed, AutoTokenizer

from Model.DataLoader.DataProcessor import OurProcessor
from Model.DataLoader.Dataset import OurDataset
from Model.Our.Dimension.ArgumentQuality import ArgumentQuality


class DataLoader:
    def __init__(
            self,
            dataset: OurDataset,
            stage='train',
    ):
        if stage not in ('train', 'dev', 'test'):
            raise ValueError(f"{stage} is not a valid stage type. Must be one of `train`, `dev`, `test`.")


        self._dataset = dataset
        self._stage = stage

        self._dataloader = data.DataLoader(
            self._dataset,
            batch_size=None,
            shuffle=False,
            collate_fn=lambda x: x,
            batch_sampler=None,
            num_workers=0,
            pin_memory=False,
            timeout=0,
            worker_init_fn=None,
        )

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def id_left(self) -> torch.Tensor:
        x, _, _ = self._dataset[:]
        return x['id_left']

    @property
    def label(self) -> np.ndarray:
        _, y, _ = self._dataset[:]
        return y.squeeze() if y is not None else None

    @property
    def support(self) -> np.ndarray:
        _, _, support = self._dataset[:]
        return support.squeeze() if support is not None else None

    def __iter__(self) -> typing.Tuple[dict, torch.tensor]:
        for batch_data in self._dataloader:
            x, y, support = batch_data

            batch_x = {}
            for key, value in x.items():
                if key == 'id_left' or key == 'id_right':
                    continue
                batch_x[key] = value

            if y.dtype == 'int':  # task='classification'
                batch_y = torch.tensor(
                    y.squeeze(axis=-1), dtype=torch.long)
            else:  # task='ranking'
                batch_y = torch.tensor(
                    y, dtype=torch.float)

            batch_support = torch.tensor(support, dtype=torch.float)

            yield batch_x, batch_y, batch_support


def situation1():
    set_seed(2024)

    data_dir = '/home/cuifulai/Projects/CQA/Data/StackExchange'
    data_name = 'meta.stackoverflow.com'

    spacy_path = "/data/cuifulai/Spacy/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1"
    nlp = spacy.load(spacy_path)
    argument_quality = ArgumentQuality(nlp)

    pretrained_model_path = '/data/cuifulai/PretrainedModel/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    test_dp = OurProcessor(
        data_name=data_name,
        stage='test',
        task='ranking',
        filtered=True,
        threshold=5,
        normalize=True,
        return_classes=False,
        limit=0,
        max_length=256,
        max_seq_length=32,
        mode='accept'
    ).get_test_examples(data_dir)

    test_dataset = OurDataset(
        argument_quality=argument_quality,
        tokenizer=tokenizer,
        data_pack=test_dp,
        mode='point',
        batch_size=4,
        resample=False,
        shuffle=False,
        max_length=256
    )

    test_dataloader = DataLoader(
        test_dataset,
        stage='test'
    )
    for example in test_dataloader:
        print(example)
        exit()


def situation2():
    set_seed(2024)

    data_dir = '/home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Situation2'
    data_name = 'meta.stackoverflow.com'

    spacy_path = "/data/cuifulai/Spacy/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1"
    nlp = spacy.load(spacy_path)
    argument_quality = ArgumentQuality(nlp)

    pretrained_model_path = '/data/cuifulai/PretrainedModel/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    all_dp = OurProcessor(
        data_name=data_name,
        stage='test',
        task='ranking',
        filtered=False,
        threshold=5,
        normalize=True,
        return_classes=False,
        limit=0,
        max_length=256,
        max_seq_length=5,
        mode='accept',
        situation=2,
    ).get_all_examples(data_dir)

    all_dataset = OurDataset(
        argument_quality=argument_quality,
        tokenizer=tokenizer,
        data_pack=all_dp,
        mode='point',
        batch_size=4,
        resample=False,
        shuffle=False,
        max_length=256
    )

    all_dataloader = DataLoader(
        all_dataset,
        stage='test'
    )
    for example in all_dataloader:
        print(example)
        exit()


def main():
    situation2()


if __name__ == '__main__':
    main()
