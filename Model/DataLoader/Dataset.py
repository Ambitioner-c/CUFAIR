# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/9 19:36
import typing
from typing import Optional

import math

import numpy as np
import pandas as pd
import spacy
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, set_seed, AutoTokenizer

from Model.DataLoader.DataPack import DataPack
from Model.DataLoader.DataProcessor import OurProcessor

from warnings import simplefilter

from Model.Our.Dimension.ArgumentQuality import ArgumentQuality

simplefilter(action='ignore', category=FutureWarning)


class OurDataset(Dataset):
    def __init__(
            self,
            argument_quality: ArgumentQuality,
            tokenizer: PreTrainedTokenizer,
            data_pack: DataPack,
            mode: str = 'pair',
            num_dup: int = 1,
            num_neg: int = 1,
            batch_size: int = 32,
            resample: bool = False,
            shuffle: bool = True,
            sort: bool = False,
            max_length: Optional[int] = None
    ):
        if mode not in ('point', 'pair', 'list'):
            raise ValueError(f"{mode} is not a valid mode type. Must be one of `point`, `pair` or `list`.")

        if shuffle and sort:
            raise ValueError(f"parameters `shuffle` and `sort` conflict, should not both be `True`.")

        dp = data_pack.copy()
        self._mode = mode
        self._num_dup = num_dup
        self._num_neg = num_neg
        self._batch_size = batch_size
        self._resample = (resample if mode != 'point' else False)
        self._shuffle = shuffle
        self._sort = sort
        self._orig_relation = dp.relation

        if mode == 'pair':
            dp.relation = self._reorganize_pair_wise(
                relation=self._orig_relation,
                num_dup=num_dup,
                num_neg=num_neg
            )

        self._data_pack = self.convert_examples_to_features(dp, argument_quality, tokenizer, max_length)
        self._batch_indices = None

        self.reset_index()

    @staticmethod
    def convert_examples_to_features(
            data_pack: DataPack,
            argument_quality: ArgumentQuality,
            tokenizer: PreTrainedTokenizer,
            max_length: Optional[int] = None,
    ) -> DataPack:
        dp = argument_quality.get_quality(data_pack)
        # dp = data_pack.copy()

        # left
        left_df = dp.left
        left_features = tokenizer(
            left_df['text_left'].tolist(),
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )['input_ids']
        for _ in range(left_df.shape[0]):
            data_pack.left.iloc[_]['text_left'] = left_features[_]

        # right
        right_df = dp.right

        right_features = tokenizer(
            [text if text is not None else '' for text in right_df['text_right'].tolist()],
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )['input_ids']
        for _ in range(right_df.shape[0]):
            data_pack.right.iloc[_]['text_right'] = right_features[_]

        # Comment
        comment_df = dp.comment
        for _ in range(comment_df.shape[0]):
            comments = comment_df.iloc[_]['comment']
            if len(comments):
                comment_features = tokenizer(
                    [text if text is not None else '' for text in comments],
                    padding='max_length',
                    truncation=True,
                    max_length=max_length,
                    return_tensors='pt'
                )['input_ids']
                data_pack.comment.iloc[_]['comment'] = comment_features
            else:
                data_pack.comment.iloc[_]['comment'] = None

        return data_pack

    def __getitem__(self, item) -> typing.Tuple[dict, np.ndarray, np.ndarray]:
        if isinstance(item, slice):
            indices = sum(self._batch_indices[item], [])
        elif isinstance(item, typing.Iterable):
            indices = [self._batch_indices[i] for i in item]
        else:
            indices = self._batch_indices[item]
        batch_data_pack = self._data_pack[indices]
        x, y, support = batch_data_pack.unpack()
        return x, y, support

    def __len__(self) -> int:
        return len(self._batch_indices)

    def __iter__(self):
        if self._resample or self._shuffle:
            self.on_epoch_end()
        for i in range(len(self)):
            yield self[i]

    def on_epoch_end(self):
        if self._resample:
            self.resample_data()
        self.reset_index()

    def resample_data(self):
        if self.mode != 'point':
            self._data_pack.relation = self._reorganize_pair_wise(
                relation=self._orig_relation,
                num_dup=self._num_dup,
                num_neg=self._num_neg
            )

    def reset_index(self):
        if self._mode == 'point':
            num_instances = len(self._data_pack)
            index_pool = list(range(num_instances))
        elif self._mode == 'pair':
            index_pool = []
            step_size = self._num_neg + 1
            num_instances = int(len(self._data_pack) / step_size)
            for i in range(num_instances):
                lower = i * step_size
                upper = (i + 1) * step_size
                indices = list(range(lower, upper))
                if indices:
                    index_pool.append(indices)
        elif self._mode == 'list':
            raise NotImplementedError(
                f'{self._mode} dataset not implemented.')
        else:
            raise ValueError(f"{self._mode} is not a valid mode type"
                             f"Must be one of `point`, `pair` or `list`.")

        if self._shuffle:
            np.random.shuffle(index_pool)

        if self._sort:
            old_index_pool = index_pool

            max_instance_right_length = []
            for row in range(len(old_index_pool)):
                instance = self._data_pack[old_index_pool[row]].unpack()[0]
                max_instance_right_length.append(max(instance['length_right']))
            sort_index = np.argsort(max_instance_right_length)

            index_pool = [old_index_pool[index] for index in sort_index]

        # batch_indices: index -> batch of indices
        self._batch_indices = []
        for i in range(math.ceil(num_instances / self._batch_size)):
            lower = self._batch_size * i
            upper = self._batch_size * (i + 1)
            candidates = index_pool[lower:upper]
            if self._mode == 'pair':
                candidates = sum(candidates, [])
            self._batch_indices.append(candidates)

    @property
    def num_neg(self):
        return self._num_neg

    @num_neg.setter
    def num_neg(self, value):
        self._num_neg = value
        self.resample_data()
        self.reset_index()

    @property
    def num_dup(self):
        return self._num_dup

    @num_dup.setter
    def num_dup(self, value):
        self._num_dup = value
        self.resample_data()
        self.reset_index()

    @property
    def mode(self):
        return self._mode

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value
        self.reset_index()

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value):
        self._shuffle = value
        self.reset_index()

    @property
    def sort(self):
        return self._sort

    @sort.setter
    def sort(self, value):
        self._sort = value
        self.reset_index()

    @property
    def resample(self):
        return self._resample

    @resample.setter
    def resample(self, value):
        self._resample = value
        self.reset_index()

    @property
    def batch_indices(self):
        return self._batch_indices

    @classmethod
    def _reorganize_pair_wise(
            cls,
            relation: pd.DataFrame,
            num_dup: int = 1,
            num_neg: int = 1
    ):
        pairs = []
        groups = relation.sort_values(
            'label', ascending=False).groupby('id_left')
        for _, group in groups:
            labels = group.label.unique()
            for label in labels[:-1]:
                pos_samples = group[group.label == label]
                pos_samples = pd.concat([pos_samples] * num_dup)
                neg_samples = group[group.label < label]
                for _, pos_sample in pos_samples.iterrows():
                    pos_sample = pd.DataFrame([pos_sample])
                    neg_sample = neg_samples.sample(num_neg, replace=True)
                    pairs.extend((pos_sample, neg_sample))
        new_relation = pd.concat(pairs, ignore_index=True)
        return new_relation



def main():
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
    print(len(test_dataset))
    # print(test_dataset[0])


if __name__ == '__main__':
    main()
