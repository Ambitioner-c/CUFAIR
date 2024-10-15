# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/9 19:36
"""
Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Train/train_1_fold XML file: 832it [00:01, 808.15it/s]
train_1_fold (7221, 10)
Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Test/test_1_fold XML file: 93it [00:00, 607.60it/s]
test_1_fold (253, 10)

Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Train/train_2_fold XML file: 832it [00:01, 800.33it/s]
train_2_fold (7156, 10)
Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Test/test_2_fold XML file: 93it [00:00, 497.07it/s]
test_2_fold (293, 10)

Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Train/train_3_fold XML file: 832it [00:01, 762.91it/s]
train_3_fold (7188, 10)
Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Test/test_3_fold XML file: 93it [00:00, 466.34it/s]
test_3_fold (340, 10)

Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Train/train_4_fold XML file: 832it [00:00, 832.05it/s]
train_4_fold (6956, 10)
Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Test/test_4_fold XML file: 93it [00:00, 504.21it/s]
test_4_fold (283, 10)

Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Train/train_5_fold XML file: 832it [00:01, 754.74it/s]
train_5_fold (7292, 10)
Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Test/test_5_fold XML file: 93it [00:00, 427.45it/s]
test_5_fold (303, 10)

Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Train/train_6_fold XML file: 833it [00:01, 811.51it/s]
train_6_fold (7117, 10)
Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Test/test_6_fold XML file: 92it [00:00, 474.67it/s]
test_6_fold (264, 10)

Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Train/train_7_fold XML file: 833it [00:01, 763.99it/s]
train_7_fold (6911, 10)
Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Test/test_7_fold XML file: 92it [00:00, 377.65it/s]
test_7_fold (274, 10)

Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Train/train_8_fold XML file: 833it [00:01, 776.96it/s]
train_8_fold (7316, 10)
Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Test/test_8_fold XML file: 92it [00:00, 525.06it/s]
test_8_fold (263, 10)

Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Train/train_9_fold XML file: 833it [00:01, 804.40it/s]
train_9_fold (7233, 10)
Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Test/test_9_fold XML file: 92it [00:00, 505.05it/s]
test_9_fold (301, 10)

Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Train/train_10_fold XML file: 833it [00:01, 805.68it/s]
train_10_fold (7232, 10)
Parsing /home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Test/test_10_fold XML file: 92it [00:00, 505.46it/s]
test_10_fold (227, 10)
"""
import os
import typing
from typing import Optional
from abc import ABC
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    DataProcessor,
    set_seed
)

from Model.DataLoader.DataPack import DataPack
from Model.Baselines.Ablation.StackExchange.DataLoader.DataProcessor import PingMatch
from Model.Unit.cprint import coloring


class OurProcessor(DataProcessor, ABC):
    def __init__(
            self,
            data_name: str = 'meta.stackoverflow.com',
            stage: str = 'train',
            task: str = 'ranking',
            filtered: bool = False,
            threshold: int = 5,
            normalize: bool = True,
            return_classes: bool = False,
            limit: int = 0,
            max_length: int = 256,
            max_seq_length: int = 32,
            mode: Optional[str] = 'accept',
            fold: int = 1,
    ):
        super(OurProcessor).__init__()

        self.ping_match = PingMatch()

        self.data_name = data_name
        self.stage = stage
        self.task = task
        self.filtered = filtered
        self.threshold = threshold
        self.normalize = normalize
        self.return_classes = return_classes
        self.limit = limit
        self.max_length = max_length
        self.max_seq_length = max_seq_length
        self.mode = mode
        self.fold = fold

    def get_all_examples(self, data_dir: str) -> DataPack:
        return self.create_examples(os.path.join(data_dir, self.data_name, self.data_name))

    def get_train_examples(self, data_dir: str) -> DataPack:
        return self.create_examples(os.path.join(data_dir, self.data_name, 'Train', f'train_{self.fold}_fold'))

    def get_test_examples(self, data_dir: str) -> DataPack:
        return self.create_examples(os.path.join(data_dir, self.data_name, 'Test', f'test_{self.fold}_fold'))

    def get_labels(self):
        pass

    def create_examples(self, filepath: str) -> DataPack:
        text_left_ids: list = []
        text_lefts: list = []
        text_right_ids: list = []
        text_rights: list = []
        labels: list = []
        text_others: [list] = []
        ping_others: [list] = []
        extends: list[dict] = []
        features: list = []

        for _, elem in tqdm(self._iterparse(filepath + '.xml', self.limit), desc=f'Parsing {coloring(filepath, "red")} XML file'):
            temp_text_left_ids = []
            temp_text_lefts = []
            temp_text_right_ids = []
            temp_text_rights = []
            temp_labels = []
            temp_comments = []
            temp_pings = []

            # Question
            question = elem.find('Question')
            q_id: str = question.attrib['ID']
            q_name: str = question.attrib['OWNER_DISPLAY_NAME']
            q_date: str = question.attrib['CREATION_DATE']
            q_title: str = question.find('QTitle').text
            q_body: str = question.find('QBody').text

            a_ids: list = []
            a_dates: list = []
            a_accepteds: list = []
            a_scores: list = []
            a_bodys: list = []
            a_participants: [list] = []
            a_pings: [list] = []
            c_dates: [list] = []
            c_scores: [list] = []
            c_bodys: [list] = []

            # Answers
            answers = elem.findall('Answer')
            for __, answer in enumerate(answers):
                a_id = answer.attrib['ID']
                a_date = answer.attrib['CREATION_DATE']
                a_accepted = answer.attrib['ACCEPTED_ANSWER']
                a_score = answer.attrib['SCORE']
                a_body = answer.find('ABody').text
                a_ids.append(a_id)
                a_dates.append(a_date)
                a_accepteds.append(a_accepted)
                a_scores.append(a_score)
                a_bodys.append(a_body)

                # Comments
                if len(c_bodys) <= __:
                    c_dates.append([])
                    c_scores.append([])
                    c_bodys.append([])
                if int(answer.attrib['COMMENT_COUNT']):
                    comments = answer.find('AComment').findall('Comment')
                    for ___, comment in enumerate(comments):
                        c_date = comment.attrib['CREATION_DATE']
                        c_score = comment.attrib['SCORE']
                        c_body = comment.find('CBody').text
                        c_scores[__].append(c_score)
                        c_dates[__].append(c_date)
                        c_bodys[__].append(c_body)

                    participants, pings, comments, scores = self.ping_match.main(answer)
                    a_participants.append(participants)
                    a_pings.append(pings)
                else:
                    a_participants.append([])
                    a_pings.append([])

                temp_text_left_ids.append(q_id)
                temp_text_lefts.append(q_body)
                temp_text_right_ids.append(a_ids[__])
                temp_text_rights.append(a_bodys[__])
                if self.mode == 'score':
                    temp_labels.append(a_scores[__])
                elif self.mode == 'accept':
                    temp_labels.append(1 if a_accepteds[__] == 'Yes' else 0)
                temp_comments.append(c_bodys[__])
                temp_pings.append(a_pings[__])

            assert len(temp_text_left_ids) == len(temp_text_lefts) == len(temp_text_right_ids) == len(temp_text_rights) == len(temp_labels) == len(temp_comments) == len(temp_pings)
            if len(temp_labels) < self.threshold:
                continue

            if self.filtered and self.stage in ('dev', 'test'):
                if all([int(label) <= 0 for label in temp_labels]):
                    continue

            if self.normalize:
                if self.mode == 'score':
                    temp_labels = [float(label) for label in temp_labels]
                    min_label = min(temp_labels)
                    max_label = max(temp_labels)
                    temp_labels = [round((label - min_label) / (max_label - min_label), 2) for label in temp_labels]
                elif self.mode == 'accept':
                    temp_labels = [float(label) for label in temp_labels]

            extend = {
                'QID': q_id,
                'QName': q_name,
                'QDate': q_date,
                'QTitle': q_title,
                'QBody': q_body,
                'AIDs': a_ids,
                'ADates': a_dates,
                'ABodys': a_bodys,
                'AAccepteds': a_accepteds,
                'AScores': a_scores,
                'AParticipants': a_participants,
                'APings': a_pings,
                'CScores': c_scores,
                'CDates': c_dates,
                'CBody': c_bodys,
            }
            temp_extends = [extend] * len(temp_labels)

            text_left_ids.extend(temp_text_left_ids)
            text_lefts.extend(temp_text_lefts)
            text_right_ids.extend(temp_text_right_ids)
            text_rights.extend(temp_text_rights)
            labels.extend(temp_labels)
            text_others.extend(temp_comments)
            ping_others.extend(temp_pings)
            extends.extend(temp_extends)
            features.extend([None] * len(temp_text_right_ids))

        df = pd.DataFrame({
            'left_id': text_left_ids,
            'text_left': text_lefts,
            'right_id': text_right_ids,
            'text_right': text_rights,
            'label': labels,
            'comment': text_others,
            'ping': ping_others,
            'extend': extends,
            'feature': features
        })

        data_pack = self.pack(df)

        if not Path.exists(Path(filepath + '.csv')):
            data_pack.frame().to_csv(filepath + '.csv')
        return data_pack

    def pack(self, df: pd.DataFrame):
        # Gather IDs
        id_left = self._gen_ids(df, 'left_id', 'L-')
        id_right = self._gen_ids(df, 'right_id', 'R-')

        # Build Relation
        relation = pd.DataFrame(data={'id_left': id_left, 'id_right': id_right})
        relation['label'] = df['label']
        if self.task == 'classification':
            relation['label'] = relation['label'].astype(int)
        elif self.task == 'ranking':
            relation['label'] = relation['label'].astype(float)
        else:
            raise ValueError(f"{self.task} is not a valid task.")

        # Build Left and Right
        left = self._merge(df, id_left, 'text_left', 'id_left')
        right_id = self._merge(df, id_right, 'right_id', 'id_right')
        right = self._merge(df, id_right, 'text_right', 'id_right')
        comment = self._merge(df, id_right, 'comment', 'id_right')
        ping = self._merge(df, id_right, 'ping', 'id_right')
        extend = self._merge(df, id_left, 'extend', 'id_left')
        feature = self._merge(df, id_right, 'feature', 'id_right')

        return DataPack(relation, left, right_id, right, comment, ping, extend, feature, self.max_length, self.max_seq_length)

    @staticmethod
    def _merge(data: pd.DataFrame, ids: typing.Union[list, np.array], text_label: str, id_label: str):
        df = pd.DataFrame(data={
            text_label: data[text_label], id_label: ids
        })
        df.drop_duplicates(id_label, inplace=True)
        df.set_index(id_label, inplace=True)
        return df

    @staticmethod
    def _gen_ids(data: pd.DataFrame, col: str, prefix: str):
        lookup = {}
        for text in data[col].unique():
            lookup[text] = prefix + str(len(lookup))
        return data[col].map(lookup)

    @staticmethod
    def _iterparse(filepath: str, limit: int) -> (int, ElementTree.Element):
        content = ElementTree.iterparse(filepath, events=('end',))
        _, root = next(content)

        idx = 0
        for event, elem in content:
            if limit:
                if elem.tag == 'Thread':
                    yield idx, elem
                    root.clear()

                    idx += 1
                    if idx == limit:
                        break
            else:
                if elem.tag == 'Thread':
                    yield idx, elem
                    root.clear()

                    idx += 1


def main():
    set_seed(2024)

    data_dir = '/home/cuifulai/Projects/CQA/Data/StackExchange'
    data_name = 'meta.stackoverflow.com'

    fold = 10

    train_dp = OurProcessor(
        data_name=data_name,
        stage='train',
        task='ranking',
        filtered=False,
        threshold=5,
        normalize=True,
        return_classes=False,
        limit=0,
        max_length=256,
        max_seq_length=32,
        mode='accept',
        fold=fold,
    ).get_train_examples(data_dir)
    print(f'train_{fold}_fold', train_dp.frame().shape)

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
        mode='accept',
        fold=fold,
    ).get_test_examples(data_dir)
    print(f'test_{fold}_fold', test_dp.frame().shape)


if __name__ == '__main__':
    main()
