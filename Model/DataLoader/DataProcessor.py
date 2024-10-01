# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/9 19:36
import os
import typing
from abc import ABC
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from pprint import pprint

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
            threshold: int = 1,
            normalize: bool = True,
            return_classes: bool = False,
            limit: int = 0
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

    def get_all_examples(self, data_dir: str) -> DataPack:
        return self.create_examples(os.path.join(data_dir, self.data_name, self.data_name))

    def get_train_examples(self, data_dir: str) -> DataPack:
        return self.create_examples(os.path.join(data_dir, self.data_name, 'Train', 'train'))

    def get_dev_examples(self, data_dir: str) -> DataPack:
        return self.create_examples(os.path.join(data_dir, self.data_name, 'Dev', 'dev'))

    def get_test_examples(self, data_dir: str) -> DataPack:
        return self.create_examples(os.path.join(data_dir, self.data_name, 'Test', 'test'))

    def get_labels(self):
        pass

    def create_examples(self, filepath: str) -> DataPack:
        text_lefts: list = []
        text_right_ids: list = []
        text_rights: list = []
        labels: list = []
        text_others: [list] = []
        extends: list[dict] = []
        features: list = []

        for _, elem in tqdm(self._iterparse(filepath + '.xml', self.limit), desc=f'Parsing {coloring(filepath, "red")} XML file'):
            temp_text_lefts = []
            temp_text_right_ids = []
            temp_text_rights = []
            temp_labels = []
            temp_comments = []

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

                temp_text_lefts.append(q_body)
                temp_text_right_ids.append(a_ids[__])
                temp_text_rights.append(a_bodys[__])
                temp_labels.append(a_scores[__])
                temp_comments.append(c_bodys[__])

            assert len(temp_text_lefts) == len(temp_text_right_ids) == len(temp_text_rights) == len(temp_labels) == len(temp_comments)
            if len(temp_labels) < self.threshold:
                continue

            if self.filtered and self.stage in ('dev', 'test'):
                if all([int(label) <= 0 for label in temp_labels]):
                    continue

            if self.normalize:
                temp_labels = [float(label) for label in temp_labels]
                min_label = min(temp_labels)
                max_label = max(temp_labels)
                temp_labels = [round((label - min_label) / (max_label - min_label), 2) for label in temp_labels]

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

            text_lefts.extend(temp_text_lefts)
            text_right_ids.extend(temp_text_right_ids)
            text_rights.extend(temp_text_rights)
            labels.extend(temp_labels)
            text_others.extend(temp_comments)
            extends.extend(temp_extends)
            features.extend([None] * len(temp_text_right_ids))

        df = pd.DataFrame({
            'text_left': text_lefts,
            'right_id': text_right_ids,
            'text_right': text_rights,
            'label': labels,
            'comment': text_others,
            'extend': extends,
            'feature': features
        })

        data_pack = self.pack(df)

        if not Path.exists(Path(filepath + '.csv')):
            data_pack.frame().to_csv(filepath + '.csv')
        return data_pack

    def pack(self, df: pd.DataFrame):
        # Gather IDs
        id_left = self._gen_ids(df, 'text_left', 'L-')
        id_right = self._gen_ids(df, 'text_right', 'R-')

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
        extend = self._merge(df, id_left, 'extend', 'id_left')
        feature = self._merge(df, id_right, 'feature', 'id_right')

        return DataPack(relation, left, right_id, right, comment, extend, feature)

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

    train_dp = OurProcessor(
        data_name=data_name,
        stage='train',
        task='ranking',
        filtered=False,
        threshold=5,
        normalize=True,
        return_classes=False,
        limit=0
    ).get_train_examples(data_dir)
    pprint(train_dp.frame().iloc[0].to_dict())

    dev_dp = OurProcessor(
        data_name=data_name,
        stage='dev',
        task='ranking',
        filtered=True,
        threshold=5,
        normalize=True,
        return_classes=False,
        limit=0
    ).get_dev_examples(data_dir)
    print(dev_dp.frame())

    test_dp = OurProcessor(
        data_name=data_name,
        stage='test',
        task='ranking',
        filtered=True,
        threshold=5,
        normalize=True,
        return_classes=False,
        limit=0
    ).get_test_examples(data_dir)
    print(test_dp.frame())


if __name__ == '__main__':
    main()
