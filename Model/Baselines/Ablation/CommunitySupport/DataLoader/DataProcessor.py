# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/10/28 10:03
import os.path
from abc import ABC
import xml.etree.ElementTree as ElementTree
from pprint import pprint

import pandas as pd
from tqdm import tqdm
from transformers import DataProcessor, set_seed

from Model.Baselines.Ablation.StackExchange.DataLoader.DataProcessor import PingMatch
from Model.Unit.cprint import coloring


class CSProcessor(DataProcessor, ABC):
    def __init__(
            self,
            data_name: str = 'meta.stackoverflow.com',
            limit: int = 0,
            fold: int = 1,
    ):
        super(CSProcessor).__init__()

        self.ping_match = PingMatch()

        self.data_name = data_name
        self.limit = limit
        self.fold = fold

        self.idx2label = None

    def get_train_examples(self, data_dir):
        self.idx2label = self.get_idx2label(os.path.join(data_dir, self.data_name, 'All', 'Annotation', 'Annotation.csv'))
        return self.create_examples(os.path.join(data_dir, self.data_name, 'Train', f'train_{self.fold}_fold'))

    def get_test_examples(self, data_dir):
        self.idx2label = self.get_idx2label(os.path.join(data_dir, self.data_name, 'All', 'Annotation', 'Annotation.csv'))
        return self.create_examples(os.path.join(data_dir, self.data_name, 'Test', f'test_{self.fold}_fold'))

    def get_labels(self):
        pass

    @staticmethod
    def get_idx2label(filepath: str) -> dict:
        dtype = {
            'QuestionID': str,
            'AnswerID': str,
            'Score': float
        }
        df = pd.read_csv(filepath, index_col=0, dtype=dtype)
        return {row['QuestionID']: {row['AnswerID']: row['Score']} for _, row in df.iterrows()}

    def create_examples(self, filepath: str) -> pd.DataFrame:
        text_left_ids: list = []
        text_lefts: list = []
        text_right_ids: list = []
        text_rights: list = []
        text_others: [list] = []
        ping_others: [list] = []
        labels: list = []

        for elem in tqdm(self._iterparse(filepath + '.xml', self.limit), desc=f'Parsing {coloring(filepath, "red")} XML file'):
            q_id = elem.attrib['ID']
            if q_id not in self.idx2label:
                continue

            # Question
            question = elem.find('Question')
            q_body = question.find('QBody').text
            text_left_ids.append(q_id)
            text_lefts.append(q_body)

            # Answers
            answers = elem.findall('Answer')
            for answer in answers:
                a_id = answer.attrib['ID']
                if a_id not in self.idx2label[q_id]:
                    continue

                a_body = answer.find('ABody').text
                text_right_ids.append(a_id)
                text_rights.append(a_body)

                if int(answer.attrib['COMMENT_COUNT']):
                    c_bodys = []
                    comments = answer.find('AComment').findall('Comment')
                    for comment in comments:
                        c_body = comment.find('CBody').text
                        c_bodys.append(c_body)
                    _, pings, _, _ = self.ping_match.main(answer)

                    text_others.append(c_bodys)
                    ping_others.append(pings)
                    labels.append(self.idx2label[q_id][a_id])

        assert len(text_lefts) == len(text_rights) == len(text_others) == len(labels)

        df = pd.DataFrame({
            'left_id': text_left_ids,
            'text_left': text_lefts,
            'right_id': text_right_ids,
            'text_right': text_rights,
            'comment': text_others,
            'ping': ping_others,
            'label': labels
        })
        return df


    @staticmethod
    def _iterparse(filepath: str, limit: int) -> (int, ElementTree.Element):
        content = ElementTree.iterparse(filepath, events=('end',))
        _, root = next(content)

        idx = 0
        for event, elem in content:
            if limit:
                if elem.tag == 'Thread':
                    yield elem
                    root.clear()

                    idx += 1
                    if idx == limit:
                        break
            else:
                if elem.tag == 'Thread':
                    yield elem
                    root.clear()


def main():
    set_seed(2024)

    data_dir = '/home/cuifulai/Projects/CQA/Data/StackExchange'
    data_name = 'meta.stackoverflow.com'

    processor = CSProcessor(
        data_name=data_name,
        limit=0,
        fold=1,
    )
    df = processor.get_train_examples(data_dir)
    pprint(df.head().to_dict())
    df = processor.get_test_examples(data_dir)
    print(df)


if __name__ == '__main__':
    main()
