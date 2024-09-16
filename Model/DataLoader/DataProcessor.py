# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/9 19:36
import os
from abc import ABC
import xml.etree.ElementTree as ElementTree
from pprint import pprint

import pandas as pd
from tqdm import tqdm
from transformers import (
    DataProcessor,
    set_seed
)


from Model.Baselines.Ablation.StackExchange.DataLoader.DataProcessor import PingMatch
from Unit.cprint import coloring


class OurProcessor(DataProcessor, ABC):
    def __init__(
            self,
            data_name: str='meta.stackoverflow.com',
            limit: int=0
    ):
        super(OurProcessor).__init__()

        self.ping_match = PingMatch()

        self.data_name = data_name
        self.limit = limit

    def get_all_examples(self, data_dir: str) -> pd.DataFrame:
        return self.create_examples(os.path.join(data_dir, self.data_name, self.data_name + '.xml'))

    def get_train_examples(self, data_dir: str) -> pd.DataFrame:
        return self.create_examples(os.path.join(data_dir, self.data_name, 'Train', 'train.xml'))

    def get_dev_examples(self, data_dir: str) -> pd.DataFrame:
        return self.create_examples(os.path.join(data_dir, self.data_name, 'Dev', 'dev.xml'))

    def get_test_examples(self, data_dir: str) -> pd.DataFrame:
        return self.create_examples(os.path.join(data_dir, self.data_name, 'Test', 'test.xml'))

    def get_labels(self):
        pass

    def create_examples(self, filepath: str) -> pd.DataFrame:
        q_ids: [str] = []
        q_names: [str] = []
        q_titles: [str] = []
        q_bodys: [str] = []
        a_ids: [list] = []
        a_bodys: [list] = []
        a_accepteds: [list] = []
        a_scores: [list] = []
        a_participants: [list] = []
        a_pings: [list] = []
        c_bodys: [[list]] = []
        c_scores: [[list]] = []
        for _, elem in tqdm(self.iterparse(filepath, self.limit), desc=f'Parsing {coloring(filepath, "red")} XML file'):
            # Question
            question = elem.find('Question')
            q_ids.append(question.attrib['ID'])
            q_names.append(question.attrib['OWNER_DISPLAY_NAME'])
            q_titles.append(question.find('QTitle').text)
            q_bodys.append(question.find('QBody').text)

            # Answers
            if len(a_ids) <= _:
                a_ids.append([])
                a_bodys.append([])
                a_accepteds.append([])
                a_scores.append([])
                a_participants.append([])
                a_pings.append([])
                c_bodys.append([])
                c_scores.append([])
            answers = elem.findall('Answer')
            for __, answer in enumerate(answers):
                a_ids[_].append(answer.attrib['ID'])
                a_bodys[_].append(answer.find('ABody').text)
                a_accepteds[_].append(answer.attrib['ACCEPTED_ANSWER'])
                a_scores[_].append(answer.attrib['SCORE'])

                # Comments
                if len(c_bodys[_]) <= __:
                    c_bodys[_].append([])
                    c_scores[_].append([])
                if int(answer.attrib['COMMENT_COUNT']):
                    comments = answer.find('AComment').findall('Comment')
                    for ___, comment in enumerate(comments):
                        c_bodys[_][__].append(comment.find('CBody').text)
                        c_scores[_][__].append(comment.attrib['SCORE'])

                    participants, pings, comments, scores = self.ping_match.main(answer)
                    a_participants[_].append(participants)
                    a_pings[_].append(pings)
                else:
                    a_participants[_].append([])
                    a_pings[_].append([])

        df = pd.DataFrame({
            'QID': q_ids,
            'QName': q_names,
            'QTitle': q_titles,
            'QBody': q_bodys,
            'AID': a_ids,
            'ABody': a_bodys,
            'AAccepted': a_accepteds,
            'AScore': a_scores,
            'AParticipants': a_participants,
            'APings': a_pings,
            'CBody': c_bodys,
            'CScore': c_scores
        })
        return df

    @staticmethod
    def iterparse(filepath: str, limit: int) -> (int, ElementTree.Element):
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
    limit = 0

    processor = OurProcessor(data_name, limit)
    df = processor.get_all_examples(data_dir)
    # example = df.head(1)
    # example = example.to_dict(orient='dict')
    # pprint(example)
    print(df.shape)

    df = processor.get_train_examples(data_dir)
    print(df.shape)

    df = processor.get_dev_examples(data_dir)
    print(df.shape)

    df = processor.get_test_examples(data_dir)
    print(df.shape)


if __name__ == '__main__':
    main()
