# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/11/12 19:26
import os
from abc import ABC
import xml.etree.ElementTree as ElementTree

import pandas as pd
from tqdm import tqdm
from transformers import DataProcessor

class Ping:
    @staticmethod
    def run(elem: ElementTree.Element) -> list:
        participants = []
        pings = []

        nomination = elem.find('Nomination')
        n_id = nomination.attrib['ID']
        participants.append(n_id)

        comments = elem.find('Comments')
        if int(comments.attrib['CommentCount']):
            for comment in comments.findall('Comment'):
                c_id = comment.attrib['ID']
                participants.append(c_id)

                c_reply_to = comment.attrib['ReplyTo']
                try:
                    pings.append(participants.index(c_reply_to))
                except ValueError:
                    pings.append(0)
        return pings


class OurProcessor(DataProcessor, ABC):
    def __init__(
            self,
            data_name: str = 'AfD',
    ):
        super(OurProcessor).__init__()

        self.data_name = data_name

    def get_train_examples(self, data_dir) -> pd.DataFrame:
        return self.create_examples(os.path.join(data_dir, self.data_name, 'Train'), 'Train')

    def get_dev_examples(self, data_dir):
        return self.create_examples(os.path.join(data_dir, self.data_name, 'Dev'), 'Dev')

    def get_test_examples(self, data_dir):
        return self.create_examples(os.path.join(data_dir, self.data_name, 'Test'), 'Test')

    def get_labels(self):
        pass

    def create_examples(self, filepath: str, mode: str) -> pd.DataFrame:
        text_rights: list = []
        text_others: [list] = []
        ping_others: [list] = []
        labels: list = []
        for elem in tqdm(self.iterparse(filepath + '/' + mode.lower() + '.xml'), desc=f"Parsing {mode.lower() + '.xml'} XML file"):
            label = elem.attrib['Label']
            if label == 'keep':
                labels.append(1)
            elif label == 'delete':
                labels.append(0)

            nomination = elem.find('Nomination')
            n_body = nomination.find('NBody').text
            text_rights.append(n_body)

            c_bodys = []
            comments = elem.find('Comments')
            if int(comments.attrib['CommentCount']):
                for comment in comments.findall('Comment'):
                    c_body = comment.find('CBody').text
                    c_bodys.append(c_body)
            text_others.append(c_bodys)

            pings = Ping.run(elem)
            ping_others.append(pings)

        df = pd.DataFrame({
            'text_right': text_rights,
            'label': labels,
            'comment': text_others,
            'ping': ping_others,
        })
        return df


    @staticmethod
    def iterparse(filename: str):
        tree = ElementTree.parse(filename)
        root = tree.getroot()

        for thread in root.findall('Thread'):
            yield thread


def main():
    data_dir = '/home/cuifulai/Projects/CQA/Data/WikiPedia'
    data_name = 'AfD'

    processor = OurProcessor(
        data_name=data_name,
    )
    processor.get_train_examples(data_dir)


if __name__ == '__main__':
    main()
