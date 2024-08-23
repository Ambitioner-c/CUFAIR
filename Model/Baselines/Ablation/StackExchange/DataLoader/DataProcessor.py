# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/8/23 11:17
import os.path
import re
from abc import ABC

import pandas as pd
from tqdm import tqdm
from transformers import DataProcessor

import xml.etree.ElementTree as ET


def cprint(content: str):
    return f'\033[33m{content}\033[0m'


class PingMatch:
    def main(self, answer: ET.Element) -> ([str], [int], [str]):
        a_name = answer.attrib['OWNER_DISPLAY_NAME']

        participants: [str] = list()
        participants_dict: {str: int} = dict()
        pings: [int] = list()
        comments: [str] = list()
        scores: [int] = list()

        participants.append(a_name)
        participants_dict[a_name] = 0

        # Comments
        for c_idx, comment in enumerate(answer.find('AComment').findall('Comment')):
            c_name = comment.attrib['DISPLAY_NAME']
            c_body = comment.find('CBody').text
            c_score = int(comment.attrib['SCORE'])

            if len(participants_dict) == 1:
                if c_name == a_name:
                    pings.append(0)
                else:
                    pings.append(participants_dict[a_name])
            elif len(participants_dict) == 2 and c_name in participants_dict:
                if c_name == a_name:
                    pings.append(participants_dict[x] for x in participants_dict if x != a_name)
                else:
                    pings.append(participants_dict[a_name])
            else:
                index = self.interaction(participants, self.parse_ping(c_body))
                pings.append(index)


            participants.append(c_name)
            participants_dict[c_name] = c_idx + 1
            comments.append(c_body)
            scores.append(c_score)

        return participants, pings, comments, scores

    def interaction(self, participants: [str], names: [str]) -> int:
        if names:
            # # fuzzy matching
            # best_match = process.extractOne(names[-1], participants)

            # rule-based matching
            best_match = self.matching(names[-1], participants)

            score = best_match[1]
            index = best_match[2]

            # 90 for fuzzy matching
            if score > 90:
                return index
            else:
                return 0
        else:
            return 0


    @staticmethod
    def matching(query: str, choices: [str]) -> (str, float, int):
        temp = None, 0.0, None

        for index, choice in enumerate(choices):
            if len(query) < 3:
                if query.lower() == choice.split(' ')[0].lower():
                    temp = choice, 100.0, index
            else:
                if choice.lower().replace(' ', '').startswith(query.lower()):
                    temp = choice, 100.0, index

        return temp


    @staticmethod
    def parse_ping(body: str) -> [str]:
        try:
            body = '🐉' + body.replace('...', '.') + '🐉'
        except AttributeError:
            return None

        # Step 1
        """
        在 body 字符串中搜索以空白字符或者 '🐉' 字符开头，
        紧跟着 @ 符号，
        然后后面跟着至少两个连续的非特定字符（特定字符指的是空白字符、逗号、冒号、斜杠、问号、感叹号、'🐉'字符、'['字符、']'字符、或者'('')'字符，
        这部分就是一个用户名，然后提取这个用户名。
        """
        match = re.findall(r'[\s🐉]@([^\s:,/!?🐉\[\]()：，！？【】（）。]{2,})', body)[:2]
        if match:
            names = []
            for name in match:
                # Step 2
                if name.endswith('.'):
                    if len(name) > 2:
                        name = name[: -1]

                # Step 3
                if name.endswith("'"):
                    name = name[: -1]
                elif name.endswith("'s"):
                    name = name[: -2]

                names.append(name)

            return names
        else:
            return None


class SEProcessor(DataProcessor, ABC):
    def __init__(self, data_name: str='meta.stackexchange.com'):
        super(SEProcessor, self).__init__()

        self.ping_match = PingMatch()

        self.data_name = data_name

    def get_train_examples(self, data_dir: str) -> pd.DataFrame:
        return self.create_examples(os.path.join(data_dir, self.data_name, self.data_name + '.xml'), limit=1)

    def get_dev_examples(self, data_dir: str) -> pd.DataFrame:
        pass

    def get_test_examples(self, data_dir: str) -> pd.DataFrame:
        pass

    def get_labels(self):
        pass

    def create_examples(self, filepath: str, limit: int) -> pd.DataFrame:

        for elem in tqdm(self.iterparse(filepath, limit), desc=f'Parsing {cprint(filepath)} XML file'):
            # Answer
            for answer in elem.findall('Answer'):
                if int(answer.attrib['COMMENT_COUNT']):
                    participants, pings, comments, scores = self.ping_match.main(answer)
                    print(participants)
                    print(pings)
                    print(comments)
                    print(scores)

        # TODO Return a DataFrame
        pass

    @staticmethod
    def iterparse(filepath: str, limit: int=0) -> ET.Element:
        content = ET.iterparse(filepath, events=('end',))
        _, root = next(content)

        n = 0
        for event, elem in content:
            if limit:
                if elem.tag == 'Thread':
                    yield elem
                    root.clear()

                    n += 1
                    if n == limit:
                        break
            else:
                if elem.tag == 'Thread':
                    yield elem
                    root.clear()


def main():
    data_dir = '/home/cuifulai/Projects/CQA/Data/StackExchange'
    data_name = 'meta.stackoverflow.com'

    SEProcessor(data_name).get_train_examples(data_dir)


if __name__ == '__main__':
    main()
