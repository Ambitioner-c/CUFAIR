# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/8/23 11:17
import os.path
import random
import re
from abc import ABC

import pandas as pd
from tqdm import tqdm
from transformers import DataProcessor

import xml.etree.ElementTree as ET


def cprint(content: str, color: str=None) -> str:
    colors = {
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'purple': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'red_bg': '\033[41m',
        'green_bg': '\033[42m',
        'yellow_bg': '\033[43m',
        'blue_bg': '\033[44m',
        'purple_bg': '\033[45m',
        'cyan_bg': '\033[46m',
        'white_bg': '\033[47m',
    }
    if color is None:
        while True:
            color = random.choice(list(colors.keys()))
            if color.endswith('_bg'):
                continue
            else:
                break
    return f'{colors[color]}{content}\033[0m'


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
                    pings.append(participants_dict[[x for x in participants_dict if x != a_name][0]])
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

    @staticmethod
    def visualize(participants: [str], pings: [int], comments: [str], scores: [int]):
        length = len(pings)

        print('\n', '-' * 100)
        print(f"{cprint('Poster', 'red_bg')}: ", participants[0])

        symbols = [' _____ ' for _ in range(length)]
        links = []
        for _, ping in enumerate(pings):
            if ping:
                symbols[ping - 1] = ' _▂|▂_ '
                symbols[_] = ' _▂|▂_ '

                links.append([ping - 1, _])
        def connection(_link: [int, int]):
            _symbols = ['       ' for _ in range(length)]
            _symbols[_link[0]] = '   ▂▂▂▂'
            _symbols[_link[1]] = '▂▂▂▂   '
            for _ in range(_link[0] + 1, _link[1]):
                _symbols[_] = '▂▂▂▂▂▂▂'
            return _symbols
        for link in sorted(links, reverse=True):
            print('       ', cprint(''.join(connection(link))))
        print('       ', ''.join(symbols))

        print(cprint('Name   ', 'green_bg'), ''.join([f'  {x[:3]}. ' for x in participants[1:]]))

        print(cprint('Index  ', 'yellow_bg'), ''.join([f"{'    ' + str(index + 1) + '  '}"[-7:] for index in range(length)]))

        print(cprint('Ping   ', 'blue_bg'), ''.join([f"{'    ' + str(ping) + '  '}"[-7:] for ping in pings]))

        print(cprint('Score  ', 'purple_bg'), ''.join([f"{'    ' + str(score) + '  '}"[-7:] for score in scores]))


class SEProcessor(DataProcessor, ABC):
    def __init__(
            self,
            data_name: str='meta.stackexchange.com',
            limit: int=0,
            show: bool=False,
            threshold: float=0.5
    ):
        super(SEProcessor, self).__init__()

        self.ping_match = PingMatch()

        self.data_name = data_name
        self.limit = limit
        self.show = show
        self.threshold = threshold

    def get_train_examples(self, data_dir: str) -> pd.DataFrame:
        return self.create_examples(os.path.join(data_dir, self.data_name, self.data_name + '.xml'))

    def get_dev_examples(self, data_dir: str) -> pd.DataFrame:
        pass

    def get_test_examples(self, data_dir: str) -> pd.DataFrame:
        pass

    def get_labels(self):
        pass

    def create_examples(self, filepath: str) -> pd.DataFrame:
        for elem in tqdm(self.iterparse(filepath), desc=f'Parsing {cprint(filepath, 'red')} XML file'):
            # Answer
            for answer in elem.findall('Answer'):
                if int(answer.attrib['COMMENT_COUNT']):
                    participants, pings, comments, scores = self.ping_match.main(answer)
                    if self.show:
                        self.ping_match.visualize(participants, pings, comments, scores)

                    n_s_pairs, s_n_pairs, n_n_pairs = self.pair_nodes(pings, comments, scores)
                    print(n_s_pairs)
                    print(s_n_pairs)
                    print(n_n_pairs)

        # TODO Return a DataFrame
        pass

    def pair_nodes(self, pings: [int], comments: [str], scores: [int]) -> [[str, str], [str, str], [str, str]]:
        average = sum(scores) / len(scores)

        links: [[int, int]] = []
        link_dict: dict[int: int] = {}
        for _, ping in enumerate(pings):
            if ping:
                links.append([ping - 1, _])
                link_dict[_] = ping - 1

        n_s_pairs: [str, str] = []
        s_n_pairs: [str, str] = []
        n_n_pairs: [str, str] = []
        def dig(_ping: int, content: str) -> str:
            if _ping in link_dict:
                content = comments[link_dict[_ping]] + ' ' + content
                dig(link_dict[_ping], content)

            return content

        for link in links:
            if scores[link[0]] > average or scores[link[1]] > average:
                # N-S
                if scores[link[0]] - scores[link[1]] >= self.threshold * average:
                    n_s_pairs.append([dig(link[0], comments[link[0]]), comments[link[1]]])

                # S-N
                if scores[link[1]] - scores[link[0]] >= self.threshold * average:
                    s_n_pairs.append([dig(link[0], comments[link[0]]), comments[link[1]]])

                # N-N
                if abs(scores[link[0]] - scores[link[1]]) < self.threshold * average:
                    n_n_pairs.append([dig(link[0], comments[link[0]]), comments[link[1]]])

        return n_s_pairs, s_n_pairs, n_n_pairs

    def iterparse(self, filepath: str) -> ET.Element:
        content = ET.iterparse(filepath, events=('end',))
        _, root = next(content)

        n = 0
        for event, elem in content:
            if self.limit:
                if elem.tag == 'Thread':
                    yield elem
                    root.clear()

                    n += 1
                    if n == self.limit:
                        break
            else:
                if elem.tag == 'Thread':
                    yield elem
                    root.clear()


def main():
    data_dir = '/home/cuifulai/Projects/CQA/Data/StackExchange'
    data_name = 'meta.stackoverflow.com'

    SEProcessor(
        data_name,
        limit=1,
        show=True,
        threshold=0.5
    ).get_train_examples(data_dir)


if __name__ == '__main__':
    main()
