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

from Model.Unit.cprint import *


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
    def visualize(participants: [str], pings: [int], scores: [int], a_id: str, show: bool=False, save: str=None):
        if save:
            f = open(save, 'a', encoding='utf-8')
        else:
            f = None

        length = len(pings)

        if show:
            print(f"\n{'-' * 25}Answer ID: {a_id}{'-' * 25}")
        if save:
            print(f"{'-' * 25}Answer ID: {a_id}{'-' * 25}", file=f)
        if show:
            print(f"{coloring('Poster ', 'red_bg')}", participants[0])
        if save:
            print(f"{coloring('Poster ', 'None')}", participants[0], file=f)

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
            if show:
                print('       ', coloring(''.join(connection(link))))
            if save:
                print('       ', coloring(''.join(connection(link)), 'None'), file=f)
        if show:
            print('       ', ''.join(symbols))
        if save:
            print('       ', ''.join(symbols), file=f)

        if show:
            print(coloring('Name   ', 'green_bg'), ''.join([f'  {x[:3]}. ' for x in participants[1:]]))
        if save:
            print(coloring('Name   ', 'None'), ''.join([f'  {x[:3]}. ' for x in participants[1:]]), file=f)

        if show:
            print(coloring('Index  ', 'yellow_bg'), ''.join([f"{'    ' + str(index + 1) + '  '}"[-7:] for index in range(length)]))
        if save:
            print(coloring('Index  ', 'None'), ''.join([f"{'    ' + str(index + 1) + '  '}"[-7:] for index in range(length)]), file=f)

        if show:
            print(coloring('Ping   ', 'blue_bg'), ''.join([f"{'    ' + str(ping) + '  '}"[-7:] for ping in pings]))
        if save:
            print(coloring('Ping   ', 'None'), ''.join([f"{'    ' + str(ping) + '  '}"[-7:] for ping in pings]), file=f)

        if show:
            print(coloring('Score  ', 'purple_bg'), ''.join([f"{'    ' + str(score) + '  '}"[-7:] for score in scores]))
        if save:
            print(coloring('Score  ', 'None'), ''.join([f"{'    ' + str(score) + '  '}"[-7:] for score in scores]), file=f)

        if save:
            f.close()


class SEProcessor(DataProcessor, ABC):
    def __init__(
            self,
            data_name: str='meta.stackexchange.com',
            limit: int=0,
            show: bool=False,
            save: str=None,
            threshold: float=0.5
    ):
        super(SEProcessor, self).__init__()

        self.ping_match = PingMatch()

        self.data_name = data_name
        self.limit = limit
        self.show = show
        self.save = save
        self.threshold = threshold

    def get_all_examples(self, data_dir: str) -> pd.DataFrame:
        return self.create_examples(os.path.join(data_dir, self.data_name, self.data_name + '.xml'))

    def get_train_examples(self, data_dir: str) -> pd.DataFrame:
        pass

    def get_dev_examples(self, data_dir: str) -> pd.DataFrame:
        pass

    def get_test_examples(self, data_dir: str) -> pd.DataFrame:
        pass

    def get_labels(self):
        pass

    def create_examples(self, filepath: str) -> pd.DataFrame:
        lefts = []
        rights = []
        labels = []
        for elem in tqdm(self.iterparse(filepath), desc=f'Parsing {coloring(filepath, "red")} XML file'):
            # Answer
            for answer in elem.findall('Answer'):
                if int(answer.attrib['COMMENT_COUNT']):
                    participants, pings, comments, scores = self.ping_match.main(answer)
                    if self.show or self.save:
                        self.ping_match.visualize(participants, pings, scores, answer.attrib['ID'], self.show, self.save)

                    if self.threshold >= 0.0:
                        n_s_pairs, s_n_pairs, n_n_pairs = self.pair_nodes(pings, comments, scores)

                        # N-S
                        for pair in n_s_pairs:
                            lefts.append(pair[0])
                            rights.append(pair[1])
                            labels.append(0)

                        # S-N
                        for pair in s_n_pairs:
                            lefts.append(pair[0])
                            rights.append(pair[1])
                            labels.append(1)

                        # N-N
                        for pair in n_n_pairs:
                            lefts.append(pair[0])
                            rights.append(pair[1])
                            labels.append(2)
                    else:
                        pairs = self.pair_nodes(pings, comments, scores)

                        for pair in pairs:
                            lefts.append(pair[0])
                            rights.append(pair[1])
                            labels.append(-1)

        df = pd.DataFrame({
            'left': lefts,
            'right': rights,
            'label': labels
        })

        return df

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
        pairs: [str, str] = []
        def dig(_ping: int, content: str) -> str:
            if _ping in link_dict:
                content = comments[link_dict[_ping]] + ' ' + content
                dig(link_dict[_ping], content)

            return content

        if self.threshold >= 0.0:
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
        else:
            for link in links:
                pairs.append([dig(link[0], comments[link[0]]), comments[link[1]]])
            return pairs

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

    save_path = f"../Result/Interaction/{data_name}.txt"

    df = SEProcessor(
        data_name,
        limit=100,
        show=False,
        save=None,
        threshold=-1.0
    ).get_all_examples(data_dir)
    print(df.head(100).to_csv())


if __name__ == '__main__':
    main()
