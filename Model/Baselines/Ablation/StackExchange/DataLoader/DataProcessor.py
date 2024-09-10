# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/8/23 11:17
import json
import os.path
import random
import re
from abc import ABC
from time import sleep
from typing import Optional

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm
from transformers import DataProcessor, set_seed

import xml.etree.ElementTree as ElementTree

from Model.Unit.cprint import coloring
from Model.Unit.translate import BaiduTranslate


class Relation(BaseModel):
    type: str
    subtype: str
    description: str


class Detail(BaseModel):
    type: str
    content: str
    explanation: str


class Annotation(BaseModel):
    relation: Relation
    left: Detail
    right: Detail
    category: str


class PingMatch:
    def main(self, answer: ElementTree.Element) -> ([str], [int], [str], [int]):
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
            data_name: str = 'meta.stackexchange.com',
            limit: int = 0,
            show: bool = False,
            save: str = None,
            threshold: float = 0.5
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
        lefts: [str] = []
        rights: [str] = []
        labels: [int] = []
        for elem in tqdm(self.iterparse(filepath, self.limit), desc=f'Parsing {coloring(filepath, "red")} XML file'):
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

    @staticmethod
    def iterparse(filepath: str, limit: int) -> ElementTree.Element:
        content = ElementTree.iterparse(filepath, events=('end',))
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


class AnnotatedSEProcessor(DataProcessor, ABC):
    def __init__(
            self,
            data_name: str = 'meta.stackexchange.com',
            model_name: str = 'gpt-4o-2024-08-06',
            limit: int = 0,
            show: bool = False,
            save: Optional[str] = None,
            sample: [int, int] = None,
            translator: BaiduTranslate = None
    ):
        super(AnnotatedSEProcessor).__init__()

        self.data_name = data_name
        self.model_name = model_name
        self.limit = limit
        self.show = show
        self.save = save
        self.sample = sample
        self.translator = translator

    def get_all_examples(self, data_dir: str) -> pd.DataFrame:
        return self.create_examples(os.path.join(data_dir, self.data_name, 'Annotation', self.model_name, 'rows.txt'))

    def get_train_examples(self, data_dir: str) -> pd.DataFrame:
        pass

    def get_dev_examples(self, data_dir: str) -> pd.DataFrame:
        pass

    def get_test_examples(self, data_dir: str) -> pd.DataFrame:
        pass

    def get_labels(self):
        pass

    def create_examples(self, filepath: str) -> pd.DataFrame:
        sampled_idxs = random.sample(range(self.sample[1]), self.sample[0])

        lefts: [str] = []
        rights: [str] = []
        labels: [int] = []
        for idx, annotation in tqdm(self.iterparse(filepath, self.limit), desc=f'Parsing {coloring(filepath, "red")} TXT file'):

            if self.show or self.save:
                if idx in sampled_idxs:
                    self.visualize(annotation, self.show, self.save, self.translator)

            lefts.append(annotation['left']['content'])
            rights.append(annotation['right']['content'])

            def get_label(_category: str) -> int:
                if 'N-S' in _category:
                    return 0
                elif 'S-N' in _category:
                    return 1
                elif 'N-N' in _category:
                    return 2
                else:
                    raise ValueError(f"Unknown category: {_category}")

            labels.append(get_label(annotation['category']))

        df = pd.DataFrame({
            'left': lefts,
            'right': rights,
            'label': labels
        })

        return df


    @ staticmethod
    def iterparse(filepath: str, limit: int) -> (int, json):
        with open(filepath, 'r', encoding='utf-8') as f:

            idx = 0
            for line in f:
                if limit:
                    yield idx, json.loads(line)
                    idx += 1

                    if idx == limit:
                        break
                else:
                    yield idx, json.loads(line)
                    idx += 1

    @ staticmethod
    def visualize(annotation: json, show: bool, save: Optional[str], translator: BaiduTranslate):
        annotation = Annotation(**annotation)

        relation_type = translator.translate(annotation.relation.type)
        relation_subtype = translator.translate(annotation.relation.subtype)
        relation_description = translator.translate(annotation.relation.description)

        left_content = translator.translate(annotation.left.content)
        left_explanation = translator.translate(annotation.left.explanation)

        right_content = translator.translate(annotation.right.content)
        right_explanation = translator.translate(annotation.right.explanation)

        sleep(random.randint(1, 3))

        if show:
            string = (f"{'{'}\n"
                      f"{' ' * 4 + '"'}relation{'": {'}\n"
                      f"{' ' * 8 + '"'}{coloring('type', 'red')}{'": "'}{annotation.relation.type}{'",'}\n"
                      f"{' ' * 8 + '"'}类型{'": "'}{relation_type}{'",'}\n"
                      f"{' ' * 8 + '"'}subtype{'": "'}{annotation.relation.subtype}{'",'}\n"
                      f"{' ' * 8 + '"'}子类型{'": "'}{relation_subtype}{'",'}\n"
                      f"{' ' * 8 + '"'}description{'": "'}{annotation.relation.description}{'",'}\n"
                      f"{' ' * 8 + '"'}描述{'": "'}{relation_description}{'"'}\n"
                      f"{' ' * 4 + '},'}\n"
                      f"{' ' * 4 + '"'}left{'": {'}\n"
                      f"{' ' * 8 + '"'}{coloring('type', 'red')}{'": "'}{coloring(annotation.left.type, 'red_bg') if annotation.left.type == 'Nucleus' else coloring(annotation.left.type, 'green_bg')}{'",'}\n"
                      f"{' ' * 8 + '"'}类型{'": "'}{'核' if annotation.left.type == 'Nucleus' else '卫星' if annotation.left.type == 'Satellite' else annotation.left.type}{'",'}\n"
                      f"{' ' * 8 + '"'}content{'": "'}{annotation.left.content}{'",'}\n"
                      f"{' ' * 8 + '"'}内容{'": "'}{left_content}{'",'}\n"
                      f"{' ' * 8 + '"'}explanation{'": "'}{annotation.left.explanation}{'",'}\n"
                      f"{' ' * 8 + '"'}解释{'": "'}{left_explanation}{'"'}\n"
                      f"{' ' * 4 + '},'}\n"
                      f"{' ' * 4 + '"'}right{'": {'}\n"
                      f"{' ' * 8 + '"'}{coloring('type', 'red')}{'": "'}{coloring(annotation.right.type, 'red_bg') if annotation.right.type == 'Nucleus' else coloring(annotation.right.type, 'green_bg')}{'",'}\n"
                      f"{' ' * 8 + '"'}类型{'": "'}{'核' if annotation.right.type == 'Nucleus' else '卫星' if annotation.right.type == 'Satellite' else annotation.right.type}{'",'}\n"
                      f"{' ' * 8 + '"'}content{'": "'}{annotation.right.content}{'",'}\n"
                      f"{' ' * 8 + '"'}内容{'": "'}{right_content}{'",'}\n"
                      f"{' ' * 8 + '"'}explanation{'": "'}{annotation.right.explanation}{'",'}\n"
                      f"{' ' * 8 + '"'}解释{'": "'}{right_explanation}{'"'}\n"
                      f"{' ' * 4 + '},'}\n"
                      f"{' ' * 4 + '"'}category{'": "'}{coloring(annotation.category, 'yellow_bg') if 'N-S' in annotation.category else coloring(annotation.category, 'blue_bg') if 'S-N' in annotation.category else coloring(annotation.category, 'purple_bg')}{'"'}\n"
                      f"{'}'}")
            print(string)

        if save:
            obj = {
                '关系(relation)': {
                    'type': annotation.relation.type,
                    '类型': relation_type,
                    'subtype': annotation.relation.subtype,
                    '子类型': relation_subtype,
                    'description': annotation.relation.description,
                    '描述': relation_description
                },
                '左节点(left)': {
                    'type': annotation.left.type,
                    '类型': '核' if annotation.left.type == 'Nucleus' else '卫星' if annotation.left.type == 'Satellite' else annotation.left.type,
                    'content': annotation.left.content,
                    '内容': left_content,
                    'explanation': annotation.left.explanation,
                    '解释': left_explanation
                },
                '右节点(right)': {
                    'type': annotation.right.type,
                    '类型': '核' if annotation.right.type == 'Nucleus' else '卫星' if annotation.right.type == 'Satellite' else annotation.right.type,
                    'content': annotation.right.content,
                    '内容': right_content,
                    'explanation': annotation.right.explanation,
                    '解释': right_explanation
                },
                '类别(category)': annotation.category,
            }
            string = json.dumps(obj, indent=4, ensure_ascii=False)
            with open(save, 'a' if os.path.exists(save) else 'w', encoding='utf-8') as f:
                print(string, file=f)


def main():
    set_seed(2024)

    data_dir = '/home/cuifulai/Projects/CQA/Data/StackExchange'
    data_name = 'meta.stackoverflow.com'
    model_name = 'gpt-4o-2024-08-06'
    save = os.path.join(data_dir, data_name, 'Annotation', model_name, 'samples.txt')

    config_path = '/home/cuifulai/Projects/CQA/config.ini'

    translator = BaiduTranslate(config_path)

    # save_path = f"../Result/Interaction/{data_name}.txt"
    #
    # df = SEProcessor(
    #     data_name,
    #     limit=10,
    #     show=True,
    #     save=None,
    #     threshold=-1.0
    # ).get_all_examples(data_dir)
    # print(df.head(10).to_csv())

    df = AnnotatedSEProcessor(
        data_name,
        model_name,
        limit=0,
        show=True,
        save=save,
        sample=[100, 1000],
        translator=translator
    ).get_all_examples(data_dir)
    # print(df.head(10).to_csv())


if __name__ == '__main__':
    main()
