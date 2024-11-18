# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/10/21 19:53
import random
import xml.etree.ElementTree as ElementTree

import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm
from transformers import set_seed

from pathlib import Path


class Extract:
    def __init__(
            self,
            data_name: str = 'meta.stackoverflow.com',
            limit: int = 0,
            mode: str = 'accept',
            kf: KFold = None,
            seed: int = 2024,
            delimiter: str = '--Fulai Cui--',
    ):
        self.data_name = data_name
        self.limit = limit
        self.mode = mode
        self.kf = kf
        self.seed = seed
        self.delimiter = delimiter

    def run(self):
        text_left_ids: list = []
        text_left_names: list = []
        text_left_times: list = []
        text_lefts: list = []
        text_right_ids: list = []
        text_right_names: list = []
        text_right_times: list = []
        text_rights: list = []
        text_other_names: list = []
        text_other_times: list = []
        text_others: list = []

        elems = []
        for elem in tqdm(self.iterparse(f'./{self.data_name}/All/all.xml', limit=self.limit), desc=f"Splitting {self.data_name} All XML file"):
            # Answers
            answers = elem.findall('Answer')
            elems.append(elem.attrib['ID'])

            if self.mode == 'accept':
                is_exist = sum([1 for answer in answers if answer.attrib['ACCEPTED_ANSWER'] == 'Yes'])
                if is_exist and random.choice([True, False]):
                    candidates = []
                    for answer in answers:
                        if int(answer.attrib['COMMENT_COUNT']):
                            candidates.append(answer)
                    if len(candidates):
                        answer = random.choice(candidates)

                        q_id = elem.attrib['ID']
                        q_name = elem.find('Question').attrib['OWNER_DISPLAY_NAME']
                        q_time = elem.find('Question').attrib['CREATION_DATE'].replace('T', ' ')
                        q_body = elem.find('Question').find('QBody').text
                        a_id = answer.attrib['ID']
                        a_name = answer.attrib['OWNER_DISPLAY_NAME']
                        a_time = answer.attrib['CREATION_DATE'].replace('T', ' ')
                        a_body = answer.find('ABody').text
                        c_name = [comment.attrib['DISPLAY_NAME'] for comment in answer.find('AComment').findall('Comment')]
                        c_time = [comment.attrib['CREATION_DATE'].replace('T', ' ') for comment in answer.find('AComment').findall('Comment')]
                        c_body = [comment.find('CBody').text for comment in answer.find('AComment').findall('Comment')]
                        text_left_ids.append(q_id)
                        text_left_names.append(q_name)
                        text_left_times.append(q_time)
                        text_lefts.append(q_body)
                        text_right_ids.append(a_id)
                        text_right_names.append(a_name)
                        text_right_times.append(a_time)
                        text_rights.append(a_body)
                        text_other_names.append(self.delimiter.join(c_name))
                        text_other_times.append(self.delimiter.join(c_time))
                        text_others.append(self.delimiter.join(c_body))

        df = pd.DataFrame({
            'question_id': text_left_ids,
            'questioner_name': text_left_names,
            'question_time': text_left_times,
            'question': text_lefts,
            'answer_id': text_right_ids,
            'answerer_name': text_right_names,
            'answer_time': text_right_times,
            'answer': text_rights,
            'commenter_name': text_other_names,
            'comment_time': text_other_times,
            'comment': text_others,
        })
        print(df)
        filepath = Path(__file__).parent.joinpath(f'{self.data_name}/All/labeled.csv')
        if not Path.exists(filepath):
            df.to_csv(filepath, index=False)

        left_id_set = set(df['question_id'])
        for i, (train_index, test_index) in enumerate(self.kf.split(elems)):
            train_elems = [elems[j] for j in train_index]
            print('#Train:', sum(1 for elem in train_elems if elem in left_id_set))
            test_elems = [elems[j] for j in test_index]
            print('#Test', sum(1 for elem in test_elems if elem in left_id_set))

    @staticmethod
    def iterparse(filepath: str, limit: int):
        with open(filepath, 'r', encoding='utf-8') as f:
            context = ElementTree.iterparse(f, events=('end',))
            _, root = next(context)

            n = 0
            for event, elem in context:
                if elem.tag == 'Thread':
                    if limit:
                        yield elem
                        root.clear()
                        n += 1

                        if n == limit:
                            break
                    else:
                        yield elem
                        root.clear()
                        n += 1


def main():
    seed = 2024
    set_seed(seed)

    data_name = 'meta.stackoverflow.com'
    limit = 0

    n_splits = 10
    shuffle = True

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    mode = 'accept'
    delimiter = '--Fulai Cui--'

    Extract(data_name=data_name, limit=limit, mode=mode, kf=kf, seed=seed, delimiter=delimiter).run()


if __name__ == '__main__':
    main()
