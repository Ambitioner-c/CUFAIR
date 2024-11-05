# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/11/5 14:54
import xml.etree.ElementTree as ElementTree
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def read_id2idxmax(filepath: str) -> dict:
    id2idxmax = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            idx, idxmax = line.strip().split(',')
            id2idxmax[int(idx.replace('L-', ''))] = int(idxmax)
    return id2idxmax


class Extract:
    def __init__(
            self,
            data_name: str = 'meta.stackoverflow.com',
            id2idxmax: dict = None,
            delimiter: str = '--Fulai Cui--',
            model: str = None,
    ):
        self.data_name = data_name
        self.id2idxmax = id2idxmax
        self.delimiter = delimiter
        self.model = model

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

        for idx, elem in tqdm(self.iterparse(f'./{self.data_name}/Situation2/{self.data_name}.xml'), desc=f"Splitting {self.data_name} All XML file"):
            # Answers
            answers = elem.findall('Answer')
            answer = answers[self.id2idxmax[idx]]

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
        filepath = Path(__file__).parent.joinpath(f'{self.data_name}/Situation2/Target/{self.model}.csv')
        if not Path.exists(filepath):
            df.to_csv(filepath, index=False)


    @staticmethod
    def iterparse(filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            context = ElementTree.iterparse(f, events=('end',))
            _, root = next(context)

            idx = 0
            for event, elem in context:
                if elem.tag == 'Thread':
                    yield idx, elem
                    root.clear()

                    idx += 1


def main():
    model = 'Zhou_BiGRU4Situation2-20241027_213304'
    id2idxmax = read_id2idxmax(f'./meta.stackoverflow.com/Situation2/Source/{model}.csv')

    Extract(
        data_name='meta.stackoverflow.com',
        id2idxmax=id2idxmax,
        delimiter='--Fulai Cui--',
        model=model
    ).run()


if __name__ == '__main__':
    main()
