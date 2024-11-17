# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/11/17 17:45
import xml.etree.ElementTree as ElementTree

from tqdm import tqdm
from transformers import set_seed


class Statistic:
    def __init__(
            self,
            data_dir: str,
            data_name: str,
    ):
        self.data_dir = data_dir
        self.data_name = data_name


    def run(self, mode: str):
        if mode == 'All':
            xml_path = f'{self.data_dir}/{self.data_name}/{self.data_name}.xml'
        elif mode == 'Train':
            xml_path = f'{self.data_dir}/{self.data_name}/Train/train.xml'
        elif mode == 'Dev':
            xml_path = f'{self.data_dir}/{self.data_name}/Dev/dev.xml'
        elif mode == 'Test':
            xml_path = f'{self.data_dir}/{self.data_name}/Test/test.xml'
        else:
            raise ValueError('Invalid')

        num_keep = 0
        num_delete = 0
        num_n = 0
        n_max_length = 0
        n_min_length = 1000000
        n_total_length = 0
        num_c = 0
        c_max_length = 0
        c_min_length = 1000000
        c_total_length = 0
        c_max_number = 0
        c_min_number = 1000000
        c_total_number = 0
        p_max_number = 0
        p_min_number = 1000000
        p_total_number = 0
        for elem in tqdm(self.iterparse(xml_path), desc="Parsing {} XML file".format(xml_path)):
            label = elem.attrib['Label']
            if label == 'keep':
                num_keep += 1
            elif label == 'delete':
                num_delete += 1

            nomination = elem.find('Nomination')
            num_n += 1
            n_body = nomination.find('NBody').text
            try:
                n_length = len(n_body.split())
            except AttributeError:
                n_length = 0
            n_max_length = max(n_max_length, n_length)
            n_min_length = min(n_min_length, n_length)
            n_total_length += n_length

            participants = []
            pings = []
            n_id = nomination.attrib['ID']
            participants.append(n_id)

            comments = elem.find('Comments')
            c_number = int(comments.attrib['CommentCount'])
            c_max_number = max(c_max_number, c_number)
            c_min_number = min(c_min_number, c_number)
            c_total_number += c_number
            if int(comments.attrib['CommentCount']):
                for comment in comments.findall('Comment'):
                    num_c += 1
                    c_body = comment.find('CBody').text
                    try:
                        c_length = len(c_body.split())
                    except AttributeError:
                        c_length = 0
                    c_max_length = max(c_max_length, c_length)
                    c_min_length = min(c_min_length, c_length)
                    c_total_length += c_length

                    c_id = comment.attrib['ID']
                    participants.append(c_id)

                    c_reply_to = comment.attrib['ReplyTo']
                    try:
                        pings.append(participants.index(c_reply_to))
                    except ValueError:
                        pings.append(0)
            p_number = len([ping for ping in pings if ping != 0])
            p_max_number = max(p_max_number, p_number)
            p_min_number = min(p_min_number, p_number)
            p_total_number += p_number
        print(f'num_keep: {num_keep}')
        print(f'num_delete: {num_delete}')
        print(f'num_n: {num_n}')
        print(f'n_max_length: {n_max_length}')
        print(f'n_min_length: {n_min_length}')
        print(f'n_avg_length: {n_total_length / num_n}')
        print(f'num_c: {num_c}')
        print(f'c_max_length: {c_max_length}')
        print(f'c_min_length: {c_min_length}')
        print(f'c_avg_length: {c_total_length / num_c}')
        print(f'c_max_number: {c_max_number}')
        print(f'c_min_number: {c_min_number}')
        print(f'c_avg_number: {c_total_number / num_n}')
        print(f'p_max_number: {p_max_number}')
        print(f'p_min_number: {p_min_number}')
        print(f'p_avg_number: {p_total_number / num_n}')


    @staticmethod
    def iterparse(filename: str):
        tree = ElementTree.parse(filename)
        root = tree.getroot()

        for thread in root.findall('Thread'):
            yield thread


def main():
    set_seed(2024)

    data_dir = '/home/cuifulai/Projects/CQA/Data/WikiPedia'
    data_name = 'AfD'

    mode = 'Test'

    statistic = Statistic(
        data_dir=data_dir,
        data_name=data_name,
    )
    statistic.run(mode)


if __name__ == '__main__':
    main()
