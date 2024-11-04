# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/11/4 15:06
import re
import xml.etree.ElementTree as ElementTree
from time import sleep
from xml.dom import minidom

from tqdm import tqdm


def read_related_questions(filepath: str):
    rel_dict = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            original_left_id, candidate_left_id, _ = line.strip().split('\t')
            if original_left_id not in rel_dict:
                rel_dict[original_left_id] = [candidate_left_id]
            else:
                rel_dict[original_left_id].append(candidate_left_id)
    return rel_dict


class Restructure:
    def __init__(
            self,
            data_name: str = 'meta.stackoverflow.com',
            rel_dict: dict = None,
            save: str = None,
    ):
        self.data_name = data_name
        self.rel_dict = rel_dict
        self.save = save

        with open(f'./{self.data_name}/{self.data_name}.xml', 'r', encoding='utf-8') as f:
            self.root = ElementTree.parse(f).getroot()

    def run(self):
        for elem in tqdm(self.iterparse(f'./{self.data_name}/{self.data_name}.xml'), desc=f"Restructure {self.data_name} XML file"):
            # Question
            question = elem.find('Question')
            q_id = question.attrib['ID']
            if q_id not in self.rel_dict:
                continue

            # Answers
            answers = elem.findall('Answer')
            assert len(answers) == 2
            length = len(answers)

            # Candidate Threads
            c_q_ids = self.rel_dict[q_id]
            for c_q_id in c_q_ids:
                c_q_elem = self.search(c_q_id)

                if c_q_elem is not None:
                    # Candidate Question
                    c_question = c_q_elem.find('Question')

                    # Candidate Answers
                    c_answers = c_q_elem.findall('Answer')

                    # Add Answers
                    for c_answer in c_answers:
                        c_answer.append(c_question)
                        elem.append(c_answer)
                    length += len(c_answers)
                    if length >= 5:
                        break
            self.write_file(elem)

    def write_file(self, thread):
        is_writing = True

        string = (re.sub(
            r'> *\n +', '>', minidom.parseString(
                ElementTree.tostring(thread, encoding='utf-8', method='html')
            ).toprettyxml(indent='  ').replace('<?xml version="1.0" ?>\n', '')
        ).replace('\n\n', '\n')
         .replace('\n  \n', '\n')
         .replace('\n    \n', '\n')
         .replace('\n        \n', '\n')
         .replace('\n            \n', '\n')
         .replace('\n                \n', '\n'))

        while is_writing:
            try:
                with open(self.save, 'a', encoding='utf-8') as f:
                    f.write(string)
                is_writing = False
            except PermissionError:
                sleep(1)
                continue

    def search(self, post_id):
        for elem in self.root.findall(".//Thread[@ID='{}']".format(post_id)):
            return elem

    @staticmethod
    def iterparse(filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            context = ElementTree.iterparse(f, events=('end',))
            _, root = next(context)

            for event, elem in context:
                if elem.tag == 'Thread':
                    yield elem
                    root.clear()


def main():
    data_name = 'meta.stackoverflow.com'

    related_questions_filepath = f'./{data_name}/Situation2/related_questions.txt'
    rel_dict = read_related_questions(related_questions_filepath)

    save = f'./{data_name}/Situation2/{data_name}.xml'

    restructure = Restructure(
        data_name=data_name,
        rel_dict=rel_dict,
        save=save
    )
    restructure.run()


if __name__ == '__main__':
    main()
