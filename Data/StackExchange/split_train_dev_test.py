# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/16 17:11
import re
import xml.etree.ElementTree as ElementTree
from typing import Optional
import xml.dom.minidom as minidom

from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Split:
    def __init__(
            self,
            data_name: str = 'meta.stackoverflow.com',
            limit: int = 0,
            split: Optional[list] = None,
            seed: int = 2024
    ):
        self.data_name = data_name
        self.limit = limit
        self.split = split
        self.seed = seed

    def run(self):
        elems = []
        for elem in tqdm(self.iterparse(f'./{self.data_name}/{self.data_name}.xml', limit=self.limit), desc=f"Splitting {self.data_name} XML file"):
            elems.append(elem)

        train_elems, temp_elems = train_test_split(elems, test_size=1-self.split[0], random_state=self.seed)
        val_elems, test_elems = train_test_split(temp_elems, test_size=self.split[2]/(1-self.split[0]), random_state=self.seed)

        self.write_file(train_elems, 'Train')
        self.write_file(val_elems, 'Dev')
        self.write_file(test_elems, 'Test')

    def write_file(self, elems: list, mode: str):
        for thread in tqdm(elems, desc=f"Writing {mode} XML file"):
            string = (re.sub(
                r'> *\n +', '>', minidom.parseString(
                    ElementTree.tostring(thread, encoding='utf-8', method='html')
                ).toprettyxml(indent='  ').replace('<?xml version="1.0" ?>\n', '')
            ).replace('\n\n', '\n').replace('\n  \n', '\n').
                      replace('\n    \n', '\n').
                      replace('\n        \n', '\n').
                      replace('\n            \n', '\n').
                      replace('\n                \n', '\n'))

            with open(f"./{self.data_name}/{mode}/{mode.lower()}.xml", 'a', encoding='utf-8') as f:
                f.write(string)

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

    data_name = 'meta.stackoverflow.com'
    limit = 0
    split = [0.8, 0.1, 0.1]

    Split(data_name, limit, split, seed).run()

if __name__ == '__main__':
    main()
