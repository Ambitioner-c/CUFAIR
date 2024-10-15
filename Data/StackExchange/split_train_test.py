# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/16 17:11
"""
Writing train_1_fold XML file: 100%|█████████| 832/832 [00:03<00:00, 223.36it/s]
Writing test_1_fold XML file: 100%|████████████| 93/93 [00:00<00:00, 227.16it/s]
Writing train_2_fold XML file: 100%|█████████| 832/832 [00:03<00:00, 209.12it/s]
Writing test_2_fold XML file: 100%|████████████| 93/93 [00:00<00:00, 208.01it/s]
Writing train_3_fold XML file: 100%|█████████| 832/832 [00:04<00:00, 206.89it/s]
Writing test_3_fold XML file: 100%|████████████| 93/93 [00:00<00:00, 212.21it/s]
Writing train_4_fold XML file: 100%|█████████| 832/832 [00:04<00:00, 202.86it/s]
Writing test_4_fold XML file: 100%|████████████| 93/93 [00:00<00:00, 222.16it/s]
Writing train_5_fold XML file: 100%|█████████| 832/832 [00:04<00:00, 203.40it/s]
Writing test_5_fold XML file: 100%|████████████| 93/93 [00:00<00:00, 227.22it/s]
Writing train_6_fold XML file: 100%|█████████| 833/833 [00:03<00:00, 210.08it/s]
Writing test_6_fold XML file: 100%|████████████| 92/92 [00:00<00:00, 210.13it/s]
Writing train_7_fold XML file: 100%|█████████| 833/833 [00:03<00:00, 212.22it/s]
Writing test_7_fold XML file: 100%|████████████| 92/92 [00:00<00:00, 156.05it/s]
Writing train_8_fold XML file: 100%|█████████| 833/833 [00:04<00:00, 206.35it/s]
Writing test_8_fold XML file: 100%|████████████| 92/92 [00:00<00:00, 249.40it/s]
Writing train_9_fold XML file: 100%|█████████| 833/833 [00:03<00:00, 210.95it/s]
Writing test_9_fold XML file: 100%|████████████| 92/92 [00:00<00:00, 237.41it/s]
Writing train_10_fold XML file: 100%|████████| 833/833 [00:04<00:00, 203.98it/s]
Writing test_10_fold XML file: 100%|███████████| 92/92 [00:00<00:00, 232.51it/s]
"""
import re
import xml.etree.ElementTree as ElementTree
import xml.dom.minidom as minidom

from sklearn.model_selection import KFold
from tqdm import tqdm


class Split:
    def __init__(
            self,
            data_name: str = 'meta.stackoverflow.com',
            threshold: int = 5,
            limit: int = 0,
            kf: KFold = None,
            seed: int = 2024
    ):
        self.data_name = data_name
        self.threshold = threshold
        self.limit = limit
        self.kf = kf
        self.seed = seed

    def run(self):
        elems = []
        for elem in tqdm(self.iterparse(f'./{self.data_name}/{self.data_name}.xml', limit=self.limit), desc=f"Splitting {self.data_name} XML file"):
            # Answers
            answers = elem.findall('Answer')
            if len(answers) >= self.threshold:
                elems.append(elem)

        for i, (train_index, test_index) in enumerate(self.kf.split(elems)):
            train_elems = [elems[j] for j in train_index]
            test_elems = [elems[j] for j in test_index]

            self.write_file(train_elems, 'Train', i)
            self.write_file(test_elems, 'Test', i)

    def write_file(self, elems: list, mode: str, fold: int):
        for thread in tqdm(elems, desc=f"Writing {mode.lower()}_{fold+1}_fold XML file"):
            string = (re.sub(
                r'> *\n +', '>', minidom.parseString(
                    ElementTree.tostring(thread, encoding='utf-8', method='html')
                ).toprettyxml(indent='  ').replace('<?xml version="1.0" ?>\n', '')
            ).replace('\n\n', '\n').replace('\n  \n', '\n').
                      replace('\n    \n', '\n').
                      replace('\n        \n', '\n').
                      replace('\n            \n', '\n').
                      replace('\n                \n', '\n'))

            with open(f"./{self.data_name}/{mode}/{mode.lower()}_{fold+1}_fold.xml", 'a', encoding='utf-8') as f:
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
    threshold = 5

    n_splits = 10
    shuffle = True

    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    Split(data_name, threshold, limit, kf, seed).run()

if __name__ == '__main__':
    main()
