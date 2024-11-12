# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/11/12 19:28
from typing import Optional
import xml.etree.ElementTree as ElementTree
from xml.dom import minidom

from sklearn.model_selection import train_test_split
from tqdm import tqdm


class Split:
    def __init__(
            self,
            xml_path: str,
            split: Optional[list] = None,
            seed: int = 2024,
            save: Optional[str] = None,
    ):
        self.xml_path = xml_path
        self.split = split
        self.seed = seed
        self.save = save

    def run(self):
        elems = []
        for elem in tqdm(self.iterparse(self.xml_path), desc="Parsing {} XML file".format(self.xml_path)):
            elems.append(elem)

        train_elems, temp_elems = train_test_split(elems, test_size=1 - self.split[0], random_state=self.seed)
        val_elems, test_elems = train_test_split(temp_elems, test_size=0.5, random_state=self.seed)

        self.generate_xml(train_elems, 'Train')
        self.generate_xml(val_elems, 'Dev')
        self.generate_xml(test_elems, 'Test')

    def generate_xml(self, elems: list, mode: str):
        root = ElementTree.Element('AfD')
        for thread in tqdm(elems):
            root.append(thread)

        xml_str = ElementTree.tostring(root, encoding='utf-8', method='html')
        parsed_str = minidom.parseString(xml_str)
        pretty_str = parsed_str.toprettyxml(indent="\t")
        if self.save:
            with open(f'{self.save}/{mode}/{mode.lower()}.xml', 'w', encoding='utf-8') as f:
                f.write(pretty_str)

    @staticmethod
    def iterparse(filename: str):
        tree = ElementTree.parse(filename)
        root = tree.getroot()

        for thread in root.findall('Thread'):
            yield thread


def main():
    seed = 2024

    data_dir = '/home/cuifulai/Projects/CQA/Data/WikiPedia'
    data_name = 'AfD'

    xml_path = f'{data_dir}/{data_name}/{data_name}.xml'

    split = [0.8, 0.1, 0.1]

    save_dir = f'{data_dir}/{data_name}'

    Split(
        xml_path,
        split=split,
        seed=seed,
        save=save_dir,
    ).run()


if __name__ == '__main__':
    main()
