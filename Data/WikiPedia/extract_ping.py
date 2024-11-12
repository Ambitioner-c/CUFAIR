# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/11/12 18:36
import xml.etree.ElementTree as ElementTree

from tqdm import tqdm
from transformers import set_seed


class Ping:
    def __init__(
            self,
            xml_path,
    ):
        self.xml_path = xml_path

    def main(self):
        for elem in tqdm(self.iterparse(self.xml_path), desc="Parsing {} XML file".format(self.xml_path)):
            participants = []
            pings = []

            nomination = elem.find('Nomination')
            n_id = nomination.attrib['ID']
            participants.append(n_id)

            comments = elem.find('Comments')
            if int(comments.attrib['CommentCount']):
                for comment in comments.findall('Comment'):
                    c_id = comment.attrib['ID']
                    participants.append(c_id)

                    c_reply_to = comment.attrib['ReplyTo']
                    try:
                        pings.append(participants.index(c_reply_to))
                    except ValueError:
                        pings.append(0)

    @staticmethod
    def iterparse(filename: str):
        tree = ElementTree.parse(filename)
        root = tree.getroot()

        for thread in root.findall('Thread'):
            yield thread


def main():
    set_seed(2024)

    data_dir = '/home/cuifulai/Projects/CQA/Data/WikiPedia'
    data_name = 'wiki-articles-for-deletion-corpus'

    xml_path = f'{data_dir}/{data_name}/AfD.xml'

    ping = Ping(
        xml_path=xml_path,
    )
    ping.main()


if __name__ == '__main__':
    main()
