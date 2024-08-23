# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/3/4 16:17
import re
import xml.etree.ElementTree as ElementTree
import xml.dom.minidom as minidom
from time import sleep

from tqdm import tqdm


class Merge:
    def __init__(
            self,
            post_path,
            comment_path,
            save,
    ):
        self.post_path = post_path
        self.comment_path = comment_path
        self.save = save

        # dict to save comment cache
        self.cache = dict()

        with open(self.comment_path, 'r', encoding='utf-8') as f:
            self.comment_root = ElementTree.parse(f).getroot()

    def main(self):
        for elem in tqdm(self.iterparse(self.post_path), desc="Parsing {} XML file".format(self.post_path)):
            # question
            question = elem.find('Question')
            q_id = question.attrib['ID']
            q_comment_count = question.attrib['COMMENT_COUNT']
            if int(q_comment_count) > 0:
                q_comment_elem = self.search(q_id)
                try:
                    q_comment_elem.tag = 'QComment'

                    question.append(q_comment_elem)
                except AttributeError as e:
                    print(e)

            # answer
            for answer in elem.findall('Answer'):
                a_id = answer.attrib['ID']
                a_comment_count = answer.attrib['COMMENT_COUNT']
                if int(a_comment_count) > 0:
                    a_comment_elem = self.search(a_id)
                    try:
                        a_comment_elem.tag = 'AComment'

                        answer.append(a_comment_elem)
                    except AttributeError as e:
                        print(e)

            self.write_file(elem)

    def write_file(self, thread):
        is_writing = True

        string = re.sub(
            r'> *\n +', '>', minidom.parseString(
                ElementTree.tostring(thread, encoding='utf-8', method='html')
            ).toprettyxml(indent='  ').replace('<?xml version="1.0" ?>\n', '')
        ).replace('\n\n', '\n').replace('\n  \n', '\n').replace('\n    \n', '\n')

        while is_writing:
            try:
                with open(self.save, 'a', encoding='utf-8') as f:
                    f.write(string)
                is_writing = False
            except PermissionError:
                sleep(1)
                continue

    @staticmethod
    def iterparse(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            context = ElementTree.iterparse(f, events=('end',))
            _, root = next(context)
            for event, elem in context:
                if elem.tag == 'Thread':
                    yield elem
                    root.clear()

    def search(self, post_id, cache=False):
        if cache:
            if post_id in self.cache:
                elem = self.cache[post_id]
                self.cache.pop(post_id, None)
                return elem
            for elem in self.iterparse(self.comment_path):
                if elem.attrib['POST_ID'] != post_id:
                    self.cache[post_id] = elem
                else:
                    return elem
        else:
            for elem in self.comment_root.findall(".//Thread[@POST_ID='{}']".format(post_id)):
                return elem


def main():
    post_path = 'Outs/meta.stackoverflow.com/Posts.xml'
    comment_path = 'Outs/meta.stackoverflow.com/Comments.xml'
    save = 'Outs/meta.stackoverflow.com/PostsWithComments.xml'

    merge = Merge(
        post_path=post_path,
        comment_path=comment_path,
        save=save
    )
    merge.main()


if __name__ == '__main__':
    main()
