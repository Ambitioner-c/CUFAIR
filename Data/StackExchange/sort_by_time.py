# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/3/11 16:45
import re
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

from datetime import datetime
from time import sleep

from tqdm import tqdm


class SortByTime:
    def __init__(
            self,
            post_with_comment,
            save,
    ):
        self.post_with_comment = post_with_comment
        self.save = save

    def main(self):
        for elem in tqdm(self.iterparse(self.post_with_comment), desc="Parsing {} XML file".format(self.post_with_comment)):
            # question
            question = elem.find('Question')
            # check question-comment
            if int(question.attrib['COMMENT_COUNT']) and self.is_chaotic(question):
                q_comment = question.find('QComment')
                try:
                    sorted_comments = self.reorder(q_comment.findall('Comment'))
                except AttributeError as e:
                    print(e)
                    continue

                # save POST_ID
                q_comment_post_id = q_comment.attrib['POST_ID']

                q_comment.clear()
                q_comment.set('POST_ID', q_comment_post_id)

                # insert question' comments
                for comment in sorted_comments:
                    q_comment.append(comment)

            # check question-answer
            if int(question.attrib['ANSWER_COUNT']) and self.is_chaotic(elem):
                sorted_answers = self.reorder(elem.findall('Answer'))

                # save ID
                thread_id = elem.attrib['ID']

                elem.clear()
                elem.set('ID', thread_id)

                # insert question and answers
                elem.append(question)
                for answer in sorted_answers:
                    elem.append(answer)

            # answer
            for answer in elem.findall('Answer'):
                # check answer-comment
                if int(answer.attrib['COMMENT_COUNT']) and self.is_chaotic(answer):
                    a_comment = answer.find('AComment')
                    try:
                        sorted_comments = self.reorder(a_comment.findall('Comment'))
                    except AttributeError as e:
                        print(e)
                        continue

                    # save POST_ID
                    a_comment_post_id = a_comment.attrib['POST_ID']

                    a_comment.clear()
                    a_comment.set('POST_ID', a_comment_post_id)

                    # insert answer' comments
                    for comment in sorted_comments:
                        a_comment.append(comment)
            self.write_file(elem)

    def write_file(self, thread):
        is_writing = True

        string = minidom.parseString(
            re.sub(r'> *\n *<', '><', ET.tostring(thread, encoding='utf-8', method='html').decode())
        ).toprettyxml(indent='    ').replace('<?xml version="1.0" ?>\n', '')

        while is_writing:
            try:
                with open(self.save, 'a', encoding='utf-8') as f:
                    f.write(string)
                is_writing = False
            except PermissionError:
                sleep(1)
                continue

    @staticmethod
    def reorder(elems):
        sorted_elems = sorted(
            elems,
            key=lambda x: datetime.strptime(x.attrib['CREATION_DATE'], "%Y-%m-%dT%H:%M:%S.%f")
        )

        return sorted_elems

    @staticmethod
    def is_chaotic(elem):
        if elem.tag == 'Thread':
            date_objs = []
            for answer in elem.findall('Answer'):
                date_objs.append(datetime.strptime(answer.attrib['CREATION_DATE'], "%Y-%m-%dT%H:%M:%S.%f"))

            is_sorted = all(date_objs[i] <= date_objs[i+1] for i in range(len(date_objs)-1))
            if is_sorted:
                return False
            else:
                return True
        elif elem.tag == 'Question':
            date_objs = []
            try:
                for comment in elem.find('QComment').findall('Comment'):
                    date_objs.append(datetime.strptime(comment.attrib['CREATION_DATE'], "%Y-%m-%dT%H:%M:%S.%f"))
            except AttributeError as e:
                print(e)
                return True

            is_sorted = all(date_objs[i] <= date_objs[i + 1] for i in range(len(date_objs) - 1))
            if is_sorted:
                return False
            else:
                return True
        elif elem.tag == 'Answer':
            date_objs = []
            try:
                for comment in elem.find('AComment').findall('Comment'):
                    date_objs.append(datetime.strptime(comment.attrib['CREATION_DATE'], "%Y-%m-%dT%H:%M:%S.%f"))
            except AttributeError as e:
                print(e)
                return True

            is_sorted = all(date_objs[i] <= date_objs[i + 1] for i in range(len(date_objs) - 1))
            if is_sorted:
                return False
            else:
                return True

    @staticmethod
    def iterparse(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            context = ET.iterparse(f, events=('end',))
            _, root = next(context)
            for event, elem in context:
                if elem.tag == 'Thread':
                    yield elem
                    root.clear()


def main():
    post_with_comment = 'Outs/meta.stackoverflow.com/PostsWithComments.xml'

    save = 'Outs/meta.stackoverflow.com/SortedPostsWithComments.xml'

    sort_by_time = SortByTime(
        post_with_comment=post_with_comment,
        save=save
    )
    sort_by_time.main()


if __name__ == '__main__':
    main()
