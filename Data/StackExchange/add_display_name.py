# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/3/12 9:34
import re
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

from time import sleep

from tqdm import tqdm


class DisplayName:
    def __init__(
            self,
            sorted_post_with_comment_path,
            user_path,
            save,
    ):
        self.sorted_post_with_comment_path = sorted_post_with_comment_path
        self.user_path = user_path
        self.save = save

        # dict to save user
        self.users = dict()

    def main(self):
        # user
        for event, elem in tqdm(ET.iterparse(self.user_path, events=('end',)), desc="Parsing {} XML file".format(self.user_path)):
            if elem.tag == "row":
                self.users[int(elem.attrib['Id'])] = elem.attrib['DisplayName']

        for elem in tqdm(self.iterparse(self.sorted_post_with_comment_path), desc="Parsing {} XML file".format(self.sorted_post_with_comment_path)):
            # question
            question = elem.find('Question')
            question.set('OWNER_DISPLAY_NAME', self.users[int(question.attrib['OWNER_USER_ID'])])

            # question comment
            if int(question.attrib['COMMENT_COUNT']):
                q_comment = question.find('QComment')
                self.parse_user_id_and_display_name(q_comment.findall('Comment'))

            # answer
            if int(question.attrib['ANSWER_COUNT']):
                for answer in elem.findall('Answer'):
                    answer.set('OWNER_DISPLAY_NAME', self.users[int(answer.attrib['OWNER_USER_ID'])])

                    # answer comment
                    if int(answer.attrib['COMMENT_COUNT']):
                        a_comment = answer.find('AComment')
                        try:
                            self.parse_user_id_and_display_name(a_comment.findall('Comment'))
                        except AttributeError as e:
                            print(e)

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

    def parse_user_id_and_display_name(self, elems):
        for comment in elems:
            user_id_or_display_name = comment.attrib['USER_ID_OR_DISPLAY_NAME']
            if 'user' in user_id_or_display_name:
                user_id = user_id_or_display_name.replace('user', '')
                display_name = user_id_or_display_name
            else:
                try:
                    display_name = self.users[int(user_id_or_display_name)]
                    user_id = user_id_or_display_name
                except ValueError:
                    display_name = user_id_or_display_name
                    user_id = '0'

            # del USER_ID_OR_DISPLAY_NAME
            del comment.attrib['USER_ID_OR_DISPLAY_NAME']

            # add user_id and display_name
            comment.set('USER_ID', user_id)
            comment.set('DISPLAY_NAME', display_name)

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
    sorted_post_with_comment_path = 'Outs/meta.stackoverflow.com/SortedPostsWithComments.xml'
    user_path = 'Dumps/meta.stackoverflow.com/Users.xml'
    save = 'Outs/meta.stackoverflow.com/SortedPostsWithCommentsWithUserIDAndDisplayName.xml'

    display_name = DisplayName(
        sorted_post_with_comment_path=sorted_post_with_comment_path,
        user_path=user_path,
        save=save
    )
    display_name.main()


if __name__ == '__main__':
    main()
