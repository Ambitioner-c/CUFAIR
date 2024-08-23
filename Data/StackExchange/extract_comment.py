# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/3/1 10:31
import traceback
import xml.etree.ElementTree as ElementTree
import xml.dom.minidom as minidom
from xml.parsers.expat import ExpatError
from tqdm import tqdm


class Comment:
    def __init__(
            self,
            post_path,
            comment_path,
            save,

    ):
        self.post_path = post_path
        self.comment_path = comment_path
        self.save = save

        # dict to save num of comments
        self.num_comments = dict()
        # dict to save comments
        self.comments = dict()

    def main(self):
        # get num of comments from Posts.xml
        with open(self.post_path, 'r', encoding='utf-8') as f:
            for _, elem in tqdm(ElementTree.iterparse(f, events=('end',)), desc="Parsing {} XML file".format(self.post_path)):
                if elem.tag == 'row':
                    try:
                        attribs = elem.attrib
                        self.num_comments[attribs['Id']] = attribs['CommentCount']

                        elem.clear()
                    except KeyError:
                        traceback.print_exc()

        with open(self.comment_path, 'r', encoding='utf-8') as f:
            for _, elem in tqdm(ElementTree.iterparse(f, events=('end',)), desc="Parsing {} XML file".format(self.comment_path)):
                if elem.tag == 'row':
                    try:
                        attribs = elem.attrib
                        if self.is_first(attribs):
                            self.comments[attribs['PostId']] = {'ParsedComments': 1, 'Comments': {}}
                        else:
                            self.comments[attribs['PostId']]['ParsedComments'] += 1
                        self.comments[attribs['PostId']]['Comments'][attribs['Id']] = self.trim_attribs(attribs)
                        self.check_complete(attribs)
                        elem.clear()
                    except KeyError:
                        traceback.print_exc()

    def is_first(self, attribs):
        if attribs['PostId'] not in self.comments:
            return True
        return False

    @staticmethod
    def trim_attribs(attribs):
        to_keep = ['Id', 'PostId', 'Score', 'Text', 'CreationDate', 'UserId', 'UserDisplayName']
        to_delete = [x for x in attribs.keys() if x not in to_keep]
        [attribs.pop(x, None) for x in to_delete]

        return attribs

    def check_complete(self, attribs):
        keys_to_del = []

        post_id = attribs['PostId']
        pack = self.comments[post_id]
        if int(pack['ParsedComments']) == int(self.num_comments[post_id]):
            keys_to_del.append(post_id)
            if int(self.num_comments[post_id]) > 0:
                self.write_file(post_id, pack['Comments'])
        for key in keys_to_del:
            self.comments.pop(key, None)
            self.num_comments.pop(key, None)

    def write_file(self, post_id, comments):
        ids = []
        scores = []
        texts = []
        create_dates = []
        user_id_or_display_names = []
        for key in sorted(comments):
            ids.append(comments[key]['Id'])
            scores.append(comments[key]['Score'])
            texts.append(comments[key]['Text'])
            create_dates.append(comments[key]['CreationDate'])
            try:
                user_id_or_display_names.append(comments[key]['UserId'])
            except KeyError:
                try:
                    user_id_or_display_names.append(comments[key]['UserDisplayName'])
                except KeyError:
                    user_id_or_display_names.append('None')

        root = ElementTree.Element('Thread')
        root.set('POST_ID', post_id)
        for j in range(len(scores)):
            comment = ElementTree.SubElement(root, 'Comment')
            comment.set('ID', ids[j])
            comment.set('SCORE', scores[j])
            comment.set('CREATION_DATE', create_dates[j])
            comment.set('USER_ID_OR_DISPLAY_NAME', user_id_or_display_names[j])
            comment_body = ElementTree.SubElement(comment, 'CBody')
            comment_body.text = texts[j]

        with open(self.save, 'a', encoding='utf-8') as f:
            try:
                f.write(
                    minidom.parseString(
                        ElementTree.tostring(root, encoding='utf-8', method='html')
                    ).toprettyxml(indent='  ').replace('<?xml version="1.0" ?>\n', '')
                )
            except ExpatError:
                traceback.print_exc()


def main():
    post_path = 'Dumps/meta.stackoverflow.com/Posts.xml'
    comment_path = 'Dumps/meta.stackoverflow.com/Comments.xml'
    save = 'Outs/meta.stackoverflow.com/Comments.xml'

    comment = Comment(
        post_path=post_path,
        comment_path=comment_path,
        save=save
    )
    comment.main()


if __name__ == '__main__':
    main()
