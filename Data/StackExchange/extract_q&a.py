# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/2/27 10:12
import traceback
import xml.etree.ElementTree as ElementTree
import xml.dom.minidom as minidom
import xml.parsers.expat

from bs4 import BeautifulSoup
from tqdm import tqdm


class QAPair:
    def __init__(
            self,
            xml_path,
            name=None,
            save="Post.xml",
            min_score=0,
            max_responses=int(1e5)
    ):
        """
        Makes a text dataset from StackExchange dumps.
        :param xml_path:
        :param name: name of StackExchanges to parse. e.g. meta.stackoverflow.
        :param save
        :param min_score: minimum score of a response in order to be included in the dataset. Default 3.
        :param max_responses: maximum number of responses (sorted by score) to include for each question. Default 3.
        """

        self.xml_path = xml_path
        self.name = name
        self.save = save
        self.min_score = min_score
        self.max_responses = max_responses

        # dict to save questions
        self.questions = dict()

    def main(self):
        """
        iterates through SE xml and:

        - stores PostTypeId="1" with AcceptedAnswerIds / Answers.
        - when an AcceptedAnswerId or Answer > min_score is reached, it should:
            > concat the Question & Accepted answer
            > Clean markup / HTML
            > Output to txt file
            > Delete from memory
        :return:
        """
        with open(self.xml_path, 'r', encoding='utf-8') as f:
            for event, elem in tqdm(ElementTree.iterparse(f, events=('end',)), desc="Parsing {} XML file".format(self.name)):
                if elem.tag == "row":
                    try:
                        attribs = elem.attrib
                        if self.is_question(attribs):
                            if self.has_answers(attribs):
                                self.trim_attribs(attribs, "question")
                                self.questions[attribs["Id"]] = attribs
                            else:
                                # if the question has no answers, discard it
                                continue
                        elif self.is_answer(attribs):
                            # if is accepted answer, append answer Body to relevant questions "AcceptedAnswer" field
                            # if the answer's score > min_score
                            # append the answer to the relevant question's OtherAnswers dict
                            self.add_answer(attribs)
                            self.check_complete(attribs)
                        elem.clear()
                    except KeyError:
                        traceback.print_exc()

    @staticmethod
    def is_question(elem_attribs):
        if elem_attribs["PostTypeId"] is not None:
            if elem_attribs["PostTypeId"] == "1":
                return True
        return False

    @staticmethod
    def is_answer(elem_attribs):
        if elem_attribs["PostTypeId"] is not None:
            if elem_attribs["PostTypeId"] == "2":
                return True
        return False

    @staticmethod
    def has_answers(elem_attribs):
        if elem_attribs["AnswerCount"] is not None:
            if int(elem_attribs["AnswerCount"]):
                return True
        return False

    @staticmethod
    def trim_attribs(elem_attribs, attrib_type="question"):
        """
        deletes non-useful data from attribs dict for questions / answers, returns remaining
        :param elem_attribs:
        :param attrib_type:
        :return:
        """
        if attrib_type == "question":
            to_keep = ['Id', 'PostTypeId', 'AcceptedAnswerId', 'CreationDate', 'OwnerUserId', 'Body',
                       'Title', 'AnswerCount', 'CommentCount']
        elif attrib_type == "answer":
            to_keep = ['Id', 'PostTypeId', 'ParentId', 'CreationDate', 'Score', 'Body',
                       'OwnerUserId', 'CommentCount']
        else:
            raise Exception('Unrecognized attribute type - please specify either question or answer')

        to_delete = [x for x in elem_attribs.keys() if x not in to_keep]
        [elem_attribs.pop(x, None) for x in to_delete]

        if attrib_type == "question":
            elem_attribs["ParsedAnswers"] = 0
            elem_attribs["AcceptedAnswer"] = {}
            elem_attribs["OtherAnswers"] = {}
        elif attrib_type == "answer":
            return elem_attribs

    def add_answer(self, a_attribs):
        """
        Adds answer to its parent question in [self.questions] if it's either an accepted answer or above self.min_score.
         If answer is an accepted answer, it gets appended to the AcceptedAnswer field, otherwise it gets appended to OtherAnswers.
          Also increments the question's 'ParsedAnswers' field.
           When ParsedAnswers = AnswerCount, the question is deleted from memory and saved to a text file.

        :param a_attribs: Answer's attribute dict
        :return:
        """

        try:
            self.questions[a_attribs["ParentId"]]
        except KeyError as e:
            print(e)
            return

        if a_attribs is not None and self.questions[a_attribs["ParentId"]] is not None:
            if self.is_accepted_answer(a_attribs, self.questions[a_attribs["ParentId"]]):
                self.questions[a_attribs["ParentId"]]["AcceptedAnswer"][a_attribs["Id"]] = self.trim_attribs(a_attribs, "answer")
                self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1
            elif self.is_above_threshold(a_attribs):
                if a_attribs["Id"] is not None:
                    parent = self.questions[a_attribs["ParentId"]]
                    if parent is not None:
                        self.questions[a_attribs["ParentId"]]["OtherAnswers"][a_attribs["Id"]] = self.trim_attribs(a_attribs, "answer")
                        self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1
                else:
                    self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1
            else:
                self.questions[a_attribs["ParentId"]]["ParsedAnswers"] += 1

    @staticmethod
    def is_accepted_answer(a_attribs, q_attribs):
        try:
            if q_attribs["AcceptedAnswerId"] is not None:
                if q_attribs["AcceptedAnswerId"] == a_attribs["Id"]:
                    return True
            else:
                return False
        except KeyError:
            return False

    def is_above_threshold(self, a_attribs):
        """
        Determines whether an answer is above the min_score threshold
        :param a_attribs: Answer's attribute dict
        :return:
        """
        if a_attribs["Score"] is not None:
            if int(a_attribs["Score"]) >= self.min_score:
                return True
        return False

    def check_complete(self, a_attribs):
        """
        checks if the parent question of the previously added answer has no future answers, and if so, removes from dict and prints to file.
        :param a_attribs:
        :return:
        """
        keys_to_del = []
        try:
            parent = self.questions[a_attribs["ParentId"]]
        except KeyError as e:
            print(e)
            return

        if a_attribs is not None and parent is not None:
            if parent["AnswerCount"] is not None and parent["ParsedAnswers"] is not None:
                if int(parent["ParsedAnswers"]) == int(parent['AnswerCount']):
                    keys_to_del.append(a_attribs["ParentId"])
                    if len(parent["AcceptedAnswer"]) > 0 or len(parent["OtherAnswers"]) > 0:
                        # print(parent)
                        self.write_file(parent)
        for key in keys_to_del:
            self.questions.pop(key, None)

    def write_file(self, parent):
        # question
        q_id = parent['Id']
        q_post_type_id = parent['PostTypeId']
        q_creation_date = parent['CreationDate']
        q_body = BeautifulSoup(parent['Body'], "html.parser").get_text().replace('\n', '')
        try:
            q_owner_user_id = parent['OwnerUserId']
        except KeyError:
            return
        q_title = parent['Title']
        answer_count = parent['AnswerCount']
        q_comment_count = parent['CommentCount']

        # answers
        a_ids = []
        a_post_type_ids = []
        a_parent_ids = []
        a_creation_dates = []
        a_scores = []
        a_bodys = []
        a_owner_user_ids = []
        a_comment_counts = []
        a_accepted_answers = []
        # accepted answer
        if len(parent['AcceptedAnswer']) > 0:
            accepted_answer = parent['AcceptedAnswer'][parent['AcceptedAnswerId']]
            a_ids.append(accepted_answer['Id'])
            a_post_type_ids.append(accepted_answer['PostTypeId'])
            a_parent_ids.append(accepted_answer['ParentId'])
            a_creation_dates.append(accepted_answer['CreationDate'])
            a_scores.append(accepted_answer['Score'])
            a_bodys.append(BeautifulSoup(accepted_answer['Body'], "html.parser").get_text().replace('\n', ''))
            try:
                a_owner_user_ids.append(accepted_answer['OwnerUserId'])
            except KeyError:
                return
            a_comment_counts.append(accepted_answer['CommentCount'])
            a_accepted_answers.append('Yes')
        # other_answers
        if len(parent['OtherAnswers']) > 0:
            for j in parent['OtherAnswers']:
                other_answer = parent['OtherAnswers'][j]
                a_ids.append(other_answer['Id'])
                a_post_type_ids.append(other_answer['PostTypeId'])
                a_parent_ids.append(other_answer['ParentId'])
                a_creation_dates.append(other_answer['CreationDate'])
                a_scores.append(other_answer['Score'])
                a_bodys.append(BeautifulSoup(other_answer['Body'], "html.parser").get_text().replace('\n', ''))
                try:
                    a_owner_user_ids.append(other_answer['OwnerUserId'])
                except KeyError:
                    return
                a_comment_counts.append(other_answer['CommentCount'])
                a_accepted_answers.append('No')

        root = ElementTree.Element('Thread')
        root.set('ID', q_id)

        # question
        question = ElementTree.SubElement(root, 'Question')
        question.set('ID', q_id)
        question.set('POST_TYPE_ID', q_post_type_id)
        question.set('CREATION_DATE', q_creation_date)
        question.set('OWNER_USER_ID', q_owner_user_id)
        question.set('ANSWER_COUNT', answer_count)
        question.set('COMMENT_COUNT', q_comment_count)
        question_title = ElementTree.SubElement(question, 'QTitle')
        question_title.text = q_title
        question_body = ElementTree.SubElement(question, 'QBody')
        question_body.text = q_body

        # answer
        for j in range(len(a_ids)):
            answer = ElementTree.SubElement(root, 'Answer')
            answer.set('ID', a_ids[j])
            answer.set('POST_TYPE_ID', a_post_type_ids[j])
            answer.set('PARENT_ID', a_parent_ids[j])
            answer.set('CREATION_DATE', a_creation_dates[j])
            answer.set('SCORE', a_scores[j])
            answer.set('OWNER_USER_ID', a_owner_user_ids[j])
            answer.set('COMMENT_COUNT', a_comment_counts[j])
            answer.set('ACCEPTED_ANSWER', a_accepted_answers[j])
            answer_body = ElementTree.SubElement(answer, 'ABody')
            answer_body.text = a_bodys[j]

        with open(self.save, 'a', encoding='utf-8') as f:
            try:
                f.write(
                    minidom.parseString(
                        ElementTree.tostring(root, encoding='utf-8', method='html')
                    ).toprettyxml(indent='  ').replace('<?xml version="1.0" ?>\n', '')
                )
            except xml.parsers.expat.ExpatError as e:
                print(e)


def main():
    xml_path = 'Dumps/meta.stackoverflow.com/Posts.xml'
    name = 'meta.stackoverflow.com'
    save = 'Outs/meta.stackoverflow.com/Posts.xml'

    qa_pair = QAPair(
        xml_path=xml_path,
        name=name,
        save=save
    )
    qa_pair.main()


if __name__ == '__main__':
    main()
