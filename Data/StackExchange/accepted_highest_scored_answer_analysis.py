# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/6/24 15:34
import xml.etree.ElementTree as ElementTree
from tqdm import tqdm
import re


class Distribution:
    def __init__(self, xml_path):
        self.xml_path = xml_path

        self.question_post_time = {}
        self.accepted_answer_post_time = {}
        self.highest_scored_answer_post_time = {}

        self.num_of_questions = 0
        self.num_of_questions_without_accepted_answer = 0

        self.num_of_accepted_answers = 0
        self.num_of_accepted_answers_with_the_highest_scores = 0
        self.num_of_answers_for_questions_with_accepted_answer = 0
        self.num_of_answers_for_questions_without_accepted_answer = 0
        self.num_of_answers_with_comments_more_than_2 = 0

        self.num_of_comments_on_questions_with_accepted_answer = 0
        self.num_of_comments_on_questions_without_accepted_answer = 0
        self.num_of_comments_on_accepted_answers = 0
        self.num_of_comments_on_un_accepted_answers = 0

    def main(self):
        for elem in tqdm(self.iterparse(self.xml_path), desc="Parsing {} XML file".format(self.xml_path)):
            self.num_of_questions += 1

            question = elem.find('Question')
            question_creation_date = re.findall(r'\d{4}-\d{2}-\d{2}', question.attrib['CREATION_DATE'])[0]

            highest_score = -1e5
            accepted_answer_id = ''
            accepted_answer_creation_date = ''
            highest_scored_answer_id = ''
            highest_scored_answer_creation_date = ''
            exist_accepted_answer = False
            temp_num_of_answers_for_question = 0
            temp_num_of_comments_for_question = 0
            for answer in elem.findall('Answer'):
                temp_num_of_answers_for_question += 1

                temp_num_of_comments_for_answer = 0
                if int(answer.attrib['COMMENT_COUNT']):
                    for _ in answer.find('AComment').findall('Comment'):
                        temp_num_of_comments_for_question += 1
                        temp_num_of_comments_for_answer += 1

                answer_creation_date = re.findall(r'\d{4}-\d{2}-\d{2}', answer.attrib['CREATION_DATE'])[0]
                answer_id = answer.attrib['ID']

                answer_score = int(answer.attrib['SCORE'])
                if answer_score > highest_score:
                    highest_score = answer_score
                    highest_scored_answer_id = answer_id
                    highest_scored_answer_creation_date = answer_creation_date

                if answer.attrib['ACCEPTED_ANSWER'] == 'Yes':
                    self.num_of_accepted_answers += 1
                    self.num_of_comments_on_accepted_answers += temp_num_of_comments_for_answer

                    exist_accepted_answer = True

                    accepted_answer_id = answer_id
                    accepted_answer_creation_date = answer_creation_date
                else:
                    self.num_of_comments_on_un_accepted_answers += temp_num_of_comments_for_answer

                if temp_num_of_comments_for_answer > 2:
                    self.num_of_answers_with_comments_more_than_2 += 1

            if exist_accepted_answer is False:
                self.num_of_questions_without_accepted_answer += 1
                self.num_of_answers_for_questions_without_accepted_answer += temp_num_of_answers_for_question
                self.num_of_comments_on_questions_without_accepted_answer += temp_num_of_comments_for_question
            else:
                self.num_of_answers_for_questions_with_accepted_answer += temp_num_of_answers_for_question
                self.num_of_comments_on_questions_with_accepted_answer += temp_num_of_comments_for_question

            if accepted_answer_creation_date is not '' and highest_score > 0:
                if accepted_answer_id == highest_scored_answer_id:
                    self.num_of_accepted_answers_with_the_highest_scores += 1
                self.question_post_time[question_creation_date] = self.question_post_time.get(question_creation_date, 0) + 1
                self.accepted_answer_post_time[accepted_answer_creation_date] = self.accepted_answer_post_time.get(accepted_answer_creation_date, 0) + 1
                self.highest_scored_answer_post_time[highest_scored_answer_creation_date] = self.highest_scored_answer_post_time.get(highest_scored_answer_creation_date, 0) + 1

        all_times = set(self.question_post_time.keys()) | set(self.accepted_answer_post_time.keys()) | set(self.highest_scored_answer_post_time.keys())
        for time in all_times:
            self.question_post_time[time] = self.question_post_time.get(time, 0)
            self.accepted_answer_post_time[time] = self.accepted_answer_post_time.get(time, 0)
            self.highest_scored_answer_post_time[time] = self.highest_scored_answer_post_time.get(time, 0)

        self.question_post_time = dict(sorted(self.question_post_time.items(), key=lambda x: x[0]))
        self.accepted_answer_post_time = dict(sorted(self.accepted_answer_post_time.items(), key=lambda x: x[0]))
        self.highest_scored_answer_post_time = dict(sorted(self.highest_scored_answer_post_time.items(), key=lambda x: x[0]))

    @staticmethod
    def iterparse(filename):
        context = ElementTree.iterparse(filename, events=('end', ))
        _, root = next(context)
        for event, elem in context:
            if elem.tag == 'Thread':
                yield elem
                root.clear()

    def statistics(self):
        self.cprint('#Questions:', self.num_of_questions)
        self.cprint('#Questions without accepted answer:', self.num_of_questions_without_accepted_answer)
        self.cprint('#Accepted answers:', self.num_of_accepted_answers)
        self.cprint('#Accepted Answers with the Highest Scores:', self.num_of_accepted_answers_with_the_highest_scores)
        self.cprint('#Answers for questions with accepted answer:', self.num_of_answers_for_questions_with_accepted_answer)
        self.cprint('#Answers for questions without accepted answer:', self.num_of_answers_for_questions_without_accepted_answer)
        self.cprint('#Answers with comments more than 2:', self.num_of_answers_with_comments_more_than_2)
        self.cprint('#Comments on questions with accepted answer:', self.num_of_comments_on_questions_with_accepted_answer)
        self.cprint('#Comments on questions without accepted answer:', self.num_of_comments_on_questions_without_accepted_answer)
        self.cprint('#Comments on accepted answers:', self.num_of_comments_on_accepted_answers)
        self.cprint('#Comments on un-accepted answers:', self.num_of_comments_on_un_accepted_answers)

    @staticmethod
    def cprint(comment, content):
        print('\033[33m' + comment + '\033[0m', content)


def main():
    xml_path = 'Outs/meta.stackoverflow.com/SortedPostsWithCommentsWithUserIDAndDisplayName.xml'

    distribution = Distribution(xml_path)
    distribution.main()

    # for time, num in distribution.question_post_time.items():
    #     print(time, num)
    # for time, num in distribution.highest_scored_answer_post_time.items():
    #     print(time, num)
    # for time, num in distribution.accepted_answer_post_time.items():
    #     print(time, num)

    distribution.statistics()


if __name__ == '__main__':
    main()
