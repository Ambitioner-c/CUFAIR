# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/3/13 15:19
import re
import xml.etree.ElementTree as ET

from tqdm import tqdm

# from rapidfuzz import process


class PingMatch:
    def __init__(
            self,
            xml_path
    ):
        self.xml_path = xml_path

        self.num_of_questions = 0
        self.num_of_questions_with_answers = 0
        self.num_of_questions_with_comments = 0
        self.num_of_questions_with_interactions = 0

        self.num_of_answers = 0
        self.num_of_answers_with_comments = 0
        self.num_of_answers_with_interactions = 0

        self.num_of_comments = 0
        self.num_of_comments_with_interactions = 0

        self.num_of_interactions = 0

        self.statistic = {}

    def main(self):
        for elem in tqdm(self.iterparse(self.xml_path), desc="Parsing {} XML file".format(self.xml_path)):
        # for elem in self.iterparse(self.xml_path):
            self.num_of_questions += 1

            question_with_answers = 0
            question_with_comments = 0
            question_with_interactions = 0

            # answer
            for answer in elem.findall('Answer'):
                self.num_of_answers += 1
                question_with_answers = 1

                answer_with_comments = 0
                answer_with_interactions = 0

                if int(answer.attrib['COMMENT_COUNT']):
                    a_name = answer.attrib['OWNER_DISPLAY_NAME']

                    participants = list()
                    participants_dict = dict()
                    pings = list()

                    participants.append(a_name)
                    participants_dict[a_name] = 0

                    # Comments
                    for c_idx, comment in enumerate(answer.find('AComment').findall('Comment')):
                        c_name = comment.attrib['DISPLAY_NAME']
                        c_body = comment.find('CBody').text

                        self.num_of_comments += 1
                        question_with_comments = 1
                        answer_with_comments = 1

                        comment_with_interactions = 0

                        if len(participants_dict) == 1:
                            if c_name == a_name:
                                pings.append(0)
                            else:
                                if participants_dict[a_name] != 0:
                                    self.num_of_interactions += 1
                                    question_with_interactions = 1
                                    answer_with_interactions = 1
                                    comment_with_interactions = 1

                                pings.append(participants_dict[a_name])
                        elif len(participants_dict) == 2 and c_name in participants_dict:
                            if c_name == a_name:
                                self.num_of_interactions += 1
                                question_with_interactions = 1
                                answer_with_interactions = 1
                                comment_with_interactions = 1

                                pings.append(participants_dict[[x for x in participants_dict if x != a_name][0]])
                            else:
                                if participants_dict[a_name] != 0:
                                    self.num_of_interactions += 1
                                    question_with_interactions = 1
                                    answer_with_interactions = 1
                                    comment_with_interactions = 1

                                pings.append(participants_dict[a_name])
                        else:
                            index = self.interaction(participants, self.parse_ping(c_body))
                            pings.append(index)

                            if index != 0:
                                self.num_of_interactions += 1
                                question_with_interactions = 1
                                answer_with_interactions = 1
                                comment_with_interactions = 1

                        participants.append(c_name)
                        participants_dict[c_name] = c_idx + 1
                        self.num_of_comments_with_interactions += comment_with_interactions
                    # print(participants, pings)
                    # self.analytic_ping(pings)
                    # self.statistic_ping(pings)
                self.num_of_answers_with_comments += answer_with_comments
                self.num_of_answers_with_interactions += answer_with_interactions
            self.num_of_questions_with_answers += question_with_answers
            self.num_of_questions_with_comments += question_with_comments
            self.num_of_questions_with_interactions += question_with_interactions

    def interaction(self, participants, names):
        if names:
            # # fuzzy matching
            # best_match = process.extractOne(names[-1], participants)

            # rule-based matching
            best_match = self.matching(names[-1], participants)

            score = best_match[1]
            index = best_match[2]

            # 90 for fuzzy matching
            if score > 90:
                return index
            else:
                return 0
        else:
            return 0

    @staticmethod
    def matching(query, choices):
        temp = None, 0.0, None

        for index, choice in enumerate(choices):
            if len(query) < 3:
                if query.lower() == choice.split(' ')[0].lower():
                    temp = choice, 100.0, index
            else:
                if choice.lower().replace(' ', '').startswith(query.lower()):
                    temp = choice, 100.0, index

        return temp

    @staticmethod
    def analytic_ping(pings):
        length = len(pings)
        for idx, ping in enumerate(pings):
            if ping is not None:
                replied_idx = None
                reply_idx = None
                if isinstance(ping, int):
                    replied_idx = ping
                    reply_idx = idx + 1
                elif isinstance(ping, list):
                    replied_idx = ping[-1]
                    reply_idx = idx + 1

                print(round(replied_idx / length, 2), round(reply_idx / length, 2))

    def statistic_ping(self, pings):
        for idx, ping in enumerate(pings):
            if ping is not None:
                replied_idx = None
                reply_idx = None
                if isinstance(ping, int):
                    replied_idx = ping
                    reply_idx = idx + 1
                elif isinstance(ping, list):
                    replied_idx = ping[-1]
                    reply_idx = idx + 1

                if (replied_idx, reply_idx) in self.statistic:
                    self.statistic[(replied_idx, reply_idx)] += 1
                else:
                    self.statistic[(replied_idx, reply_idx)] = 1


    @staticmethod
    def parse_ping(body):
        try:
            body = '🐉' + body.replace('...', '.') + '🐉'
        except AttributeError:
            return None

        # Step 1
        """
        在 body 字符串中搜索以空白字符或者 '🐉' 字符开头，
        紧跟着 @ 符号，
        然后后面跟着至少两个连续的非特定字符（特定字符指的是空白字符、逗号、冒号、斜杠、问号、感叹号、'🐉'字符、'['字符、']'字符、或者'('')'字符，
        这部分就是一个用户名，然后提取这个用户名。
        """
        match = re.findall(r'[\s🐉]@([^\s:,/!?🐉\[\]()：，！？【】（）。]{2,})', body)[:2]
        if match:
            names = []
            for name in match:
                # Step 2
                if name.endswith('.'):
                    if len(name) > 2:
                        name = name[: -1]

                # Step 3
                if name.endswith("'"):
                    name = name[: -1]
                elif name.endswith("'s"):
                    name = name[: -2]

                names.append(name)

            return names
        else:
            return None

    @staticmethod
    def iterparse(filename):
        context = ET.iterparse(filename, events=('end',))
        _, root = next(context)
        for event, elem in context:
            if elem.tag == 'Thread':
                yield elem
                root.clear()

    @staticmethod
    def get_test_participants():
        participants = [
            'Poster',
            'Bob',
            'Alice',
            'name',
            'Jo Miller',
            'John',
            'B. Gates',
            'name',
            'B.Gates',
            'Peter Smith',
            'Peter Johns',
            'P Smith',
            'Pawe艂',
            'Jörg',
        ]

        return participants

    @staticmethod
    def get_test_examples():
        examples = [
            "@O'Conner's.)",
            '@name some text',
            '@name: some text',
            '@name. Some text',
            '@name, some text',
            'some text, @name',
            'some text, @name, more text',
            'Some text, @name.',
            "This is mentioned in @name's comment.",
            '@nobody and @name, some text',
            '@Poster and @name, some text',
            '@Jo',
            '@B.',
            '@P.',
            '@psm',
            '@psmith',
            '@pet',
            '@peter',
            '@peters',
            '@petersmith',
            '@peterj',
            '@name...',
            '@alix',
            '@aliceinwonderlan',
            '@Pawe艂',
            '@piere',
            '@jorg',
            '@joerg',
            'abc@name',
            '*@name*',
            '*@name:*',
            '[@name](https://some-url)',
            '@[name](https://some-url)',
            '@P Smith',
            '`@name Hi!`',
        ]

        return examples

    def statistics(self):
        self.cprint('#Questions:', self.num_of_questions)
        self.cprint('#Questions with Answers:', self.num_of_questions_with_answers)
        self.cprint('#Questions with Comments:', self.num_of_questions_with_comments)
        self.cprint('#Questions with Interactions:', self.num_of_questions_with_interactions)

        self.cprint('#Answers:', self.num_of_answers)
        self.cprint('#Answers with Comments:', self.num_of_answers_with_comments)
        self.cprint('#Answers with Interactions:', self.num_of_answers_with_interactions)

        self.cprint('#Comments:', self.num_of_comments)
        self.cprint('#Comments with Interactions:', self.num_of_comments_with_interactions)

        self.cprint('#Interactions:', self.num_of_interactions)

    def test(self):
        examples = self.get_test_examples()
        for example in examples:
            _, index = self.interaction(self.get_test_participants(), self.parse_ping(example))
            print(_, index)

    @staticmethod
    # 写一个cprint函数，输入两个字符串，一个是注释，一个是内容，注释黄色高亮
    def cprint(comment, content):
        print('\033[33m' + comment + '\033[0m', content)


def main():
    xml_path = 'Outs/meta.stackoverflow.com/SortedPostsWithCommentsWithUserIDAndDisplayName.xml'

    ping_match = PingMatch(
        xml_path=xml_path
    )
    ping_match.main()
    ping_match.statistics()

    # for key, value in ping_match.statistic.items():
    #     print(f'{key[0]}, {key[1]}, {value}')


if __name__ == '__main__':
    main()
