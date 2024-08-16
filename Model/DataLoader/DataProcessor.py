# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/6/29 8:53
import random
from typing import *
import emoji
from transformers import DataProcessor
import logging
from abc import ABC
import os
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


try:
    import xml.etree.cElementTree as ElementTree
except ImportError:
    import xml.etree.ElementTree as ElementTree
from tqdm.auto import tqdm

import pandas as pd


logger = logging.getLogger(__name__)


class HeartQAProcessor(DataProcessor, ABC):
    def __init__(self, k=10):
        """

        """
        self.K = k
        super(HeartQAProcessor, self).__init__()

    def get_train_examples(self, data_dir) -> pd.DataFrame:
        df, __ = self.create_examples(os.path.join(data_dir, 'Train', os.listdir(data_dir + '/Train')[0]), 'Train@1')
        _, __ = self.create_examples(os.path.join(data_dir, 'Train', os.listdir(data_dir + '/Train')[1]), 'Train@2', __)
        return pd.concat([df, _], ignore_index=True)
        # return self.create_examples(os.path.join(data_dir, 'Train', os.listdir(data_dir + '/Train')[0]), 'Train@1')

    def get_dev_examples(self, data_dir) -> pd.DataFrame:
        df, _ = self.create_examples(os.path.join(data_dir, 'Dev', os.listdir(data_dir + '/Dev')[0]), 'Dev')
        return df

    def get_test_examples(self, data_dir) -> pd.DataFrame:
        df, _ = self.create_examples(os.path.join(data_dir, 'Test', os.listdir(data_dir + '/Test')[0]), 'Test')
        return df

    def get_labels(self) -> None:
        return

    def create_examples(self, filepath, mode, id_question=0) -> (pd.DataFrame, int):
        """
        #questions:#answers:#labels=1:10:10
        :param filepath:
        :param mode
        :param id_question
        :return: questions, answers, labels
        """

        id_questions = []
        questions = []
        question_ids = []
        all_answers = []
        answers = []
        answer_ids = []
        all_answer_ids = []
        labels = []
        all_labels = []

        tree = ElementTree.parse(filepath)
        root = tree.getroot()

        for Thread in tqdm(root.findall('Thread')):
            RelQuestion = Thread.find('RelQuestion')
            question = RelQuestion.find('RelQBody').text
            question_id = int(str(RelQuestion.get('RELQ_USERID')).replace('U', ''))
            # print(emoji.emojize('Question:thinking_face:' + question))

            if question is None:
                question = Thread.find('RelQuestion').find('RelQSubject').text

            for RelComment in Thread.findall('RelComment'):
                answer = RelComment.find('RelCText').text
                answer_id = int(str(RelComment.get('RELC_USERID')).replace('U', ''))
                label = RelComment.get('RELC_RELEVANCE2RELQ')

                if label == 'Good':
                    label = 1
                elif label == 'PotentiallyUseful':
                    label = 0
                else:
                    label = 0

                # print(emoji.emojize("Answer" + str(n) + ":winking_face:" + answer + "\n" + "Label:label:" + label))

                id_questions.append(id_question)
                questions.append(question)
                question_ids.append(question_id)
                answers.append(answer)
                answer_ids.append(answer_id)
                labels.append(label)
            for j in range(10):
                if self.K == 10:
                    temp_answers = answers[-10:]
                    temp_answer_ids = answer_ids[-10:]
                    temp_labels = labels[-10:]
                else:
                    temp_answers = answers[-10:self.K-10]
                    temp_answer_ids = answer_ids[-10:self.K-10]
                    temp_labels = labels[-10:self.K-10]

                # random.shuffle(temp_answers)
                # random.shuffle(temp_answer_ids)
                # random.shuffle(temp_labels)

                all_answers.append(temp_answers)
                all_answer_ids.append(temp_answer_ids)
                all_labels.append(temp_labels)

            id_question += 1

        logger.warning(mode)
        logger.warning(emoji.emojize(
            '\tQuestion:thinking_face:' + str(int(len(questions)/10))))
        logger.warning(emoji.emojize(
            '\tAnswer:winking_face:' + str(len(answers))))

        df = pd.DataFrame({
            'id': id_questions,
            'question': questions,
            'question_id': question_ids,
            'answers': all_answers,
            'answer_ids': all_answer_ids,
            'labels': all_labels,
            'answer': answers,
            'label': labels
        })
        return df, id_question


def main():
    data_dir = '../../Data/SemEvalv3.2'

    processor = HeartQAProcessor(10)
    df = processor.get_train_examples(data_dir)

    # print(df)
    # print(df.head(10))
    print(df.head(10).to_csv())


if __name__ == '__main__':
    main()
