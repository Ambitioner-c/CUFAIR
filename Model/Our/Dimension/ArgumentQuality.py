# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/10 11:23
import re
import html
from collections import Counter
from datetime import datetime
from typing import Optional

import spacy
from spellchecker import SpellChecker
from tqdm import trange
from transformers import set_seed

from Model.DataLoader.DataPack import DataPack
from Model.DataLoader.DataProcessor import OurProcessor


class ArgumentQuality:
    def __init__(
            self,
            nlp: spacy.Language
    ):
        self.nlp = nlp

    @staticmethod
    def get_features(depth, readability, objectivity, timeliness, accuracy, structure):
        depth_features = [float(value) for key, value in depth.items()]
        readability_features = [float(value) for key, value in readability.items()]
        objectivity_features = [float(value) for key, value in objectivity.items()]
        timeliness_features = [float(value) for key, value in timeliness.items()]
        accuracy_features = [float(value) for key, value in accuracy.items()]
        structure_features = [float(value) for key, value in structure.items()]

        return depth_features + readability_features + objectivity_features + timeliness_features + accuracy_features + structure_features

    @staticmethod
    def get_features_name(depth, readability, objectivity, timeliness, accuracy, structure):
        depth_features_name = [key for key, value in depth.items()]
        readability_features_name = [key for key, value in readability.items()]
        objectivity_features_name = [key for key, value in objectivity.items()]
        timeliness_features_name = [key for key, value in timeliness.items()]
        accuracy_features_name = [key for key, value in accuracy.items()]
        structure_features_name = [key for key, value in structure.items()]

        return depth_features_name + readability_features_name + objectivity_features_name + timeliness_features_name + accuracy_features_name + structure_features_name

    def get_quality(
            self,
            data_pack: DataPack = None,
            q_name: str = None,
            q_date: str = None,
            a_id: str = None,
            a_date: str = None,
            answer: str = None,
            pre_a_date: Optional[str] = None,
            a_ids: list[str] = None,
            comments: list[str] = None,
            participants: list[str] = None,
            pings: list[int] = None,
    ):
        if data_pack is not None:
            dp = data_pack.copy()
            relation_df = dp.relation
            right_id_df = dp.right_id
            right_df = dp.right
            extend_df = dp.extend

            for _ in trange(relation_df.shape[0], desc='Getting quality features'):
                id_left = relation_df.iloc[_]['id_left']
                id_right = relation_df.iloc[_]['id_right']

                extend_json = extend_df.loc[id_left].extend

                a_id = right_id_df.loc[id_right].right_id
                answer = right_df.loc[id_right].text_right
                a_ids = extend_json['AIDs']
                idx = a_ids.index(a_id)

                q_name = extend_json['QName']
                q_date = extend_json['QDate']
                a_id = extend_json['AIDs'][idx]
                a_date = extend_json['ADates'][idx]
                try:
                    answer = self.unescape_html(answer)
                except TypeError:
                    answer = ''
                pre_a_date = extend_json['ADates'][idx - 1] if idx > 0 else None
                a_ids = extend_json['AIDs']
                comments = extend_json['CBody'][idx]
                participants = extend_json['AParticipants'][idx]
                pings = extend_json['APings'][idx]

                depth = self.get_depth(answer)
                readability = self.get_readability(answer, depth)
                objectivity = self.get_objectivity(answer, comments, q_name, participants, pings)
                timeliness = self.get_timeliness(q_date, a_date, pre_a_date)
                accuracy = self.get_accuracy(answer)
                structure = self.get_structure(q_name, a_id, a_ids, participants, pings)

                data_pack.feature.loc[id_right].feature = self.get_features(depth, readability, objectivity, timeliness, accuracy, structure)
            return data_pack
        else:
            answer = self.unescape_html(answer)

            depth = self.get_depth(answer)
            readability = self.get_readability(answer, depth)
            objectivity = self.get_objectivity(answer, comments, q_name, participants, pings)
            timeliness = self.get_timeliness(q_date, a_date, pre_a_date)
            accuracy = self.get_accuracy(answer)
            structure = self.get_structure(q_name, a_id, a_ids, participants, pings)

            return self.get_features(depth, readability, objectivity, timeliness, accuracy, structure)

    @staticmethod
    def replace_link(_text):
        for old in re.findall(r'(\[.+?]\(.+?\))', _text):
            try:
                _text = _text.replace(old, re.findall(r'(\[.+?])\(.+?\)', _text)[0])
            except IndexError:
                _text = _text

        return _text

    def get_depth(self, answer: str) -> dict:
        # Number of characters in an answer
        num_characters = len(answer.replace(' ', ''))

        # Number of words in an answer
        words = re.sub(r'[^\w\s]', '', self.replace_link(answer)).split()
        num_words = len(words)

        # Number of unique words in an answer
        num_unique_words = len(set(words))

        # Number of sentences in an answer
        num_sentences = len(re.split(r'[.!?][\'\"]? ', answer))

        # Number of nomenclature (e.g., programming code, math formula) in an answer
        num_nomenclature = len(re.findall(r'```.+?```', answer))

        # Number of web links in an answer
        num_web_links = len(re.findall(r'https?://', answer))

        # Number of quotations in an answer
        num_quotations = len(re.findall(r'(\[.+?]\(.+?\))', answer))

        depth = {
            'num_characters': num_characters,
            'num_words': num_words,
            'num_unique_words': num_unique_words,
            'num_sentences': num_sentences,
            'num_nomenclature': num_nomenclature,
            'num_web_links': num_web_links,
            'num_quotations': num_quotations
        }
        return depth

    def get_readability(self, answer: str, depth: dict) -> dict:
        # Number of nouns, adjectives, comparatives, verbs, adverbs, punctuation, and symbols in an answer
        counts = Counter({
            'NOUN': 0,
            'ADJ': 0,
            'COMP': 0,
            'VERB': 0,
            'ADV': 0,
            'PUNCT': 0,
            'SYM': 0
        })
        doc = self.nlp(answer)
        for token in doc:
            if token.pos_ == 'NOUN':
                counts['NOUN'] += 1
            elif token.pos_ == 'ADJ':
                counts['ADJ'] += 1
                if token.tag_ in ('JJR', 'RBR'):
                    counts['COMP'] += 1
            elif token.pos_ == 'VERB':
                counts['VERB'] += 1
            elif token.pos_ == 'ADV':
                counts['ADV'] += 1
            elif token.pos_ == 'PUNCT':
                counts['PUNCT'] += 1
            elif token.pos_ == 'SYM':
                counts['SYM'] += 1

        # Ratio of nouns, adjectives, comparatives, verbs, adverbs, punctuation, and symbols in an answer
        readability = {
            'ratio_nouns': round(counts['NOUN'] / len(doc) if len(doc) > 0 else 0, 4),
            'ratio_adjectives': round(counts['ADJ'] / len(doc) if len(doc) > 0 else 0, 4),
            'ratio_comparatives': round(counts['COMP'] / len(doc) if len(doc) > 0 else 0, 4),
            'ratio_verbs': round(counts['VERB'] / len(doc) if len(doc) > 0 else 0, 4),
            'ratio_adverbs': round(counts['ADV'] / len(doc) if len(doc) > 0 else 0, 4),
            'ratio_punctuation': round(counts['PUNCT'] / len(doc) if len(doc) > 0 else 0, 4),
            'ratio_symbols': round(counts['SYM'] / len(doc) if len(doc) > 0 else 0, 4)
        }

        # Characters to sentences ratio in an answer
        ratio_characters_sentences = round(depth['num_characters'] / depth['num_sentences'] if depth['num_sentences'] > 0 else 0, 4)
        readability['ratio_characters_sentences'] = ratio_characters_sentences

        # Words to sentences ratio in an answer
        ratio_words_sentences = round(depth['num_words'] / depth['num_sentences'] if depth['num_sentences'] > 0 else 0, 4)
        readability['ratio_words_sentences'] = ratio_words_sentences

        # Number of “wh”-type words in an answer
        num_wh_words = len([token.text for token in doc if token.tag_ in ('WDT', 'WP', 'WP$', 'WRB')])
        readability['num_wh_words'] = num_wh_words

        # Number of question marks in an answer
        num_question_marks = answer.count('?')
        readability['num_question_marks'] = num_question_marks

        return readability

    @staticmethod
    def get_objectivity(answer: str, comments: list[str], q_name: str, participants: list[str], pings: list[int]):
        try:
            a_name = participants[0]
        except IndexError:
            a_name = 'Fulai Cui'

        # Number of “thank” words of the question asker or community users to the answerer in an answer-thread
        content = ''
        for idx, ping in enumerate(pings):
            if participants[idx + 1] != a_name and participants[ping] == a_name:
                try:
                    content += comments[idx]
                except TypeError:
                    content += ''
        # Thank you, Thanks, Thx (Thanks), Ty (Thank you), TYVM (Thank you very much), TYSM (Thank you so much), Appreciate, Cheers, Grateful, Gratitude, etc.
        num_thank_words = (
                content.lower().count('thank')
                + content.lower().count('thx')
                + content.lower().count('ty')
                + content.lower().count('appreciate')
                + content.lower().count('cheers')
                + content.lower().count('grateful')
                + content.lower().count('gratitude')
        )

        # Ratio of positive and negative words of the question asker to the answerer in an answer-thread
        content = ''
        for idx, ping in enumerate(pings):
            if participants[idx + 1] == q_name and participants[ping] == a_name:
                content += comments[idx]
        # Positive words: Good, Great, Excellent, Wonderful, Fantastic, Amazing, Awesome, Superb, Perfect, etc.
        # Negative words: Bad, Poor, Terrible, Horrible, Awful, Disgusting, Disappointing, etc.
        num_positive_words = (
                content.lower().count('good')
                + content.lower().count('great')
                + content.lower().count('excellent')
                + content.lower().count('wonderful')
                + content.lower().count('fantastic')
                + content.lower().count('amazing')
                + content.lower().count('awesome')
                + content.lower().count('superb')
                + content.lower().count('perfect')
        )
        num_negative_words = (
                content.lower().count('bad')
                + content.lower().count('poor')
                + content.lower().count('terrible')
                + content.lower().count('horrible')
                + content.lower().count('awful')
                + content.lower().count('disgusting')
                + content.lower().count('disappointing')
        )
        ratio_positive_negative_words_asker = round(num_positive_words / (num_negative_words + 0.0001), 4)

        # Ratio of positive and negative words of the community users to the answerer in an answer-thread
        content = ''
        for idx, ping in enumerate(pings):
            if participants[idx + 1] != q_name and participants[idx + 1] != a_name and participants[ping] == a_name:
                try:
                    content += comments[idx]
                except TypeError:
                    content += ''
        num_positive_words = (
                content.lower().count('good')
                + content.lower().count('great')
                + content.lower().count('excellent')
                + content.lower().count('wonderful')
                + content.lower().count('fantastic')
                + content.lower().count('amazing')
                + content.lower().count('awesome')
                + content.lower().count('superb')
                + content.lower().count('perfect')
        )
        num_negative_words = (
                content.lower().count('bad')
                + content.lower().count('poor')
                + content.lower().count('terrible')
                + content.lower().count('horrible')
                + content.lower().count('awful')
                + content.lower().count('disgusting')
                + content.lower().count('disappointing')
        )
        ratio_positive_negative_words_users = round(num_positive_words / (num_negative_words + 0.0001), 4)

        # Ratio of positive and negative words in an answer
        num_positive_words = (
                answer.lower().count('good')
                + answer.lower().count('great')
                + answer.lower().count('excellent')
                + answer.lower().count('wonderful')
                + answer.lower().count('fantastic')
                + answer.lower().count('amazing')
                + answer.lower().count('awesome')
                + answer.lower().count('superb')
                + answer.lower().count('perfect')
        )
        num_negative_words = (
                answer.lower().count('bad')
                + answer.lower().count('poor')
                + answer.lower().count('terrible')
                + answer.lower().count('horrible')
                + answer.lower().count('awful')
                + answer.lower().count('disgusting')
                + answer.lower().count('disappointing')
        )
        ratio_positive_negative_words_answer = round(num_positive_words / (num_negative_words + 0.0001), 4)

        objectivity = {
            'num_thank_words': num_thank_words,
            'ratio_positive_negative_words_asker': ratio_positive_negative_words_asker,
            'ratio_positive_negative_words_users': ratio_positive_negative_words_users,
            'ratio_positive_negative_words_answer': ratio_positive_negative_words_answer
        }
        return objectivity

    @staticmethod
    def get_timeliness(q_date: str, a_date: str, pre_a_date: Optional[str]) -> dict:
        q_data = datetime.strptime(q_date, "%Y-%m-%dT%H:%M:%S.%f")
        a_date = datetime.strptime(a_date, "%Y-%m-%dT%H:%M:%S.%f")
        if pre_a_date:
            pre_a_date = datetime.strptime(pre_a_date, "%Y-%m-%dT%H:%M:%S.%f")
        else:
            pre_a_date = a_date

        # Time-lapse between a question and an answer
        time_lapse_a_q = (a_date - q_data).total_seconds() / 60

        # Time-lapse between an answer and the previous answer
        time_lapse_a_pre_a = (a_date - pre_a_date).total_seconds() / 60

        # Answer’s time as hour of day and day of week
        hour_of_day = a_date.hour
        day_of_week = a_date.weekday()

        timeliness = {
            'time_lapse_a_q': time_lapse_a_q,
            'time_lapse_a_pre_a': time_lapse_a_pre_a,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week
        }
        return timeliness

    def get_accuracy(self, answer: str) -> dict:
        # Number and ratio of capitalization errors in an answer
        errors = 0

        def replace_code(_text):
            _text = re.sub(r'```.+?```', '', _text)
            return _text

        words = replace_code(self.replace_link(answer)).split()

        pre_end = True
        for word in words:
            if 'I' in word:
                pre_end = False
                continue
            if word.startswith(('"', '[', '(')):
                word.replace('"', '').replace('[', '').replace('(', '')
                pre_end = True
            if pre_end:
                if word[0].islower():
                    errors += 1
                pre_end = False
            else:
                if not word[0].islower():
                    errors += 1

                if word.endswith(('.', '?', '!',)):
                    pre_end = True
                else:
                    pre_end = False
        num_capitalization_errors = errors
        ratio_capitalization_errors = round(errors / len(words) if len(words) > 0 else 0, 4)

        # Number and ratio of punctuation errors in an answer
        sentences = re.split(r'(?<=[.!?`)]) +', answer)
        errors = sum(1 for sentence in sentences if re.search(r"[。，；：！？（）【】{}‘“]", sentence))
        num_punctuation_errors = errors
        ratio_punctuation_errors = round(errors / len(sentences) if len(sentences) > 0 else 0, 4)

        # Number and ratio of typos in an answer
        doc = self.nlp(answer)
        spell = SpellChecker()
        errors = sum(1 for token in doc if token.is_alpha and token.text not in spell)
        num_typos = errors
        ratio_typos = round(errors / len(words) if len(words) > 0 else 0, 4)

        # Number and ratio of out-of-vocabulary words in an answer
        errors = sum(1 for token in doc if token.is_alpha and not token.is_oov)
        num_oov_words = errors
        ratio_oov_words = round(errors / len(words) if len(words) > 0 else 0, 4)

        accuracy = {
            'num_capitalization_errors': num_capitalization_errors,
            'ratio_capitalization_errors': ratio_capitalization_errors,
            'num_punctuation_errors': num_punctuation_errors,
            'ratio_punctuation_errors': ratio_punctuation_errors,
            'num_typos': num_typos,
            'ratio_typos': ratio_typos,
            'num_oov_words': num_oov_words,
            'ratio_oov_words': ratio_oov_words
        }
        return accuracy

    @staticmethod
    def get_structure(q_name: str, a_id: str, a_ids: list[str], participants: list[str], pings: list[int]) -> dict:
        # An answer’s position metrics in term of answer order in a question-thread
        position = round((a_ids.index(a_id) + 1) / (len(a_ids) - a_ids.index(a_id)), 4)

        # An answer’s position metrics in terms of answer order from top or bottom in a question-thread
        position_from_top = a_ids.index(a_id) + 1
        position_from_bottom = len(a_ids) - a_ids.index(a_id)

        # Number of “pinging” to the answerer’s comments in an answer-thread
        num_pinging_answerer = 0
        try:
            a_name = participants[0]
        except IndexError:
            a_name = 'Fulai Cui'
        for idx, participant in enumerate(participants):
            if idx == 0:
                continue
            if participant == a_name:
                num_pinging_answerer += pings.count(idx)

        # Number of “pinging” to the community users’ comments in an answer-thread
        num_pinging_user = 0
        try:
            a_name = participants[0]
        except IndexError:
            a_name = 'Fulai Cui'
        for idx, participant in enumerate(participants):
            if idx == 0:
                continue
            if participant != a_name:
                num_pinging_user += pings.count(idx)

        # If the answerer or community users appear multiple times in an answer-thread
        if_appear_multiple = 1 if participants.count(q_name) > 1 or participants.count(a_name) > 1 else 0

        # If the community users “pinging” the answerer in an answer-thread
        if_pinging_answerer = 1 if num_pinging_answerer > 0 else 0

        # If the community users “pinging” the other community users in an answer-thread
        if_pinging_user = 1 if num_pinging_user > 0 else 0

        # If the answer is the first response to the question in a question-thread
        if_first_response = 1 if a_ids.index(a_id) == 0 else 0

        # If the answer is the second response to the question in a question-thread
        if_second_response = 1 if a_ids.index(a_id) == 1 else 0

        structure = {
            'position': position,
            'position_from_top': position_from_top,
            'position_from_bottom': position_from_bottom,
            'num_pinging_answerer': num_pinging_answerer,
            'num_pinging_user': num_pinging_user,
            'if_appear_multiple': if_appear_multiple,
            'if_pinging_answerer': if_pinging_answerer,
            'if_pinging_user': if_pinging_user,
            'if_first_response': if_first_response,
            'if_second_response': if_second_response
        }
        return structure

    def get_relevancy(self):
        # TODO: self.get_features_length() is 44
        pass

    @staticmethod
    def unescape_html(text: str) -> str:
        return html.unescape(text)


def simulation():
    spacy_path = "/data/cuifulai/Spacy/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1"
    nlp = spacy.load(spacy_path)

    answer = "So goed, njbaj. What is your name? Why, I don know. Beeutiful girl. I saw &quot;John Smith&quot; yesterday. &quot;Really?&quot; he asked! ```Python``` [Google](https://www.google.com) ```Programming``` [Baidu](http://www.baidu.com)"
    comments = [
        "Thank you for your answer, thx.",
        "Good answer.",
        "Ok.",
        "Great answer! Thanks.",
        "Perfect answer! TYVM."
    ]
    q_name = 'Alice'
    participants = [
        'Bob',
        'Charlie',
        'Alice',
        'Alice',
        'Bob',
        'Jerry'
    ]
    pings = [
        0,
        0,
        1,
        2,
        4
    ]
    q_date = '2016-07-28T09:15:01.607'
    a_date = '2016-07-29T10:30:01.607'
    pre_a_date = '2016-07-29T10:15:01.607'
    a_id = '123'
    a_ids = [
        '456',
        '123',
        '789',
        '101',
        '112'
    ]

    argument_quality = ArgumentQuality(nlp)
    features = argument_quality.get_quality(None, q_name, q_date, a_id, a_date, answer, pre_a_date, a_ids, comments, participants, pings)
    print(features)


def fact():
    set_seed(2024)

    data_dir = '/home/cuifulai/Projects/CQA/Data/StackExchange'
    data_name = 'meta.stackoverflow.com'

    spacy_path = "/data/cuifulai/Spacy/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1"
    nlp = spacy.load(spacy_path)

    argument_quality = ArgumentQuality(nlp)

    # train_data_pack = OurProcessor(
    #     data_name=data_name,
    #     stage='train',
    #     task='ranking',
    #     filtered=False,
    #     threshold=5,
    #     normalize=True,
    #     return_classes=False,
    #     limit=0
    # ).get_train_examples(data_dir)
    # train_data_pack = argument_quality.get_quality(train_data_pack)
    # print(train_data_pack.frame())

    # dev_data_pack = OurProcessor(
    #     data_name=data_name,
    #     stage='dev',
    #     task='ranking',
    #     filtered=True,
    #     threshold=5,
    #     normalize=True,
    #     return_classes=False,
    #     limit=0
    # ).get_dev_examples(data_dir)
    # dev_data_pack = argument_quality.get_quality(dev_data_pack)
    # print(dev_data_pack.frame())

    test_data_pack = OurProcessor(
        data_name=data_name,
        stage='test',
        task='ranking',
        filtered=True,
        threshold=5,
        normalize=True,
        return_classes=False,
        limit=0
    ).get_test_examples(data_dir)
    test_data_pack = argument_quality.get_quality(test_data_pack)
    print(test_data_pack.frame())


if __name__ == '__main__':
    fact()
