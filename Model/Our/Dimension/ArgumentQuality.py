# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/10 11:23
import re
import html
from collections import Counter
from datetime import datetime
from pprint import pprint
from typing import Optional

import spacy
from spellchecker import SpellChecker
from transformers import set_seed

from DataLoader.DataProcessor import OurProcessor


class ArgumentQuality:
    def __init__(
            self,
            nlp: spacy.Language
    ):
        self.nlp = nlp

        self.depth = {
            'num_characters': 0,
            'num_words': 0,
            'num_unique_words': 0,
            'num_sentences': 0,
            'num_nomenclature': 0,
            'num_web_links': 0,
            'num_quotations': 0
        }

        self.readability = {
            'ratio_nouns': 0.0,
            'ratio_adjectives': 0.0,
            'ratio_comparatives': 0.0,
            'ratio_verbs': 0.0,
            'ratio_adverbs': 0.0,
            'ratio_punctuation': 0.0,
            'ratio_symbols': 0.0,
            'ratio_characters_sentences': 0.0,
            'ratio_words_sentences': 0.0,
            'num_wh_words': 0,
            'num_question_marks': 0
        }

        self.objectivity = {
            'num_thank_words': 0,
            'ratio_positive_negative_words_asker': 0.0,
            'ratio_positive_negative_words_users': 0.0,
            'ratio_positive_negative_words_answer': 0.0
        }

        self.timeliness = {
            'time_lapse_a_q': 0.0,
            'time_lapse_a_pre_a': 0.0,
            'hour_of_day': 0,
            'day_of_week': 0
        }

        self.accuracy = {
            'num_capitalization_errors': 0,
            'ratio_capitalization_errors': 0.0,
            'num_punctuation_errors': 0,
            'ratio_punctuation_errors': 0.0,
            'num_typos': 0,
            'ratio_typos': 0.0,
            'num_oov_words': 0,
            'ratio_oov_words': 0.0
        }

        self.structure = {
            'position': 0.0,
            'position_from_top': 0,
            'position_from_bottom': 0,
            'num_pinging_answerer': 0,
            'num_pinging_user': 0,
            'if_appear_multiple': 0,
            'if_pinging_answerer': 0,
            'if_pinging_user': 0,
            'if_first_response': 0,
            'if_second_response': 0
        }

    def get_features(self):
        depth_features = [float(value) for key, value in self.depth.items()]
        readability_features = [float(value) for key, value in self.readability.items()]
        objectivity_features = [float(value) for key, value in self.objectivity.items()]
        timeliness_features = [float(value) for key, value in self.timeliness.items()]
        accuracy_features = [float(value) for key, value in self.accuracy.items()]
        structure_features = [float(value) for key, value in self.structure.items()]

        return depth_features + readability_features + objectivity_features + timeliness_features + accuracy_features + structure_features

    def get_features_name(self):
        depth_features_name = [key for key, value in self.depth.items()]
        readability_features_name = [key for key, value in self.readability.items()]
        objectivity_features_name = [key for key, value in self.objectivity.items()]
        timeliness_features_name = [key for key, value in self.timeliness.items()]
        accuracy_features_name = [key for key, value in self.accuracy.items()]
        structure_features_name = [key for key, value in self.structure.items()]

        return depth_features_name + readability_features_name + objectivity_features_name + timeliness_features_name + accuracy_features_name + structure_features_name

    def get_features_length(self):
        return len(self.get_features())

    def get_quality(
            self,
            q_name: str,
            q_date: str,
            a_id: str,
            a_date: str,
            answer: str,
            pre_a_date: Optional[str],
            a_ids: list[str],
            comments: list[str],
            participants: list[str],
            pings: list[int],
    ):
        answer = self.unescape_html(answer)

        self.get_depth(answer)
        self.get_readability(answer)
        self.get_objectivity(answer, comments, q_name, participants, pings)
        self.get_timeliness(q_date, a_date, pre_a_date)
        self.get_accuracy(answer)
        self.get_structure(q_name, a_id, a_ids, participants, pings)

    def get_depth(self, answer: str):
        # Number of characters in an answer
        num_characters = len(answer.replace(' ', ''))

        def replace_link(_text):
            for old in re.findall(r'(\[.+?]\(.+?\))', _text):
                _text = _text.replace(old, re.findall(r'\[(.+?)]\(.+?\)', _text)[0])
            return _text

        # Number of words in an answer
        words = re.sub(r'[^\w\s]', '', replace_link(answer)).split()
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

        self.depth = {
            'num_characters': num_characters,
            'num_words': num_words,
            'num_unique_words': num_unique_words,
            'num_sentences': num_sentences,
            'num_nomenclature': num_nomenclature,
            'num_web_links': num_web_links,
            'num_quotations': num_quotations
        }

    def get_readability(self, answer: str):
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
        self.readability = {
            'ratio_nouns': round(counts['NOUN'] / len(doc) if len(doc) > 0 else 0, 4),
            'ratio_adjectives': round(counts['ADJ'] / len(doc) if len(doc) > 0 else 0, 4),
            'ratio_comparatives': round(counts['COMP'] / len(doc) if len(doc) > 0 else 0, 4),
            'ratio_verbs': round(counts['VERB'] / len(doc) if len(doc) > 0 else 0, 4),
            'ratio_adverbs': round(counts['ADV'] / len(doc) if len(doc) > 0 else 0, 4),
            'ratio_punctuation': round(counts['PUNCT'] / len(doc) if len(doc) > 0 else 0, 4),
            'ratio_symbols': round(counts['SYM'] / len(doc) if len(doc) > 0 else 0, 4)
        }

        # Characters to sentences ratio in an answer
        ratio_characters_sentences = round(self.depth['num_characters'] / self.depth['num_sentences'] if self.depth['num_sentences'] > 0 else 0, 4)
        self.readability['ratio_characters_sentences'] = ratio_characters_sentences

        # Words to sentences ratio in an answer
        ratio_words_sentences = round(self.depth['num_words'] / self.depth['num_sentences'] if self.depth['num_sentences'] > 0 else 0, 4)
        self.readability['ratio_words_sentences'] = ratio_words_sentences

        # Number of “wh”-type words in an answer
        num_wh_words = len([token.text for token in doc if token.tag_ in ('WDT', 'WP', 'WP$', 'WRB')])
        self.readability['num_wh_words'] = num_wh_words

        # Number of question marks in an answer
        num_question_marks = answer.count('?')
        self.readability['num_question_marks'] = num_question_marks

    def get_objectivity(self, answer: str, comments: list[str], q_name: str, participants: list[str], pings: list[int]):
        try:
            a_name = participants[0]
        except IndexError:
            a_name = 'Fulai Cui'

        # Number of “thank” words of the question asker or community users to the answerer in an answer-thread
        content = ''
        for idx, ping in enumerate(pings):
            if participants[idx + 1] != a_name and participants[ping] == a_name:
                content += comments[idx]
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
                content += comments[idx]
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

        self.objectivity = {
            'num_thank_words': num_thank_words,
            'ratio_positive_negative_words_asker': ratio_positive_negative_words_asker,
            'ratio_positive_negative_words_users': ratio_positive_negative_words_users,
            'ratio_positive_negative_words_answer': ratio_positive_negative_words_answer
        }

    def get_timeliness(self, q_date: str, a_date: str, pre_a_date: Optional[str]):
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

        self.timeliness = {
            'time_lapse_a_q': time_lapse_a_q,
            'time_lapse_a_pre_a': time_lapse_a_pre_a,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week
        }

    def get_accuracy(self, answer: str):
        # Number and ratio of capitalization errors in an answer
        errors = 0

        def replace_link(_text):
            for old in re.findall(r'(\[.+?]\(.+?\))', _text):
                _text = _text.replace(old, re.findall(r'(\[.+?])\(.+?\)', _text)[0])
            return _text

        def replace_code(_text):
            _text = re.sub(r'```.+?```', '', _text)
            return _text

        words = replace_code(replace_link(answer)).split()

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

        self.accuracy = {
            'num_capitalization_errors': num_capitalization_errors,
            'ratio_capitalization_errors': ratio_capitalization_errors,
            'num_punctuation_errors': num_punctuation_errors,
            'ratio_punctuation_errors': ratio_punctuation_errors,
            'num_typos': num_typos,
            'ratio_typos': ratio_typos,
            'num_oov_words': num_oov_words,
            'ratio_oov_words': ratio_oov_words
        }

    def get_structure(self, q_name: str, a_id: str, a_ids: list[str], participants: list[str], pings: list[int]):
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

        self.structure = {
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

    def get_relevancy(self):
        # TODO: self.get_features_length() is 44
        pass

    @staticmethod
    def unescape_html(text: str) -> str:
        return html.unescape(text)


def main():
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
    argument_quality.get_quality(q_name, q_date, a_id, a_date, answer, pre_a_date, a_ids, comments, participants, pings)
    pprint(argument_quality.depth)
    pprint(argument_quality.readability)
    pprint(argument_quality.objectivity)
    pprint(argument_quality.timeliness)
    pprint(argument_quality.accuracy)
    pprint(argument_quality.structure)


def fact():
    set_seed(2024)

    data_dir = '/home/cuifulai/Projects/CQA/Data/StackExchange'
    data_name = 'meta.stackoverflow.com'
    limit = 0

    spacy_path = "/data/cuifulai/Spacy/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1"
    nlp = spacy.load(spacy_path)

    argument_quality = ArgumentQuality(nlp)

    processor = OurProcessor(data_name, limit)
    df = processor.get_train_examples(data_dir)
    for idx in range(df.shape[0]):
        example = df.iloc[idx: idx + 1]
        example = example.to_dict(orient='dict')

        q_id = example['QID'][idx]
        q_name = example['QName'][idx]
        q_date = example['QDate'][idx]
        q_title = example['QTitle'][idx]
        q_body = example['QBody'][idx]
        a_ids = example['AID'][idx]
        a_dates = example['ADate'][idx]
        a_bodys = example['ABody'][idx]
        a_accepteds = example['AAccepted'][idx]
        a_scores = example['AScore'][idx]
        a_participants = example['AParticipants'][idx]
        a_pings = example['APings'][idx]
        c_scores = example['CScore'][idx]
        c_dates = example['CDate'][idx]
        c_bodys = example['CBody'][idx]

        argument_quality.get_quality(
            q_name=q_name,
            q_date=q_date,
            a_id=a_ids[0],
            a_date=a_dates[0],
            answer=a_bodys[0],
            pre_a_date=None,
            a_ids=a_ids,
            comments=c_bodys[0],
            participants=a_participants[0],
            pings=a_pings[0]
        )
        print(argument_quality.get_features())
        print(argument_quality.get_features_length())
        exit()


if __name__ == '__main__':
    fact()
