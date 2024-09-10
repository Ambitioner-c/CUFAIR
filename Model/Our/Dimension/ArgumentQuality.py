# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/10 11:23
import re
import html
from collections import Counter
from pprint import pprint

import spacy


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

    def get_quality(
            self,
            answer: str,
            comments: list[str],
            q_name: str,
            participants: list[str],
            pings: list[int]
    ):
        answer = self.unescape_html(answer)

        self.get_depth(answer)
        self.get_readability(answer)
        self.get_objectivity(answer, comments, q_name, participants, pings)

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
            'ratio_nouns': round(counts['NOUN'] / len(doc), 4),
            'ratio_adjectives': round(counts['ADJ'] / len(doc), 4),
            'ratio_comparatives': round(counts['COMP'] / len(doc), 4),
            'ratio_verbs': round(counts['VERB'] / len(doc), 4),
            'ratio_adverbs': round(counts['ADV'] / len(doc), 4),
            'ratio_punctuation': round(counts['PUNCT'] / len(doc), 4),
            'ratio_symbols': round(counts['SYM'] / len(doc), 4)
        }

        # Characters to sentences ratio in an answer
        ratio_characters_sentences = round(self.depth['num_characters'] / self.depth['num_sentences'], 4)
        self.readability['ratio_characters_sentences'] = ratio_characters_sentences

        # Words to sentences ratio in an answer
        ratio_words_sentences = round(self.depth['num_words'] / self.depth['num_sentences'], 4)
        self.readability['ratio_words_sentences'] = ratio_words_sentences

        # Number of “wh”-type words in an answer
        num_wh_words = len([token.text for token in doc if token.tag_ in ('WDT', 'WP', 'WP$', 'WRB')])
        self.readability['num_wh_words'] = num_wh_words

        # Number of question marks in an answer
        num_question_marks = answer.count('?')
        self.readability['num_question_marks'] = num_question_marks

    def get_objectivity(self, answer: str, comments: list[str], q_name: str, participants: list[str], pings: list[int]):
        a_name = participants[0]

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

    def get_timeliness(self):
        pass

    def get_accuracy(self):
        pass

    def get_structure(self):
        pass

    def get_relevancy(self):
        pass

    @staticmethod
    def unescape_html(text: str) -> str:
        return html.unescape(text)


def main():
    spacy_path = "/data/cuifulai/Spacy/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1"
    nlp = spacy.load(spacy_path)

    answer = "$ So good. What is your name? Why. Beautiful. I saw &quot;John Smith&quot; yesterday. &quot;Really?&quot; he asked! ```Python``` [Google](https://www.google.com) ```Programming``` [Baidu](http://www.baidu.com)"
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
        'Tom',
        'Jerry'
    ]
    pings = [
        0,
        0,
        1,
        2,
        0
    ]

    argument_quality = ArgumentQuality(nlp)
    argument_quality.get_quality(answer, comments, q_name, participants, pings)
    pprint(argument_quality.depth)
    pprint(argument_quality.readability)
    pprint(argument_quality.objectivity)


if __name__ == '__main__':
    main()
