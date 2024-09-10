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

    def get_quality(self, answer: str):
        answer = self.unescape_html(answer)

        self.get_depth(answer)
        self.get_readability(answer)

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
            'ratio_nouns': counts['NOUN'] / len(doc),
            'ratio_adjectives': counts['ADJ'] / len(doc),
            'ratio_comparatives': counts['COMP'] / len(doc),
            'ratio_verbs': counts['VERB'] / len(doc),
            'ratio_adverbs': counts['ADV'] / len(doc),
            'ratio_punctuation': counts['PUNCT'] / len(doc),
            'ratio_symbols': counts['SYM'] / len(doc)
        }

        # Characters to sentences ratio in an answer
        ratio_characters_sentences = self.depth['num_characters'] / self.depth['num_sentences']
        self.readability['ratio_characters_sentences'] = ratio_characters_sentences

        # Words to sentences ratio in an answer
        ratio_words_sentences = self.depth['num_words'] / self.depth['num_sentences']
        self.readability['ratio_words_sentences'] = ratio_words_sentences

        # Number of “wh”-type words in an answer
        num_wh_words = len([token.text for token in doc if token.tag_ in ('WDT', 'WP', 'WP$', 'WRB')])
        self.readability['num_wh_words'] = num_wh_words

        # Number of question marks in an answer
        num_question_marks = answer.count('?')
        self.readability['num_question_marks'] = num_question_marks

    def get_objectivity(self, answer: str):
        # Number of “thank” words of the question asker or community users to the answerer in an answer-thread

        # Ratio of positive and negative words of the question asker to the answerer in an answer-thread

        # Ratio of positive and negative words of the community users to the answerer in an answer-thread

        # Ratio of positive and negative words in an answer

        pass

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

    text = "$ So bigger. What is your name? Why. Beautiful. I saw &quot;John Smith&quot; yesterday. &quot;Really?&quot; he asked! ```Python``` [Google](https://www.google.com) ```Programming``` [Baidu](http://www.baidu.com)"

    argument_quality = ArgumentQuality(nlp)
    argument_quality.get_quality(text)
    pprint(argument_quality.depth)
    pprint(argument_quality.readability)


if __name__ == '__main__':
    main()
