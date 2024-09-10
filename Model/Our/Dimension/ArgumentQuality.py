# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/10 11:23
import re
import html


class ArgumentQuality:
    def __init__(self):
        pass

    def get_quality(self, text: str):
        text = self.unescape_html(text)
        depth = self.get_depth(text)
        print(depth)

    @staticmethod
    def get_depth(text: str):
        # Number of characters in an answer
        num_characters = len(text.replace(' ', ''))

        def replace_link(_text):
            for old in re.findall(r'(\[.+?]\(.+?\))', _text):
                _text = _text.replace(old, re.findall(r'\[(.+?)]\(.+?\)', _text)[0])
            return _text

        # Number of words in an answer
        words = re.sub(r'[^\w\s]', '', replace_link(text)).split()
        num_words = len(words)

        # Number of unique words in an answer
        num_unique_words = len(set(words))

        # Number of sentences in an answer
        num_sentences = len(re.split(r'[.!?][\'\"]? ', text))

        # Number of nomenclature (e.g., programming code, math formula) in an answer
        num_nomenclature = len(re.findall(r'```.+?```', text))

        # Number of web links in an answer
        num_web_links = len(re.findall(r'https?://', text))

        # Number of quotations in an answer
        num_quotations = len(re.findall(r'(\[.+?]\(.+?\))', text))

        return [num_characters, num_words, num_unique_words, num_sentences, num_nomenclature, num_web_links, num_quotations]

    def get_readability(self):
        pass

    def get_objectivity(self):
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
    text = "X XX &quot;XXX XXX&quot; XXXX. &quot;XXX?&quot; XXXX! ```XXX``` [XXXX](https://www.google.com) ```XXXX``` [XX](http://www.baidu.com)"

    argument_quality = ArgumentQuality()
    argument_quality.get_quality(text)


if __name__ == '__main__':
    main()
