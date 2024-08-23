# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/2/28 9:51
import xml.etree.ElementTree as ElementTree

from tqdm import tqdm

import matplotlib.pyplot as plt


class QAInteraction:
    def __init__(
            self,
            xml_path,
    ):
        self.xml_path = xml_path

    def main(self):
        num_q = 0
        num_interaction_q = 0
        num_interaction_a = 0
        length_dict_q = {}
        length_dict_a = {}
        for _, thread in tqdm(ElementTree.iterparse(self.xml_path, events=('end',)),
                              desc="Parsing {} XML file".format(self.xml_path)):
            if thread.tag == "Thread":
                num_q += 1

                # question
                q_id = thread.find('Question').attrib['OWNER_USER_ID']

                # answers
                a_ids = []
                for answer in thread.findall('Answer'):
                    a_ids.append(answer.attrib['OWNER_USER_ID'])

                # first case
                if q_id in a_ids:
                    num_interaction_q += 1

                    if len(a_ids) not in length_dict_q:
                        length_dict_q[len(a_ids)] = 1
                    else:
                        length_dict_q[len(a_ids)] += 1

                # second case
                a_ids_set = set(a_ids)
                if len(a_ids) != len(a_ids_set):
                    num_interaction_a += 1

                    if len(a_ids) not in length_dict_a:
                        length_dict_a[len(a_ids)] = 1
                    else:
                        length_dict_a[len(a_ids)] += 1

        print("Num of question: %s" % str(num_q))
        print("Num of first case interaction: %s (%s)" % (str(num_interaction_q), str(num_interaction_q/num_q)))
        print("Num of second case interaction: %s (%s)" % (str(num_interaction_a), str(num_interaction_a / num_q)))

        self.show(length_dict_q)
        self.show(length_dict_a)

    @staticmethod
    def show(_dict):
        sorted_key = sorted(_dict.items())

        labels = [item[0] for item in sorted_key]
        values = [item[1] for item in sorted_key]

        plt.bar(labels, values)
        plt.title('Frequency Histogram')
        plt.xlabel('Number')
        plt.ylabel('Frequency')

        plt.show()


def main():
    xml_path = 'Outs/meta.stackoverflow.com/Posts.xml'

    qa_interaction = QAInteraction(
        xml_path=xml_path
    )
    qa_interaction.main()


if __name__ == '__main__':
    main()
