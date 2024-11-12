# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/11/12 10:58
import json
from datetime import datetime
from typing import Optional

from tqdm import tqdm, trange
from transformers import set_seed

import xml.etree.ElementTree as ElementTree
from xml.dom import minidom


class Utterance:
    def __init__(
            self,
            conversation_path,
            utterance_path,
            samples: Optional[list] = None,
            shuffle: bool = False,
            save: Optional[str] = False,
    ):
        self.conversation_path = conversation_path
        self.utterance_path = utterance_path
        self.samples = samples
        self.shuffle = shuffle
        self.save = save

        self.root = ElementTree.Element('AfD')

    def main(self):
        idx2label = self.read_conversation()
        nominations, comments_list, labels = self.read_utterance(idx2label)
        self.generate_xml(nominations, comments_list, labels)

    def generate_xml(self, nominations: list, comments_list: list[list], labels: list):
        for i in trange(len(nominations)):
            nomination = nominations[i]
            comments = comments_list[i]
            label = labels[i]

            thread = ElementTree.SubElement(self.root, 'Thread')
            thread.set('ID', nomination['conversation_id'])
            if label == 0:
                thread.set('Label', 'delete')
            elif label == 1:
                thread.set('Label', 'keep')

            right = ElementTree.SubElement(thread, 'Nomination')
            right.set('ID', nomination['id'])
            right.set('ConversationID', nomination['conversation_id'])
            right.set('Type', nomination['meta']['type'])
            right.set('Speaker', nomination['speaker'])
            right.set('Timestamp', str(int(nomination['timestamp'])))
            r_body = ElementTree.SubElement(right, 'NBody')
            r_body.text = nomination['text'] or ''

            others = ElementTree.SubElement(thread, 'Comments')
            others.set('CommentCount', str(len(comments)))
            for comment in comments:
                other = ElementTree.SubElement(others, 'Comment')
                other.set('ID', comment['id'])
                other.set('ConversationID', comment['conversation_id'])
                other.set('Type', comment['meta']['type'])
                other.set('ReplyTo', comment['reply-to'] or '')
                other.set('Speaker', comment['speaker'])
                other.set('Timestamp', str(int(comment['timestamp'])))
                o_body = ElementTree.SubElement(other, 'CBody')
                o_body.text = comment['text'] or ''

        xml_str = ElementTree.tostring(self.root, encoding='utf-8')
        parsed_str = minidom.parseString(xml_str)
        pretty_str = parsed_str.toprettyxml(indent="\t")
        if self.save:
            with open(self.save, 'w', encoding='utf-8') as f:
                f.write(pretty_str)

    def read_utterance(self, idx2label: json):
        nominations = []
        comments_list = []
        labels = []
        with open(self.utterance_path, 'r', encoding='utf-8') as f:
            temp_nomination = None
            temp_comments = []
            for line in tqdm(f, desc="Reading utterances.jsonl"):
                utterance = json.loads(line)

                conversation_id = utterance['conversation_id']
                if conversation_id not in idx2label:
                    continue
                try:
                    datetime.fromtimestamp(utterance['timestamp'])
                except TypeError:
                    continue

                meta_type = utterance['meta']['type']
                if meta_type == 'nomination':
                    if temp_nomination is None:
                        temp_nomination = utterance
                    elif conversation_id != temp_nomination['conversation_id']:
                        nominations.append(temp_nomination)
                        comments_list.append(temp_comments)
                        labels.append(idx2label[conversation_id])

                        temp_nomination = utterance
                        temp_comments = []
                else:
                    temp_comments.append(utterance)
        assert len(nominations) == len(comments_list) == len(labels)
        return nominations, comments_list, labels

    @staticmethod
    def parse_utterance(utterance: json):
        idx = utterance['id']
        conversation_id = utterance['conversation_id']
        meta_type = utterance['meta']['type']
        reply_to = utterance['reply-to']
        speaker = utterance['speaker']
        text = utterance['text']
        timestamp = utterance['timestamp']
        return idx, conversation_id, meta_type, reply_to, speaker, text, timestamp

    def read_conversation(self):
        idx2label = dict()
        with open(self.conversation_path, 'r', encoding='utf-8') as f:
            for line in f:
                idx, label = line.strip().split(',')
                idx2label[idx] = label

        return idx2label


def main():
    set_seed(2024)

    data_dir = '/home/cuifulai/Projects/CQA/Data/WikiPedia'
    data_name = 'wiki-articles-for-deletion-corpus'

    conversation_path = f'{data_dir}/Outs/{data_name}/idx2label.csv'
    utterance_path = f'{data_dir}/Dumps/{data_name}/utterances.jsonl'

    save_path = f'{data_dir}/Outs/{data_name}/AfD.xml'

    utterance = Utterance(
        conversation_path,
        utterance_path,
        save=save_path,
    )
    utterance.main()


if __name__ == '__main__':
    main()
