# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/11/12 10:58
import json
import random
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
            threshold: int = 1,
    ):
        self.conversation_path = conversation_path
        self.utterance_path = utterance_path
        self.samples = samples
        self.shuffle = shuffle
        self.save = save
        self.threshold = threshold

        self.num = self.samples[0] + self.samples[1]

        self.root = ElementTree.Element('AfD')

    def main(self):
        idx2label = self.read_conversation()
        nominations, comments_list = self.read_utterance(idx2label)
        self.generate_xml(nominations, comments_list, idx2label)

    def generate_xml(self, nominations: list, comments_list: list[list], idx2label: json):
        for i in trange(len(nominations), desc="Generating AfD.xml"):
            nomination = nominations[i]
            comments = comments_list[i]
            label = idx2label[nomination['conversation_id']]

            thread = ElementTree.SubElement(self.root, 'Thread')
            thread.set('ID', nomination['conversation_id'])
            if label == '0':
                thread.set('Label', 'delete')
            elif label == '1':
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
                        if len(temp_comments) >= self.threshold:
                            nominations.append(temp_nomination)
                            comments_list.append(temp_comments)

                        temp_nomination = utterance
                        temp_comments = []
                else:
                    try:
                        if conversation_id == temp_nomination['conversation_id']:
                            temp_comments.append(utterance)
                    except TypeError:
                        continue
        assert len(nominations) == len(comments_list)
        return nominations[: self.num], comments_list[: self.num]

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

    def read_conversation(self) -> dict:
        idx2label = dict()
        with open(self.conversation_path, 'r', encoding='utf-8') as f:
            for line in f:
                idx, label = line.strip().split(',')
                idx2label[idx] = label

        if self.samples:
            idx2label = self.do_sample(idx2label)

        if self.shuffle:
            idx2label = self.do_shuffle(idx2label)

        return idx2label

    def do_sample(self, idx2label: dict) -> dict:
        keys_0 = [k for k, v in idx2label.items() if v == '0']
        keys_1 = [k for k, v in idx2label.items() if v == '1']

        sample_0 = random.sample(keys_0, min(self.samples[0] + 1500, len(keys_0)))
        sample_1 = random.sample(keys_1, min(self.samples[1] + 1500, len(keys_1)))

        sampled_idx2label = {k: idx2label[k] for k in sample_0 + sample_1}
        return sampled_idx2label

    @staticmethod
    def do_shuffle(idx2label: dict) -> dict:
        idx = list(idx2label.items())
        random.shuffle(idx)

        shuffled_idx2label = dict(idx)
        return shuffled_idx2label


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
        samples=[5000, 5000],
        shuffle=True,
        save=save_path,
        threshold=1,
    )
    utterance.main()


if __name__ == '__main__':
    main()
