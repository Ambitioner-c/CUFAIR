# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/11/11 21:21
import json
import random
from typing import Optional

from tqdm import tqdm

from transformers import set_seed


class Conversation:
    def __init__(
            self,
            conversation_path: str,
            samples: Optional[list] = None,
            shuffle: bool = False,
            save: Optional[str] = False,
    ):
        self.conversation_path = conversation_path
        self.samples = samples
        self.shuffle = shuffle
        self.save = save

    def main(self):
        idx2label = self.read_conversation()

        if self.samples:
            idx2label = self.do_sample(idx2label)

        if self.shuffle:
            idx2label = self.do_shuffle(idx2label)

        if self.save:
            self.write(idx2label)

        self.statistic(idx2label)

    def read_conversation(self) -> dict:
        idx2label = dict()
        with open(self.conversation_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
            for idx in tqdm(conversations, desc="Reading conversations.json"):
                conversation = conversations[idx]
                # article_title = conversation['meta']['article_title']
                # outcome_id = conversation['meta']['outcome_id']
                outcome_label = conversation['meta']['outcome_label']
                # outcome_raw_label = conversation['meta']['outcome_raw_label']
                # outcome_decision_maker_id = conversation['meta']['outcome_decision_maker_id']
                # outcome_timestamp = conversation['meta']['outcome_timestamp']
                # try:
                #     outcome_timestamp = datetime.datetime.fromtimestamp(outcome_timestamp)
                # except TypeError:
                #     continue
                # outcome_rationale = conversation['meta']['outcome_rationale']
                # vectors = conversation['vectors']

                if outcome_label == 'keep':
                    label = 1
                elif outcome_label == 'delete':
                    label = 0
                else:
                    continue

                idx2label[idx] = label
        return idx2label

    def do_sample(self, idx2label: dict) -> dict:
        keys_0 = [k for k, v in idx2label.items() if v == 0]
        keys_1 = [k for k, v in idx2label.items() if v == 1]

        sample_0 = random.sample(keys_0, min(self.samples[0], len(keys_0)))
        sample_1 = random.sample(keys_1, min(self.samples[1], len(keys_1)))

        sampled_idx2label = {k: idx2label[k] for k in sample_0 + sample_1}
        return sampled_idx2label

    @staticmethod
    def do_shuffle(idx2label: dict) -> dict:
        idx = list(idx2label.items())
        random.shuffle(idx)
        shuffled_idx2label = dict(idx)
        return shuffled_idx2label

    def write(self, idx2label: dict):
        with open(self.save, 'w', encoding='utf-8') as f:
            for idx, label in idx2label.items():
                f.write(f'{idx},{label}\n')

    @staticmethod
    def statistic(idx2label: dict):
        print(f'Number of samples: {len(idx2label)}')
        print(f'Number of positive (keep) samples: {len([v for v in idx2label.values() if v == 1])}')
        print(f'Number of negative (delete) samples: {len([v for v in idx2label.values() if v == 0])}')


def main():
    set_seed(2024)

    data_dir = '/home/cuifulai/Projects/CQA/Data/WikiPedia'
    data_name = 'wiki-articles-for-deletion-corpus'

    conversation_path = f'{data_dir}/Dumps/{data_name}/conversations.json'
    sava_path = f'{data_dir}/Outs/{data_name}/idx2label.csv'

    conversation = Conversation(
        conversation_path,
        samples=None,
        shuffle=True,
        save=sava_path,
    )
    conversation.main()


if __name__ == '__main__':
    main()
