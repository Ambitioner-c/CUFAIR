# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/11/11 21:21
import json
import random

from tqdm import tqdm

from transformers import set_seed


class Conversation:
    def __init__(
            self,
            conversation_path: str,
            samples: list,
            shuffle: bool,
    ):
        self.conversation_path = conversation_path
        self.samples = samples
        self.shuffle = shuffle

    def main(self) -> dict:
        idx2label = self.read_conversation()
        if self.shuffle:
            idx2label = self.do_shuffle(idx2label)
        return idx2label

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


def main():
    set_seed(2024)

    data_dir = '/home/cuifulai/Projects/CQA/Data/WikiPedia'
    data_name = 'wiki-articles-for-deletion-corpus'

    samples = [5000, 5000]

    conversation_path = f'{data_dir}/Dumps/{data_name}/conversations.json'
    conversation = Conversation(
        conversation_path,
        samples=samples,
        shuffle=True,
    )
    idx2label = conversation.main()
    print(idx2label)


if __name__ == '__main__':
    main()
