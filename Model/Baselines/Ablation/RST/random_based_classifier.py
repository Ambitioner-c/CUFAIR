# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/8/16 19:53
import argparse
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'


import numpy as np
import torch
from torch.utils.data import DataLoader

import transformers
from tqdm import tqdm

transformers.logging.set_verbosity_error()

from transformers import (
    set_seed,
    AutoTokenizer,
)

from Model.Baselines.Ablation.RST.DataLoader.Dataset import RSTDataset
from Model.Unit.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


def get_metrics(input: np.array, target: np.array):
    acc = accuracy_score(input, target)
    pre = precision_score(input, target, average='weighted')
    rec = recall_score(input, target, average='weighted')
    f1 = f1_score(input, target, average='weighted')

    return acc, pre, rec, f1


def evaluate(task_name, test_dataloader):
    test_accs, test_pres, test_recs, test_f1s = [], [], [], []
    for test_sample in tqdm(test_dataloader):
        test_labels = test_sample['label']
        with torch.no_grad():
            test_output = np.random.rand(test_labels.shape[0], 3)

            test_acc, test_pre, test_rec, test_f1 = get_metrics(test_output, test_labels.numpy())
            test_accs.append(test_acc)
            test_pres.append(test_pre)
            test_recs.append(test_rec)
            test_f1s.append(test_f1)
    temp_test_result = (f'{task_name}\t'
                        f'test_acc:{round(np.mean(test_accs), 4)}\t'
                        f'test_pre:{round(np.mean(test_pres), 4)}\t'
                        f'test_rec:{round(np.mean(test_recs), 4)}\t'
                        f'test_f1:{round(np.mean(test_f1s), 4)}')
    print(temp_test_result)


def parse_args():
    parser = argparse.ArgumentParser(description='Random-based Classifier')

    parser.add_argument('--task_name', nargs='?', default='random_based_classifier',
                        help='Task name')
    parser.add_argument('--data_dir', nargs='?', default='/home/cuifulai/Projects/CQA/Data/RST/GUM',
                        help='Data directory')
    parser.add_argument('--pretrained_model_path', nargs='?', default='/data/cuifulai/PretrainedModel/bert-base-uncased',
                        help='Pretrained model path')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Max length')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--num_labels', type=int, default=3,
                        help='Number of labels')
    parser.add_argument('--gum_types', nargs='?', default=None,
                        help='GUM types')

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

    test_dataset = RSTDataset(tokenizer, args.data_dir, mode='Test', max_length=args.max_length, types=args.gum_types)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    evaluate(args.task_name, test_dataloader)


if __name__ == '__main__':
    main()
