# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/8/26 10:56
import argparse
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import transformers
from tqdm import tqdm

transformers.logging.set_verbosity_error()

from transformers import (
    set_seed,
    AutoTokenizer,
)

from Model.Baselines.Ablation.StackExchange.DataLoader.Dataset import SEDataset
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
    parser = argparse.ArgumentParser(description='Random-based SE Classifier')

    parser.add_argument('--task_name', nargs='?', default='Random_based_SE_classifier',
                        help='Task name')
    parser.add_argument('--data_dir', nargs='?', default='/home/cuifulai/Projects/CQA/Data/StackExchange',
                        help='Data directory')
    parser.add_argument('--data_name', nargs='?', default='meta.stackoverflow.com',
                        help='Data name')
    parser.add_argument('--pretrained_model_path', nargs='?', default='/data/cuifulai/PretrainedModel/bert-base-uncased',
                        help='Pretrained model path')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Max length')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--num_labels', type=int, default=3,
                        help='Number of labels')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold')
    parser.add_argument('--split', nargs='?', default=[0.8, 0.1, 0.1],
                        help='Split Data into Train, Dev, Test')

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

    all_dataset = SEDataset(
        tokenizer,
        args.data_dir,
        args.data_name,
        limit=args.limit,
        threshold=args.threshold,
        mode='All',
        max_length=args.max_length
    )
    train_dataset, dev_dataset, test_dataset = random_split(all_dataset, args.split)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    evaluate(args.task_name, test_dataloader)


if __name__ == '__main__':
    main()
