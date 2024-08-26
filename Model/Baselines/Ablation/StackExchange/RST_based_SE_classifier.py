# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/8/26 10:56
import argparse
import copy
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import os.path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import transformers
from tqdm import tqdm

transformers.logging.set_verbosity_error()

from transformers import (
    set_seed,
    AutoTokenizer,
    BertForSequenceClassification,
    AutoConfig
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


class RSTModel(nn.Module):
    def __init__(self, freeze=False, pretrained_model_path=None, device=None, num_labels=2):
        super(RSTModel, self).__init__()
        self.device = device

        self.bert = BertForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=num_labels)
        for p in self.bert.parameters():
            p.data = p.data.contiguous()

        self.config = AutoConfig.from_pretrained(pretrained_model_path)

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, sample: pd.DataFrame):
        feature = sample['feature'].to(self.device)
        output = self.bert(feature).logits

        return output


def get_metrics(input: np.array, target: np.array):
    acc = accuracy_score(input, target)
    pre = precision_score(input, target, average='weighted')
    rec = recall_score(input, target, average='weighted')
    f1 = f1_score(input, target, average='weighted')

    return acc, pre, rec, f1


def train():
    pass


def evaluate(task_name, model, test_dataloader):
    model.eval()

    test_accs, test_pres, test_recs, test_f1s = [], [], [], []
    for test_sample in tqdm(test_dataloader):
        test_labels = test_sample['label']
        with torch.no_grad():
            test_output = model(test_sample)

            test_acc, test_pre, test_rec, test_f1 = get_metrics(test_output.cpu().numpy(), test_labels.cpu().numpy())

            test_accs.append(test_acc)
            test_pres.append(test_pre)
            test_recs.append(test_rec)
            test_f1s.append(test_f1)
    temp_test_result = (f'{task_name}\t'
                        f'test_acc:{round(np.mean(test_accs), 4)}\t'
                        f'test_pre:{round(np.mean(test_pres), 4)}\t'
                        f'test_rec:{round(np.mean(test_recs), 4)}\t'
                        f'test_f1:{round(np.mean(test_f1s), 4)}\t')
    print(temp_test_result)


def parse_args():
    parser = argparse.ArgumentParser(description='RSE-based SE Classifier')

    parser.add_argument('--task_name', nargs='?', default='RSE_based_SE_classifier',
                        help='Task name')
    parser.add_argument('--data_dir', nargs='?', default='/home/cuifulai/Projects/CQA/Data/StackExchange',
                        help='Data directory')
    parser.add_argument('--data_name', nargs='?', default='meta.stackoverflow.com',
                        help='Data name')
    parser.add_argument('--pretrained_model_path', nargs='?', default='/data/cuifulai/PretrainedModel/bert-base-uncased',
                        help='Pretrained model path')
    parser.add_argument('--finetuned_model_path', nargs='?', default='/home/cuifulai/Projects/CQA/Model/Baselines/Ablation/RST/FinetunedModel/RST-20240818_195316/bert-base-uncased',
                        help='Finetuned model path')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Max length')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--device', nargs='?', default='cuda:0',
                        help='Device')
    parser.add_argument('--freeze', type=bool, default=False,
                        help='Freeze')
    parser.add_argument('--num_labels', type=int, default=3,
                        help='Number of labels')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold')



    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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
    all_dataloader = DataLoader(all_dataset, batch_size=args.batch_size, shuffle=False)

    model = RSTModel(
        freeze=args.freeze,
        pretrained_model_path=args.finetuned_model_path,
        device=device,
        num_labels=args.num_labels
    ).to(device)

    evaluate(args.task_name, model, all_dataloader)


if __name__ == '__main__':
    main()
