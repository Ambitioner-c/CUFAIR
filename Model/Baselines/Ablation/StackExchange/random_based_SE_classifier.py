# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/8/26 10:56
import argparse
import json
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import os.path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import transformers

transformers.logging.set_verbosity_error()

from transformers import (
    set_seed,
    AutoTokenizer,
)

from Model.Baselines.Ablation.StackExchange.DataLoader.Dataset import (
    AnnotatedSEDataset
)
from Model.Unit.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from Model.Unit.cprint import coloring

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


def get_metrics(input: np.array, target: np.array):
    acc = accuracy_score(input, target)
    pre = precision_score(input, target, average='weighted')
    micro_pre = precision_score(input, target, average='micro')
    macro_pre = precision_score(input, target, average='macro')
    rec = recall_score(input, target, average='weighted')
    micro_rec = recall_score(input, target, average='micro')
    macro_rec = recall_score(input, target, average='macro')
    f1 = f1_score(input, target, average='weighted')
    micro_f1 = f1_score(input, target, average='micro')
    macro_f1 = f1_score(input, target, average='macro')
    try:
        auc = roc_auc_score(input, target, average='weighted')
        micro_auc = roc_auc_score(input, target, average='micro')
        macro_auc = roc_auc_score(input, target, average='macro')
    except ValueError:
        auc = 0.5
        micro_auc = 0.5
        macro_auc = 0.5

    return acc, (pre, micro_pre, macro_pre), (rec, micro_rec, macro_rec), (f1, micro_f1, macro_f1), (auc, micro_auc, macro_auc)


def mkdir(file_path: str) -> str:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    return file_path


def save_args_to_file(args, file_path):
    args_dict = vars(args)

    with open(file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)


def evaluate(args, task_name, test_dataloader):
    test_accs, (test_pres, test_micro_pres, test_macro_pres), (test_recs, test_micro_recs, test_macro_recs), (test_f1s, test_micro_f1s, test_macro_f1s), (test_aucs, test_micro_aucs, test_macro_aucs) \
        = [], ([], [], []), ([], [], []), ([], [], []), ([], [], [])
    for test_sample in test_dataloader:
        test_labels = test_sample['label']
        with torch.no_grad():
            test_output = np.random.rand(test_labels.shape[0], args.num_labels)

            test_acc, (test_pre, test_micro_pre, test_macro_pre), (test_rec, test_micro_rec, test_macro_rec), (test_f1, test_micro_f1, test_macro_f1), (test_auc, test_micro_auc, test_macro_auc) \
                = get_metrics(test_output, test_labels.numpy())

            test_accs.append(test_acc)
            test_pres.append(test_pre)
            test_micro_pres.append(test_micro_pre)
            test_macro_pres.append(test_macro_pre)
            test_recs.append(test_rec)
            test_micro_recs.append(test_micro_rec)
            test_macro_recs.append(test_macro_rec)
            test_f1s.append(test_f1)
            test_micro_f1s.append(test_micro_f1)
            test_macro_f1s.append(test_macro_f1)
            test_aucs.append(test_auc)
            test_micro_aucs.append(test_micro_auc)
            test_macro_aucs.append(test_macro_auc)
    best_test_result = (
        f'{task_name}\t'
        f'{coloring("test_acc", "green_bg")}:{round(np.mean(test_accs), 4)}\t'
        f'{coloring("test_pre", "yellow_bg")}:{round(np.mean(test_pres), 4)}\t'
        f'test_micro_pre:{round(np.mean(test_micro_pres), 4)}\t'
        f'test_macro_pre:{round(np.mean(test_macro_pres), 4)}\t'
        f'{coloring("test_rec", "blue_bg")}:{round(np.mean(test_recs), 4)}\t'
        f'test_micro_rec:{round(np.mean(test_micro_recs), 4)}\t'
        f'test_macro_rec:{round(np.mean(test_macro_recs), 4)}\t'
        f'{coloring("test_f1", "purple_bg")}:{round(np.mean(test_f1s), 4)}\t'
        f'test_micro_f1:{round(np.mean(test_micro_f1s), 4)}\t'
        f'test_macro_f1:{round(np.mean(test_macro_f1s), 4)}\t'
        f'{coloring("test_auc", "cyan_bg")}:{round(np.mean(test_aucs), 4)}\t'
        f'test_micro_auc:{round(np.mean(test_micro_aucs), 4)}\t'
        f'test_macro_auc:{round(np.mean(test_macro_aucs), 4)}'
    )
    print(best_test_result)


def parse_args():
    parser = argparse.ArgumentParser(description='Random-based SE Classifier')

    parser.add_argument('--task_name', nargs='?', default='random_based_SE_classifier',
                        help='Task name')
    parser.add_argument('--data_dir', nargs='?', default='/home/cuifulai/Projects/CQA/Data/StackExchange',
                        help='Data directory')
    parser.add_argument('--data_name', nargs='?', default='meta.stackoverflow.com',
                        help='Data name')
    parser.add_argument('--model_name', nargs='?', default='gpt-4o-2024-08-06',
                        help='Model name')
    parser.add_argument('--pretrained_model_path', nargs='?', default='/data/cuifulai/PretrainedModel/bert-base-uncased',
                        help='Pretrained model path')
    parser.add_argument('--finetuned_model_path', nargs='?', default='/home/cuifulai/Projects/CQA/Model/Baselines/Ablation/RST/FinetunedModel/RST_based_classifier-20240914_113410/best_model.pth',
                        help='Finetuned model path')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--hidden_size', type=int, default=768,
                        help='Hidden size')
    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--dropout_prob', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Max length')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--device', nargs='?', default='cuda:1',
                        help='Device')
    parser.add_argument('--freeze', type=bool, default=False,
                        help='Freeze')
    parser.add_argument('--num_labels', type=int, default=2,
                        help='Number of labels')
    parser.add_argument('--limit', type=int, default=0,
                        help='Limit')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold')
    parser.add_argument('--step', type=int, default=1,
                        help='Step to evaluate')
    parser.add_argument('--split', nargs='?', default=[0.8, 0.1, 0.1],
                        help='Split Data into Train, Dev, Test')
    parser.add_argument('--is_from_finetuned', type=bool, default=True,
                        help='Is from finetuned')
    parser.add_argument('--is_train', type=bool, default=True,
                        help='Is train')
    parser.add_argument('--gum_types', nargs='?', default=None,
                        help='GUM types')

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

    # all_dataset = SEDataset(
    #     tokenizer,
    #     args.data_dir,
    #     args.data_name,
    #     limit=args.limit,
    #     threshold=args.threshold,
    #     mode='All',
    #     max_length=args.max_length
    # )

    all_dataset = AnnotatedSEDataset(
        tokenizer,
        args.data_dir,
        args.data_name,
        args.model_name,
        limit=args.limit,
        mode='All',
        max_length=args.max_length
    )
    train_dataset, dev_dataset, test_dataset = random_split(all_dataset, args.split)
    all_dataloader = DataLoader(all_dataset, batch_size=args.batch_size, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    evaluate(args, args.task_name, test_dataloader)


if __name__ == '__main__':
    main()
