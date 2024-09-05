# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/8/26 10:56
import argparse
import copy
import json
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import os.path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

import transformers
from tqdm import tqdm

transformers.logging.set_verbosity_error()

from transformers import (
    set_seed,
    AutoTokenizer,
    BertModel
)

from Model.Baselines.Ablation.StackExchange.DataLoader.Dataset import (
    SEDataset,
    AnnotatedSEDataset
)
from Model.Unit.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from Model.Unit.modeling_bert import (
    BertSelfAttention,
)
from Model.Unit.cprint import coloring, decoloring

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


class RSTModel(nn.Module):
    def __init__(
            self,
            freeze: bool = False,
            pretrained_model_name_or_path: str = 'bert-base-uncased',
            device: torch.device= torch.device('cuda:0'),
            num_labels: int = 3,
            hidden_size: int = 768,
            num_attention_heads: int = 12,
            dropout_prob: float = 0.1
    ):
        super(RSTModel, self).__init__()
        self.device = device

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        for p in self.bert.parameters():
            p.data = p.data.contiguous()

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.self_attention = BertSelfAttention(
            hidden_size,
            num_attention_heads,
            dropout_prob
        )

        self.dropout = nn.Dropout(dropout_prob)

        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(self, sample: pd.DataFrame):
        x_left = sample['left'].to(self.device)
        x_right = sample['right'].to(self.device)

        left_init = self.dropout(
            self.bert(x_left)['pooler_output']
        )
        right_init = self.dropout(
            self.bert(x_right)['pooler_output']
        )

        left_right = self.dropout(
            self.self_attention(
                torch.stack([left_init, right_init], dim=1)
            )[0]
        )

        outputs = self.classifier(torch.flatten(left_right, start_dim=1))

        return outputs


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

    return acc, (pre, micro_pre, macro_pre), (rec, micro_rec, macro_rec), (f1, micro_f1, macro_f1)


def mkdir(file_path: str) -> str:
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    return file_path


def save_args_to_file(args, file_path):
    args_dict = vars(args)

    with open(file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)


def train(args, task_name, model, train_dataloader, dev_dataloader, epochs, lr, device, step):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    args_path = f'./Result/Temp/{task_name}-{timestamp}/args.json'
    temp_train_tsv = f'./Result/Temp/{task_name}-{timestamp}/train.tsv'
    temp_dev_tsv = f'./Result/Temp/{task_name}-{timestamp}/dev.tsv'
    best_dev_tsv = f'./Result/Temp/{task_name}-{timestamp}/best_dev.tsv'
    finetuned_model_path = f'./FinetunedModel/{task_name}-{timestamp}/best_model.pth'
    finetuned_bert_model_path = f'./FinetunedModel/{task_name}-{timestamp}/bert-base-uncased'

    save_args_to_file(args, mkdir(args_path))

    optimizer = Adam(model.parameters(), lr)
    loss_function = nn.CrossEntropyLoss()

    best_model = None
    best_loss = float('inf')
    best_acc, (best_pre, best_micro_pre, best_macro_pre), (best_rec, best_micro_rec, best_macro_rec), (best_f1, best_micro_f1, best_macro_f1) \
        = -1, (-1, -1, -1), (-1, -1, -1), (-1, -1, -1)

    n = 0
    for epoch in range(epochs):
        for train_sample in tqdm(train_dataloader):
            train_labels = train_sample['label']

            optimizer.zero_grad()

            train_output = model(train_sample)

            train_loss = loss_function(input=train_output, target=train_labels.view(-1).to(device))
            train_loss.backward()

            optimizer.step()

            temp_train_result = (f'{task_name}\t'
                                 f'epoch/epochs:{epoch + 1}/{epochs}\t'
                                 f'{coloring("train_loss", "red_bg")}:{np.mean(train_loss.item())}')
            with open(mkdir(temp_train_tsv), 'a' if os.path.exists(temp_train_tsv) else 'w') as f:
                f.write(decoloring(temp_train_result) + '\n')
            # print(temp_train_result)

            if n % step == 0:
                dev_losses = []
                dev_accs, (dev_pres, dev_micro_pres, dev_macro_pres), (dev_recs, dev_micro_recs, dev_macro_recs), (dev_f1s, dev_micro_f1s, dev_macro_f1s) \
                    = [], ([], [], []), ([], [], []), ([], [], [])
                for dev_sample in dev_dataloader:
                    dev_labels = dev_sample['label']
                    with torch.no_grad():
                        dev_output = model(dev_sample)

                        dev_loss = loss_function(input=dev_output, target=dev_labels.view(-1).to(device))
                        dev_acc, (dev_pre, dev_micro_pre, dev_macro_pre), (dev_rec, dev_micro_rec, dev_macro_rec), (dev_f1, dev_micro_f1, dev_macro_f1) \
                            = get_metrics(dev_output.cpu().numpy(), dev_labels.cpu().numpy())

                        dev_losses.append(dev_loss.item())
                        dev_accs.append(dev_acc)
                        dev_pres.append(dev_pre)
                        dev_micro_pres.append(dev_micro_pre)
                        dev_macro_pres.append(dev_macro_pre)
                        dev_recs.append(dev_rec)
                        dev_micro_recs.append(dev_micro_rec)
                        dev_macro_recs.append(dev_macro_rec)
                        dev_f1s.append(dev_f1)
                        dev_micro_f1s.append(dev_micro_f1)
                        dev_macro_f1s.append(dev_macro_f1)
                temp_dev_result = (
                    f'{task_name}\t'
                    f'epoch/epochs:{epoch + 1}/{epochs}\t'
                    f'{coloring("dev_loss", "red_bg")}:{round(np.mean(dev_losses), 4)}\t'
                    f'{coloring("dev_acc", "green_bg")}:{round(np.mean(dev_accs), 4)}\t'
                    f'{coloring("dev_pre", "yellow_bg")}:{round(np.mean(dev_pres), 4)}\t'
                    f'dev_micro_pre:{round(np.mean(dev_micro_pres), 4)}\t'
                    f'dev_macro_pre:{round(np.mean(dev_macro_pres), 4)}\t'
                    f'{coloring("dev_rec", "blue_bg")}:{round(np.mean(dev_recs), 4)}\t'
                    f'dev_micro_rec:{round(np.mean(dev_micro_recs), 4)}\t'
                    f'dev_macro_rec:{round(np.mean(dev_macro_recs), 4)}\t'
                    f'{coloring("dev_f1", "purple_bg")}:{round(np.mean(dev_f1s), 4)}\t'
                    f'dev_micro_f1:{round(np.mean(dev_micro_f1s), 4)}\t'
                    f'dev_macro_f1:{round(np.mean(dev_macro_f1s), 4)}'
                )
                with open(mkdir(temp_dev_tsv), 'a' if os.path.exists(temp_dev_tsv) else 'w') as f:
                    f.write(decoloring(temp_dev_result) + '\n')
                print(temp_dev_result)

                if np.mean(dev_losses) < best_loss:
                    best_loss = np.mean(dev_losses)
                    best_acc = np.mean(dev_accs)
                    best_pre = np.mean(dev_pres)
                    best_micro_pre = np.mean(dev_micro_pres)
                    best_macro_pre = np.mean(dev_macro_pres)
                    best_rec = np.mean(dev_recs)
                    best_micro_rec = np.mean(dev_micro_recs)
                    best_macro_rec = np.mean(dev_macro_recs)
                    best_f1 = np.mean(dev_f1s)
                    best_micro_f1 = np.mean(dev_micro_f1s)
                    best_macro_f1 = np.mean(dev_macro_f1s)
                    best_model = copy.deepcopy(model)
            n += 1

    torch.save(best_model.state_dict(), mkdir(finetuned_model_path))
    best_model.bert.save_pretrained(mkdir(finetuned_bert_model_path))
    best_dev_result = (
        f'{coloring("best_loss", "red_bg")}:{best_loss}\t'
        f'{coloring("best_acc", "green_bg")}:{best_acc}\t'
        f'{coloring("best_pre", "yellow_bg")}:{best_pre}\t'
        f'best_micro_pre:{best_micro_pre}\t'
        f'best_macro_pre:{best_macro_pre}\t'
        f'{coloring("best_rec", "blue_bg")}:{best_rec}\t'
        f'best_micro_rec:{best_micro_rec}\t'
        f'best_macro_rec:{best_macro_rec}\t'
        f'{coloring("best_f1", "purple_bg")}:{best_f1}\t'
        f'best_micro_f1:{best_micro_f1}\t'
        f'best_macro_f1:{best_macro_f1}'
    )
    with open(mkdir(best_dev_tsv), 'a' if os.path.exists(best_dev_tsv) else 'w') as f:
        f.write(decoloring(best_dev_result) + '\n')
    print(best_dev_result)

    print(f'{coloring("Finetuned model path", "red_bg")}: {finetuned_model_path}')
    print(f'{coloring("Finetuned bert model path", "green_bg")}: {finetuned_bert_model_path}')

    return best_model, timestamp


def evaluate(args, task_name, model, test_dataloader, timestamp):
    model.eval()

    test_accs, (test_pres, test_micro_pres, test_macro_pres), (test_recs, test_micro_recs, test_macro_recs), (test_f1s, test_micro_f1s, test_macro_f1s) \
        = [], ([], [], []), ([], [], []), ([], [], [])
    for test_sample in test_dataloader:
        test_labels = test_sample['label']
        with torch.no_grad():
            test_output = model(test_sample)

            test_acc, (test_pre, test_micro_pre, test_macro_pre), (test_rec, test_micro_rec, test_macro_rec), (test_f1, test_micro_f1, test_macro_f1) \
                = get_metrics(test_output.cpu().numpy(), test_labels.cpu().numpy())

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
        f'test_macro_f1:{round(np.mean(test_macro_f1s), 4)}'
    )
    if args.is_train:
        best_test_tsv = f'./Result/Temp/{task_name}-{timestamp}/best_test.tsv'
        with open(mkdir(best_test_tsv), 'a' if os.path.exists(best_test_tsv) else 'w') as f:
            f.write(decoloring(best_test_result) + '\n')
    print(best_test_result)


def parse_args():
    parser = argparse.ArgumentParser(description='RSE-based SE Classifier')

    parser.add_argument('--task_name', nargs='?', default='RSE_based_SE_classifier',
                        help='Task name')
    parser.add_argument('--data_dir', nargs='?', default='/home/cuifulai/Projects/CQA/Data/StackExchange',
                        help='Data directory')
    parser.add_argument('--data_name', nargs='?', default='meta.stackoverflow.com',
                        help='Data name')
    parser.add_argument('--model_name', nargs='?', default='gpt-4o-2024-08-06',
                        help='Model name')
    parser.add_argument('--pretrained_model_path', nargs='?', default='/data/cuifulai/PretrainedModel/bert-base-uncased',
                        help='Pretrained model path')
    parser.add_argument('--finetuned_model_path', nargs='?', default='/home/cuifulai/Projects/CQA/Model/Baselines/Ablation/RST/FinetunedModel/RST_based_classifier-20240905_113024/best_model.pth',
                        help='Finetuned model path')
    parser.add_argument('--epochs', type=int, default=3,
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
    parser.add_argument('--num_labels', type=int, default=3,
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

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = RSTModel(
        freeze=args.freeze,
        pretrained_model_name_or_path=args.pretrained_model_path,
        device=device,
        num_labels=args.num_labels,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        dropout_prob=args.dropout_prob
    ).to(device)

    timestamp = None
    if args.is_from_finetuned:
        model.load_state_dict(torch.load(args.finetuned_model_path))
    if args.is_train:
        model, timestamp = train(args, args.task_name, model, train_dataloader, dev_dataloader, args.epochs, args.lr, device, args.step)

    evaluate(args, args.task_name, model, test_dataloader, timestamp)


if __name__ == '__main__':
    main()
