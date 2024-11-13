# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/10/27 14:09
import argparse
import copy
import os
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertModel, set_seed, AutoTokenizer

from Model4AfD.DataLoader.Dataset import OurDataset
from Model.LSTM.LSTM import LSTMModel

from warnings import simplefilter

from Model.Unit.cprint import coloring, decoloring
from Model.Unit.function import mkdir, save_args_to_file, ignore_warning
from Model.Unit.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=DeprecationWarning)
ignore_warning(name="transformers")


class GuptaModel(nn.Module):
    def __init__(
            self,
            freeze: bool = False,
            pretrained_model_name_or_path: str = 'bert-base-uncased',
            device: torch.device = torch.device('cuda:0'),
            hidden_size: int = 108,
            bert_hidden_size: int = 768,
            dropout_prob: float = 0.1,
            num_layers: int = 1,
            num_attention_heads: int = 12,
            num_labels: int = 2,
            is_peephole: bool = False,
            ci_mode: str = 'all',
    ):
        super(GuptaModel, self).__init__()
        self.device = device

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        for p in self.bert.parameters():
            p.data = p.data.contiguous()

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.reduction_layer = nn.Linear(bert_hidden_size, hidden_size)

        self.lstm = LSTMModel(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=hidden_size,
        )

        self.consensus_layer = nn.Linear(hidden_size * 2, num_labels)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs: dict):
        text_right = torch.stack([x.to(self.device) for x in inputs['text_right']], dim=0)          # torch.Size([batch_size, max_length])
        comment = torch.stack([x.to(self.device) for x in inputs['comment']], dim=0)                # torch.Size([batch_size, seq_length, max_length])

        bert_output_right = self.reduction_layer(self.bert(text_right)['pooler_output'])            # torch.Size([batch_size, hidden_size])

        bert_output_comment = []
        for j in range(len(comment)):
            bert_output_comment.append(self.reduction_layer(self.bert(comment[j])['pooler_output']))
        bert_output_comment = torch.stack(bert_output_comment, dim=0)                               # torch.Size([batch_size, seq_length, hidden_size])

        # Consensus
        consensus = self.lstm(bert_output_comment)[:, -1, :]                                        # torch.Size([batch_size, hidden_size])

        outputs = self.consensus_layer(
            torch.cat([bert_output_right, consensus], dim=-1))                               # torch.Size([batch_size, 2])

        return outputs


def train(args, task_name, model, train_dataloader, dev_dataloader, epochs, lr, device, step):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    args_path = f'./Result/Temp/{task_name}-{timestamp}/args.json'
    temp_train_tsv = f'./Result/Temp/{task_name}-{timestamp}/train.tsv'
    temp_dev_tsv = f'./Result/Temp/{task_name}-{timestamp}/dev.tsv'
    best_dev_tsv = f'./Result/Temp/{task_name}-{timestamp}/best_dev.tsv'
    finetuned_model_path = f'./FinetunedModel/{task_name}-{timestamp}/best_model.pth'
    finetuned_bert_model_path = f'./FinetunedModel/{task_name}-{timestamp}/bert-base-uncased'

    save_args_to_file(args, mkdir(args_path))

    optimizer = Adam(model.parameters(), lr=lr)
    loss_function = nn.CrossEntropyLoss()

    best_model = None
    best_loss = float('inf')
    best_acc, best_pre, best_rec, best_f1, best_auc = -1, -1, -1, -1, -1

    n = 0
    for epoch in range(epochs):
        for train_sample in tqdm(train_dataloader):
            train_labels = train_sample['label']

            optimizer.zero_grad()

            train_outputs = model(train_sample)

            train_loss = loss_function(input=train_outputs, target=train_labels.view(-1).to(device))
            train_loss.backward()

            optimizer.step()

            temp_train_result = (f'{task_name}\t'
                                 f'epoch/epochs:{epoch + 1}/{epochs}\t'
                                 f'{coloring("train_loss", "red_bg")}:{train_loss.item()}')
            with open(mkdir(temp_train_tsv), 'a' if os.path.exists(temp_train_tsv) else 'w') as f:
                f.write(decoloring(temp_train_result) + '\n')

            if n % step == 0:
                dev_losses = []
                dev_accs, dev_pres, dev_recs, dev_f1s, dev_aucs = [], [], [], [], []
                for dev_sample in dev_dataloader:
                    dev_labels = dev_sample['label']

                    with torch.no_grad():
                        dev_outputs = model(dev_sample)

                        dev_loss = loss_function(input=dev_outputs, target=dev_labels.view(-1).to(device))
                        dev_acc, dev_pre, dev_rec, dev_f1, dev_auc = get_metrics(dev_outputs.cpu().numpy(), dev_labels.cpu().numpy())

                        dev_losses.append(dev_loss.item())
                        dev_accs.append(dev_acc)
                        dev_pres.append(dev_pre)
                        dev_recs.append(dev_rec)
                        dev_f1s.append(dev_f1)
                        dev_aucs.append(dev_auc)
                temp_dev_result = (
                    f'{task_name}\t'
                    f'epoch/epochs:{epoch + 1}/{epochs}\t'
                    f'{coloring("dev_loss", "red_bg")}:{round(np.mean(dev_losses), 4)}\t'
                    f'{coloring("dev_acc", "green_bg")}:{round(np.mean(dev_accs), 4)}\t'
                    f'{coloring("dev_pre", "yellow_bg")}:{round(np.mean(dev_pres), 4)}\t'
                    f'{coloring("dev_rec", "blue_bg")}:{round(np.mean(dev_recs), 4)}\t'
                    f'{coloring("dev_f1", "purple_bg")}:{round(np.mean(dev_f1s), 4)}\t'
                    f'{coloring("dev_auc", "cyan_bg")}:{round(np.mean(dev_aucs), 4)}'
                )
                with open(mkdir(temp_dev_tsv), 'a' if os.path.exists(temp_dev_tsv) else 'w') as f:
                    f.write(decoloring(temp_dev_result) + '\n')
                print(temp_dev_result)

                if np.mean(dev_losses) < best_loss:
                    best_loss = np.mean(dev_losses)
                    best_acc = np.mean(dev_accs)
                    best_pre = np.mean(dev_pres)
                    best_rec = np.mean(dev_recs)
                    best_f1 = np.mean(dev_f1s)
                    best_auc = np.mean(dev_aucs)
                    best_model = copy.deepcopy(model)
            n += 1

    torch.save(best_model.state_dict(), mkdir(finetuned_model_path))
    best_model.bert.save_pretrained(mkdir(finetuned_bert_model_path))
    best_dev_result = (
        f'{coloring("best_loss", "red_bg")}:{best_loss}\t'
        f'{coloring("best_acc", "green_bg")}:{best_acc}\t'
        f'{coloring("best_pre", "yellow_bg")}:{best_pre}\t'
        f'{coloring("best_rec", "blue_bg")}:{best_rec}\t'
        f'{coloring("best_f1", "purple_bg")}:{best_f1}\t'
        f'{coloring("best_auc", "cyan_bg")}:{best_auc}'
    )
    with open(mkdir(best_dev_tsv), 'a' if os.path.exists(best_dev_tsv) else 'w') as f:
        f.write(decoloring(best_dev_result) + '\n')
    print(best_dev_result)

    print(f'{coloring("Finetuned model path", "red_bg")}: {finetuned_model_path}')
    print(f'{coloring("Finetuned bert model path", "green_bg")}: {finetuned_bert_model_path}')

    return best_model, timestamp


def get_metrics(input: np.array, target: np.array):
    acc = accuracy_score(input, target)
    pre = precision_score(input, target, average='weighted')
    rec = recall_score(input, target, average='weighted')
    f1 = f1_score(input, target, average='weighted')
    try:
        auc = roc_auc_score(input, target, average='weighted')
    except ValueError:
        auc = 0.5

    return acc, pre, rec, f1, auc


def evaluate(args, task_name, model, test_dataloader, timestamp, save_test):
    model.eval()

    test_accs, test_pres, test_recs, test_f1s, test_aucs = [], [], [], [], []
    for test_sample in test_dataloader:
        test_labels = test_sample['label']

        with torch.no_grad():
            test_outputs = model(test_sample)

            test_acc, test_pre, test_rec, test_f1, test_auc = get_metrics(test_outputs.cpu().numpy(), test_labels.cpu().numpy())

            test_accs.append(test_acc)
            test_pres.append(test_pre)
            test_recs.append(test_rec)
            test_f1s.append(test_f1)
            test_aucs.append(test_auc)
    best_test_result = (
        f'{task_name}\t'
        f'{coloring("test_acc", "green_bg")}:{round(np.mean(test_accs), 4)}\t'
        f'{coloring("test_pre", "yellow_bg")}:{round(np.mean(test_pres), 4)}\t'
        f'{coloring("test_rec", "blue_bg")}:{round(np.mean(test_recs), 4)}\t'
        f'{coloring("test_f1", "purple_bg")}:{round(np.mean(test_f1s), 4)}\t'
        f'{coloring("test_auc", "cyan_bg")}:{round(np.mean(test_aucs), 4)}'
    )
    if args.is_train or save_test:
        best_test_tsv = f'./Result/Temp/{task_name}-{timestamp}/best_test.tsv'
        with open(mkdir(best_test_tsv), 'a' if os.path.exists(best_test_tsv) else 'w') as f:
            f.write(decoloring(best_test_result) + '\n')
    print(best_test_result)
    print(f'{np.mean(test_accs)}\t{np.mean(test_pres)}\t{np.mean(test_recs)}\t{np.mean(test_f1s)}\t{np.mean(test_aucs)}')


def parse_args():
    parser = argparse.ArgumentParser(description='Gupta Model for AfD')
    parser.add_argument('--task_name', nargs='?', default='Gupta4AfD',
                        help='Task name')

    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size')
    parser.add_argument('--bert_hidden_size', type=int, default=768,
                        help='Bert hidden size')
    parser.add_argument('--ci_mode', nargs='?', default='all',
                        help='CI Mode')
    parser.add_argument('--data_dir', nargs='?', default='/home/cuifulai/Projects/CQA/Data/WikiPedia',
                        help='Data directory')
    parser.add_argument('--data_name', nargs='?', default='AfD',
                        help='Data name')
    parser.add_argument('--device', nargs='?', default='cuda:1',
                        help='Device')
    parser.add_argument('--dropout_prob', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--finetuned_model_path', nargs='?', default='./FinetunedModel/XXX/best_model.pth',
                        help='Finetuned model path')
    parser.add_argument('--freeze', type=bool, default=True,
                        help='Freeze')
    parser.add_argument('--hidden_size', type=int, default=108,
                        help='Hidden size')
    parser.add_argument('--is_from_finetuned', type=bool, default=False,
                        help='Is from finetuned')
    parser.add_argument('--is_peephole', type=bool, default=False,
                        help='Is peephole')
    parser.add_argument('--is_train', type=bool, default=True,
                        help='Is train')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Max length')
    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--num_labels', type=int, default=2,
                        help='Number of labels')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of layers')
    parser.add_argument('--pretrained_model_path', nargs='?', default='/data/cuifulai/PretrainedModel/bert-base-uncased',
                        help='Pretrained model path')
    parser.add_argument('--save_test', type=bool, default=False,
                        help='Save test')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--seq_length', type=int, default=10,
                        help='Max sequence length')
    parser.add_argument('--step', type=int, default=1,
                        help='Step')

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

    if args.is_train:
        train_dataset = OurDataset(
            tokenizer=tokenizer,
            data_dir=args.data_dir,
            data_name=args.data_name,
            max_length=args.max_length,
            mode='train',
            seq_length=args.seq_length,
        )
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        train_dataloader = None

    dev_dataset = OurDataset(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        data_name=args.data_name,
        max_length=args.max_length,
        mode='dev',
        seq_length=args.seq_length,
    )
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)

    test_dataset = OurDataset(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        data_name=args.data_name,
        max_length=args.max_length,
        mode='test',
        seq_length=args.seq_length,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = GuptaModel(
        freeze=args.freeze,
        pretrained_model_name_or_path=args.pretrained_model_path,
        device=device,
        hidden_size=args.hidden_size,
        bert_hidden_size=args.bert_hidden_size,
        dropout_prob=args.dropout_prob,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_labels=args.num_labels,
        is_peephole=args.is_peephole,
        ci_mode=args.ci_mode,
    ).to(device)

    timestamp = None
    if args.is_from_finetuned:
        model.load_state_dict(torch.load(args.finetuned_model_path))
    if args.is_train:
        model, timestamp = train(args, args.task_name, model, train_dataloader, dev_dataloader, args.epochs, args.lr, device, args.step)

    evaluate(args, args.task_name, model, test_dataloader, timestamp, args.save_test)


if __name__ == '__main__':
    main()
