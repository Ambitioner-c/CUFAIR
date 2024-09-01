# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/8/16 19:53
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
    BertModel,
    AutoConfig
)

from Model.Baselines.Ablation.RST.DataLoader.Dataset import RSTDataset
from Model.Unit.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from Model.Unit.modeling_bert import (
    BertSelfAttention,
)


from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


class RSTModel(nn.Module):
    def __init__(self, freeze=False, pretrained_model_path=None, device=None, num_labels=2):
        super(RSTModel, self).__init__()
        self.device = device

        self.bert = BertModel.from_pretrained(pretrained_model_path)
        for p in self.bert.parameters():
            p.data = p.data.contiguous()

        self.config = AutoConfig.from_pretrained(pretrained_model_path)

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.self_attention = BertSelfAttention(
            self.config.hidden_size,
            self.config.num_attention_heads,
            self.config.attention_probs_dropout_prob
        )

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.classifier = nn.Linear(self.config.hidden_size * 2, num_labels)

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
    rec = recall_score(input, target, average='weighted')
    f1 = f1_score(input, target, average='weighted')

    return acc, pre, rec, f1


def train(task_name, model, train_dataloader, dev_dataloader, epochs, lr, device, step):
    temp_train_csv = f'./Result/Temp/{task_name}-{datetime.now().strftime("%Y%m%d_%H%M%S")}-train.csv'
    temp_dev_csv = f'./Result/Temp/{task_name}-{datetime.now().strftime("%Y%m%d_%H%M%S")}-dev.csv'
    finetuned_model_path = f'./FinetunedModel/{task_name}-{datetime.now().strftime("%Y%m%d_%H%M%S")}/bert-base-uncased'

    model.to(device)

    optimizer = Adam(model.parameters(), lr)
    loss_function = nn.CrossEntropyLoss()

    best_model = None
    best_loss = float('inf')
    best_acc, best_pre, best_rec, best_f1 = -1, -1, -1, -1

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
                                 f'train_loss:{np.mean(train_loss.item())}')
            with open(temp_train_csv, 'a' if os.path.exists(temp_train_csv) else 'w') as f:
                f.write(temp_train_result + '\n')
            # print(temp_train_result)

            if n % step == 0:
                dev_losses = []
                dev_accs, dev_pres, dev_recs, dev_f1s = [], [], [], []
                for dev_sample in dev_dataloader:
                    dev_labels = dev_sample['label']
                    with torch.no_grad():
                        dev_output = model(dev_sample)

                        dev_loss = loss_function(input=dev_output, target=dev_labels.view(-1).to(device))
                        dev_acc, dev_pre, dev_rec, dev_f1 = get_metrics(dev_output.cpu().numpy(), dev_labels.cpu().numpy())

                        dev_losses.append(dev_loss.item())
                        dev_accs.append(dev_acc)
                        dev_pres.append(dev_pre)
                        dev_recs.append(dev_rec)
                        dev_f1s.append(dev_f1)
                temp_dev_result = (f'{task_name}\t'
                                   f'epoch/epochs:{epoch + 1}/{epochs}\t'
                                   f'dev_loss:{round(np.mean(dev_losses), 4)}\t'
                                   f'dev_acc:{round(np.mean(dev_accs), 4)}\t'
                                   f'dev_pre:{round(np.mean(dev_pres), 4)}\t'
                                   f'dev_rec:{round(np.mean(dev_recs), 4)}\t'
                                   f'dev_f1:{round(np.mean(dev_f1s), 4)}\t')
                with open(temp_dev_csv, 'a' if os.path.exists(temp_dev_csv) else 'w') as f:
                    f.write(temp_dev_result + '\n')
                print(temp_dev_result)

                if np.mean(dev_losses) < best_loss:
                    best_loss = np.mean(dev_losses)
                    best_acc = np.mean(dev_accs)
                    best_pre = np.mean(dev_pres)
                    best_rec = np.mean(dev_recs)
                    best_f1 = np.mean(dev_f1s)
                    best_model = copy.deepcopy(model)
            n += 1

    best_model.bert.save_pretrained(finetuned_model_path)
    print(f'Best loss:{best_loss}\t'
          f'Best acc:{best_acc}\t'
          f'Best pre:{best_pre}\t'
          f'Best rec:{best_rec}\t'
          f'Best f1:{best_f1}\t')
    print(f'Finetuned model path: {finetuned_model_path}')

    return best_model


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
    parser = argparse.ArgumentParser(description='RST-based Classifier')

    parser.add_argument('--task_name', nargs='?', default='RST',
                        help='Task name')
    parser.add_argument('--data_dir', nargs='?', default='/home/cuifulai/Projects/CQA/Data/RST/GUM',
                        help='Data directory')
    parser.add_argument('--pretrained_model_path', nargs='?', default='/data/cuifulai/PretrainedModel/bert-base-uncased',
                        help='Pretrained model path')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Max length')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--device', nargs='?', default='cuda:0',
                        help='Device')
    parser.add_argument('--freeze', type=bool, default=False,
                        help='Freeze')
    parser.add_argument('--num_labels', type=int, default=3,
                        help='Number of labels')
    parser.add_argument('--step', type=int, default=1,
                        help='Step')

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

    train_dataset = RSTDataset(tokenizer, args.data_dir, mode='Train', max_length=args.max_length)
    dev_dataset = RSTDataset(tokenizer, args.data_dir, mode='Dev', max_length=args.max_length)
    test_dataset = RSTDataset(tokenizer, args.data_dir, mode='Test', max_length=args.max_length)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    model = RSTModel(
        freeze=args.freeze,
        pretrained_model_path=args.pretrained_model_path,
        device=device,
        num_labels=args.num_labels
    )

    best_model = train(args.task_name, model, train_dataloader, dev_dataloader, args.epochs, args.lr, device, args.step)

    evaluate(args.task_name, best_model, test_dataloader)


if __name__ == '__main__':
    main()
