# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/8/16 19:53
import argparse
import copy
import os.path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import transformers
from tqdm import tqdm, trange

transformers.logging.set_verbosity_error()

from transformers import (
    set_seed,
    AutoTokenizer,
    BertForSequenceClassification,
    AutoConfig
)

from Model.Baselines.Ablation.RST.DataLoader.Dataset import RSTDataset

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


class RSTModel(nn.Module):
    def __init__(self, freeze=False, pretrained_model_path=None, device=None):
        super(RSTModel, self).__init__()
        self.device = device

        self.bert = BertForSequenceClassification.from_pretrained(pretrained_model_path, num_labels=3)
        self.config = AutoConfig.from_pretrained(pretrained_model_path)

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, sample: pd.DataFrame):
        feature = sample['feature'].to(self.device)
        output = self.bert(feature).logits

        return output


def train(task_name, model, train_dataloader, dev_dataloader, epochs, lr, device):
    temp_train_csv = f'./Result/Temp/{task_name}-{datetime.now().strftime("%Y%m%d_%H%M%S")}-train.csv'
    temp_dev_csv = f'./Result/Temp/{task_name}-{datetime.now().strftime("%Y%m%d_%H%M%S")}-dev.csv'
    finetuned_model_path = f'./FinetunedModel/{task_name}-{datetime.now().strftime("%Y%m%d_%H%M%S")}/bert-base-uncased'

    model.to(device)

    optimizer = Adam(model.parameters(), lr)
    loss_function = nn.CrossEntropyLoss()

    best_model = None
    best_loss = float('inf')

    n = 0
    for epoch in trange(epochs):
        train_losses = []
        for train_sample in tqdm(train_dataloader):
            train_labels = train_sample['label']

            optimizer.zero_grad()

            train_output = model(train_sample)

            train_loss = loss_function(input=train_output, target=train_labels.view(-1).to(device))
            train_loss.backward()

            optimizer.step()

            train_losses.append(train_loss.item())
            temp_train_result = f'{task_name}\tepoch/epochs: {epoch + 1}/{epochs}\tloss: {np.mean(train_losses)}'
            with open(temp_train_csv, 'a' if os.path.exists(temp_train_csv) else 'w') as f:
                f.write(temp_train_result + '\n')
            print(temp_train_result)

            if n % 10 == 0:
                dev_losses = []
                for dev_sample in dev_dataloader:
                    dev_labels = dev_sample['label']
                    with torch.no_grad():
                        dev_output = model(dev_sample)

                        dev_loss = loss_function(input=dev_output, target=dev_labels.view(-1).to(device))

                        dev_losses.append(dev_loss.item())
                temp_dev_result = f'{task_name}\tepoch/epochs: {epoch + 1}/{epochs}\tloss: {np.mean(dev_losses)}'
                with open(temp_dev_csv, 'a' if os.path.exists(temp_dev_csv) else 'w') as f:
                    f.write(temp_dev_result + '\n')
                print(temp_dev_result)

                if np.mean(dev_losses) < best_loss:
                    best_loss = np.mean(dev_losses)
                    best_model = copy.deepcopy(model)
            n += 1

    best_model.bert.save_pretrained(finetuned_model_path)
    print(f'Best loss: {best_loss}')
    print(f'Finetuned model path: {finetuned_model_path}')

    return best_model


def evaluate(task_name, model, test_dataloader, device):
    model.eval()

    test_losses = []
    for test_sample in tqdm(test_dataloader):
        test_labels = test_sample['label']
        with torch.no_grad():
            test_output = model(test_sample)

            test_loss = nn.CrossEntropyLoss()(input=test_output, target=test_labels.view(-1).to(device))

            test_losses.append(test_loss.item())
    temp_test_result = f'{task_name}\tloss: {np.mean(test_losses)}'
    print(temp_test_result)


def parse_args():
    parser = argparse.ArgumentParser(description='RST-based Classifier')

    parser.add_argument('--task_name', nargs='?', default='RST',
                        help='Task name')
    parser.add_argument('--data_dir', nargs='?', default='/home/cuifulai/Projects/CQA/Data/RST/GUM',
                        help='Data directory')
    parser.add_argument('--pretrained_model_path', nargs='?', default='/data/cuifulai/PretrainedModel',
                        help='Pretrained model path')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Max length')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--device', nargs='?', default='cuda:0',
                        help='Device')

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
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = RSTModel(
        freeze=False,
        pretrained_model_path=args.pretrained_model_path,
        device=device
    )

    best_model = train(args.task_name, model, train_dataloader, dev_dataloader, args.epochs, args.lr, device)

    evaluate(args.task_name, best_model, test_dataloader, device)


if __name__ == '__main__':
    main()
