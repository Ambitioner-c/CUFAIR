# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/10/28 15:57
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
from transformers import set_seed, AutoTokenizer, BertModel

from Model.LSTM.SQACILSTM import SQACILSTMModel
from Model.Baselines.Ablation.CommunitySupport.DataLoader.Dataset import CSDataset

from warnings import simplefilter

from Model.Unit.function import mkdir, save_args_to_file, ignore_warning
from Model.Unit.cprint import coloring, decoloring

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=DeprecationWarning)
ignore_warning(name="transformers")


class RegressionModel(nn.Module):
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
        super(RegressionModel, self).__init__()
        self.device = device

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        for p in self.bert.parameters():
            p.data = p.data.contiguous()

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.reduction_layer = nn.Linear(bert_hidden_size, hidden_size)

        self.sqacilstm = SQACILSTMModel(
            question_size=hidden_size,
            answer_size=hidden_size,
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=hidden_size,
            num_attention_heads=num_attention_heads,
            is_peephole=is_peephole,
            ci_mode=ci_mode,
        )

        self.community_support_layer = nn.Linear(hidden_size, num_labels)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs: dict):
        text_left = torch.stack([x.to(self.device) for x in inputs['text_left']], dim=0)        # torch.Size([batch_size, max_length])
        text_right = torch.stack([x.to(self.device) for x in inputs['text_right']], dim=0)      # torch.Size([batch_size, max_length])
        comment = torch.stack([x.to(self.device) for x in inputs['comment']], dim=0)            # torch.Size([batch_size, max_sequence_length, max_length])
        ping = torch.stack([x.to(self.device) for x in inputs['ping']], dim=0)                  # torch.Size([batch_size, max_sequence_length])

        bert_output_left = self.reduction_layer(self.bert(text_left)['pooler_output'])          # torch.Size([batch_size, hidden_size])
        bert_output_right = self.reduction_layer(self.bert(text_right)['pooler_output'])        # torch.Size([batch_size, hidden_size])

        bert_output_comment = []
        for j in range(len(comment)):
            bert_output_comment.append(self.reduction_layer(self.bert(comment[j])['pooler_output']))
        bert_output_comment = torch.stack(bert_output_comment, dim=0)                           # torch.Size([batch_size, max_sequence_length, hidden_size])

        community_support = self.sqacilstm(bert_output_left.unsqueeze(1), bert_output_right.unsqueeze(1), bert_output_comment, ping
                                           )[:, -1, :]                                          # torch.Size([batch_size, hidden_size])
        outputs = self.community_support_layer(community_support)                               # torch.Size([batch_size, num_labels])

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
    loss_function = nn.MSELoss()

    best_model = None
    best_loss = float('inf')

    n = 0
    for epoch in range(epochs):
        for train_sample in tqdm(train_dataloader):
            train_labels = train_sample['label']

            optimizer.zero_grad()

            train_outputs = model(train_sample)

            train_loss = loss_function(input=train_outputs, target=train_labels.to(device))
            train_loss.backward()

            optimizer.step()

            temp_train_result = (f'{task_name}\t'
                                 f'epoch/epochs:{epoch + 1}/{epochs}\t'
                                 f'{coloring("train_loss", "red_bg")}:{np.mean(train_loss.item())}')
            with open(mkdir(temp_train_tsv), 'a' if os.path.exists(temp_train_tsv) else 'w') as f:
                f.write(decoloring(temp_train_result) + '\n')

            if n % step == 0:
                dev_losses = []
                for dev_sample in dev_dataloader:
                    dev_labels = dev_sample['label']
                    with torch.no_grad():
                        dev_outputs = model(dev_sample)

                        dev_loss = loss_function(input=dev_outputs, target=dev_labels.to(device))
                        dev_losses.append(dev_loss.item())
                temp_dev_result = (
                    f'{task_name}\t'
                    f'epoch/epochs:{epoch + 1}/{epochs}\t'
                    f'{coloring("dev_loss", "red_bg")}:{round(np.mean(dev_losses), 4)}'
                )
                with open(mkdir(temp_dev_tsv), 'a' if os.path.exists(temp_dev_tsv) else 'w') as f:
                    f.write(decoloring(temp_dev_result) + '\n')
                print(temp_dev_result)

                if np.mean(dev_losses) < best_loss:
                    best_loss = np.mean(dev_losses)
                    best_model = copy.deepcopy(model)
            n += 1

    torch.save(best_model.state_dict(), mkdir(finetuned_model_path))
    best_model.bert.save_pretrained(mkdir(finetuned_bert_model_path))
    best_dev_result = (
        f'{coloring("best_loss", "red_bg")}:{best_loss}'
    )
    with open(mkdir(best_dev_tsv), 'a' if os.path.exists(best_dev_tsv) else 'w') as f:
        f.write(decoloring(best_dev_result) + '\n')
    print(best_dev_result)

    print(f'{coloring("Finetuned model path", "red_bg")}: {finetuned_model_path}')
    print(f'{coloring("Finetuned bert model path", "green_bg")}: {finetuned_bert_model_path}')

    return best_model, timestamp


def parse_args():
    parser = argparse.ArgumentParser(description='SQACI-based Community Support Regression')
    parser.add_argument('--task_name', nargs='?', default='SQACI_CS_Regression',
                        help='Task name')

    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--bert_hidden_size', type=int, default=768,
                        help='Bert hidden size')
    parser.add_argument('--ci_mode', nargs='?', default='all',
                        help='CI Mode')
    parser.add_argument('--data_dir', nargs='?', default='/home/cuifulai/Projects/CQA/Data/StackExchange',
                        help='Data directory')
    parser.add_argument('--data_name', nargs='?', default='meta.stackoverflow.com',
                        help='Data name')
    parser.add_argument('--device', nargs='?', default='cuda:1',
                        help='Device')
    parser.add_argument('--dropout_prob', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--finetuned_model_path', nargs='?', default='./FinetunedModel/Our_model-20241004_191930/best_model.pth',
                        help='Finetuned model path')
    parser.add_argument('--fold', type=int, default=1,
                        help='Fold')
    parser.add_argument('--freeze', type=bool, default=False,
                        help='Freeze')
    parser.add_argument('--hidden_size', type=int, default=108,
                        help='Hidden size')
    parser.add_argument('--is_from_finetuned', type=bool, default=False,
                        help='Is from finetuned')
    parser.add_argument('--is_peephole', type=bool, default=False,
                        help='Is peephole')
    parser.add_argument('--is_train', type=bool, default=True,
                        help='Is train')
    parser.add_argument('--limit', nargs='?', default=[0, 0, 0],
                        help='Limit')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Max length')
    parser.add_argument('--seq_length', type=int, default=5,
                        help='Max sequence length')
    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--num_labels', type=int, default=1,
                        help='Number of labels')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='Number of layers')
    parser.add_argument('--pretrained_model_path', nargs='?', default='/data/cuifulai/PretrainedModel/bert-base-uncased',
                        help='Pretrained model path')
    parser.add_argument('--save_test', type=bool, default=False,
                        help='Save test')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--step', type=int, default=1,
                        help='Step')

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

    train_dataset = CSDataset(
        tokenizer=tokenizer,
        data_name=args.data_name,
        data_dir=args.data_dir,
        limit=args.limit[0],
        fold=args.fold,
        mode='train',
        max_length=args.max_length,
        seq_length=args.seq_length,
    )
    test_dataset = CSDataset(
        tokenizer=tokenizer,
        data_name=args.data_name,
        data_dir=args.data_dir,
        limit=args.limit[0],
        fold=args.fold,
        mode='test',
        max_length=args.max_length,
        seq_length=args.seq_length,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = RegressionModel(
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

    if args.is_from_finetuned:
        model.load_state_dict(torch.load(args.finetuned_model_path))
    if args.is_train:
        train(args, args.task_name, model, train_dataloader, test_dataloader, args.epochs, args.lr, device, args.step)


if __name__ == '__main__':
    main()
