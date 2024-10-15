# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/29 19:05
import argparse
import copy
import os
import typing
from datetime import datetime

import numpy as np
import pandas as pd
import spacy
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import BertModel, set_seed, AutoTokenizer

from Losses.RankHingeLoss import RankHingeLoss
from Model.DataLoader.DataLoader import DataLoader
from Model.DataLoader.DataProcessor import OurProcessor
from Model.DataLoader.Dataset import OurDataset
from Model.LSTM.SALSTM import SALSTMModel
from Model.Our.Dimension.ArgumentQuality import ArgumentQuality

from warnings import simplefilter

from Model.Unit.cprint import coloring, decoloring
from Model.Unit.function import mkdir, save_args_to_file, ignore_warning
from Model.Unit.metrics import (
    precision, average_precision, mean_average_precision, mean_reciprocal_rank,
    discounted_cumulative_gain, normalized_discounted_cumulative_gain
)

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=DeprecationWarning)
ignore_warning(name="transformers")


class OurModel(nn.Module):
    def __init__(
            self,
            freeze: bool = False,
            pretrained_model_name_or_path: str = 'bert-base-uncased',
            device: torch.device = torch.device('cuda:0'),
            hidden_size: int = 108,
            bert_hidden_size: int = 768,
            dropout_prob: float = 0.1,
            num_layers: int = 1,
            num_labels: int = 2,
    ):
        super(OurModel, self).__init__()
        self.device = device

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        for p in self.bert.parameters():
            p.data = p.data.contiguous()

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.reduction_layer = nn.Linear(bert_hidden_size, hidden_size)

        self.feature_norm = nn.BatchNorm1d(44)

        self.relevancy_layer = nn.Linear(hidden_size * 2, 20)

        self.attention_lstm = SALSTMModel(
            attention_size=hidden_size,
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=hidden_size,
        )

        self.credibility_layer = nn.Linear(hidden_size, 64)

        self.argument_quality_layer = nn.Linear(64, num_labels)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs: dict):
        text_left = torch.stack([x.to(self.device) for x in inputs['text_left']], dim=0)            # torch.Size([batch_size, max_length])
        text_right = torch.stack([x.to(self.device) for x in inputs['text_right']], dim=0)               # torch.Size([batch_size, max_sequence_length, max_length])
        feature = torch.stack([torch.tensor(x).to(self.device) for x in inputs['feature']], dim=0)  # torch.Size([batch_size, 44])
        try:
            feature = self.feature_norm(feature)                                                        # torch.Size([batch_size, 44])
        except ValueError:
            feature = feature

        bert_output_left = self.reduction_layer(self.bert(text_left)['pooler_output'])              # torch.Size([batch_size, hidden_size])
        bert_output_right = self.reduction_layer(self.bert(text_right)['pooler_output'])            # torch.Size([batch_size, hidden_size])

        # AQ
        # Relevancy
        relevancy = self.relevancy_layer(
            torch.cat([bert_output_left, bert_output_right], dim=-1))                        # torch.Size([batch_size, 20])
        argument_quality = torch.cat([relevancy, feature], dim=-1)                           # torch.Size([batch_size, 64])

        outputs = self.argument_quality_layer(argument_quality)[:, 1].unsqueeze(1)                  # torch.Size([batch_size, 1])

        return outputs


def train(args, task_name, model, train_dataloader, dev_dataloader, epochs, lr, step):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    args_path = f'./Result/Temp/{task_name}-{timestamp}/args.json'
    temp_train_tsv = f'./Result/Temp/{task_name}-{timestamp}/train.tsv'
    temp_dev_tsv = f'./Result/Temp/{task_name}-{timestamp}/dev.tsv'
    best_dev_tsv = f'./Result/Temp/{task_name}-{timestamp}/best_dev.tsv'
    finetuned_model_path = f'./FinetunedModel/{task_name}-{timestamp}/best_model.pth'
    finetuned_bert_model_path = f'./FinetunedModel/{task_name}-{timestamp}/bert-base-uncased'

    save_args_to_file(args, mkdir(args_path))

    optimizer = Adam(model.parameters(), lr=lr)
    loss_function = RankHingeLoss(
        num_neg=args.num_neg,
    )

    best_model = None
    (best_p_1, best_p_3, best_p_5, best_ap, best_map), best_mrr, (best_dcg_1, best_dcg_3, best_dcg_5), (best_ndcg_1, best_ndcg_3, best_ndcg_5) = (
        (-1, -1, -1, -1, -1), -1, (-1, -1, -1), (-1, -1, -1))

    n = 0
    for epoch in range(epochs):
        for train_sample in tqdm(train_dataloader):
            train_inputs, _ = train_sample

            optimizer.zero_grad()

            train_outputs = model(train_inputs)

            train_loss = loss_function(train_outputs)
            train_loss.backward()

            optimizer.step()

            temp_train_result = (f'{task_name}\t'
                                 f'epoch/epochs:{epoch + 1}/{epochs}\t'
                                 f'{coloring("train_loss", "red_bg")}:{train_loss.item()}')
            with open(mkdir(temp_train_tsv), 'a' if os.path.exists(temp_train_tsv) else 'w') as f:
                f.write(decoloring(temp_train_result) + '\n')

            if n % step == 0:
                predictions = []
                for dev_sample in dev_dataloader:
                    dev_inputs, _ = dev_sample
                    with torch.no_grad():
                        dev_outputs = model(dev_inputs).detach().cpu()
                        predictions.append(dev_outputs)
                y_pred = torch.cat(predictions, dim=0).numpy()
                y_true = dev_dataloader.label
                id_left = dev_dataloader.id_left
                (p_1, p_3, p_5, ap, map_), mrr, (dcg_1, dcg_3, dcg_5), (ndcg_1, ndcg_3, ndcg_5) = (
                    eval_ranking_metrics_on_data_frame(id_left, y_true, y_pred.squeeze(axis=-1)))
                temp_dev_result = (
                    f'{task_name}\t'
                    f'epoch/epochs:{epoch + 1}/{epochs}\t'
                    f'Ranking:\t'
                    f'{coloring("p@1", "red_bg")}:{round(p_1, 4)}\t'
                    f'{coloring("p@3", "red_bg")}:{round(p_3, 4)}\t'
                    f'{coloring("p@5", "red_bg")}:{round(p_5, 4)}\t'
                    f'{coloring("ap", "green_bg")}:{round(ap, 4)}\t'
                    f'{coloring("map", "yellow_bg")}:{round(map_, 4)}\t'
                    f'{coloring("mrr", "blue_bg")}:{round(mrr, 4)}\t'
                    f'dcg@1:{round(dcg_1, 4)}\t'
                    f'dcg@3:{round(dcg_3, 4)}\t'
                    f'dcg@5:{round(dcg_5, 4)}\t'
                    f'{coloring("ndcg@1", "purple_bg")}:{round(ndcg_1, 4)}\t'
                    f'{coloring("ndcg@3", "purple_bg")}:{round(ndcg_3, 4)}\t'
                    f'{coloring("ndcg@5", "purple_bg")}:{round(ndcg_5, 4)}'
                )
                with open(mkdir(temp_dev_tsv), 'a' if os.path.exists(temp_dev_tsv) else 'w') as f:
                    f.write(decoloring(temp_dev_result) + '\n')
                print(temp_dev_result)

                if best_map < map_:
                    best_p_1, best_p_3, best_p_5, best_ap = p_1, p_3, p_5, ap
                    best_map = map_
                    best_mrr = mrr
                    best_dcg_1, best_dcg_3, best_dcg_5 = dcg_1, dcg_3, dcg_5
                    best_ndcg_1, best_ndcg_3, best_ndcg_5 = ndcg_1, ndcg_3, ndcg_5
                    best_model = copy.deepcopy(model)
            n += 1

    torch.save(best_model.state_dict(), mkdir(finetuned_model_path))
    best_model.bert.save_pretrained(mkdir(finetuned_bert_model_path))
    best_dev_result = (
        f'{coloring("best_p@1", "red_bg")}:{round(best_p_1, 4)}\t'
        f'{coloring("best_p@3", "red_bg")}:{round(best_p_3, 4)}\t'
        f'{coloring("best_p@5", "red_bg")}:{round(best_p_5, 4)}\t'
        f'{coloring("best_ap", "green_bg")}:{round(best_ap, 4)}\t'
        f'{coloring("best_map", "yellow_bg")}:{round(best_map, 4)}\t'
        f'{coloring("best_mrr", "blue_bg")}:{round(best_mrr, 4)}\t'
        f'best_dcg@1:{round(best_dcg_1, 4)}\t'
        f'best_dcg@3:{round(best_dcg_3, 4)}\t'
        f'best_dcg@5:{round(best_dcg_5, 4)}\t'
        f'{coloring("best_ndcg@1", "purple_bg")}:{round(best_ndcg_1, 4)}\t'
        f'{coloring("best_ndcg@3", "purple_bg")}:{round(best_ndcg_3, 4)}\t'
        f'{coloring("best_ndcg@5", "purple_bg")}:{round(best_ndcg_5, 4)}'
    )
    with open(mkdir(best_dev_tsv), 'a' if os.path.exists(best_dev_tsv) else 'w') as f:
        f.write(decoloring(best_dev_result) + '\n')
    print(best_dev_result)

    print(f'{coloring("Finetuned model path", "red_bg")}: {finetuned_model_path}')
    print(f'{coloring("Finetuned bert model path", "green_bg")}: {finetuned_bert_model_path}')

    return best_model, timestamp

def get_ranking_metrics(input: np.array, target: np.array, threshold: float = 0.) -> tuple:
    p_1 = precision(input, target, k=1, threshold=threshold)
    p_3 = precision(input, target, k=3, threshold=threshold)
    p_5 = precision(input, target, k=5, threshold=threshold)
    ap = average_precision(input, target, threshold=threshold)
    map_ = mean_average_precision(input, target, threshold=threshold)
    mrr = mean_reciprocal_rank(input, target, threshold=threshold)
    dcg_1 = discounted_cumulative_gain(input, target, k=1, threshold=threshold)
    dcg_3 = discounted_cumulative_gain(input, target, k=3, threshold=threshold)
    dcg_5 = discounted_cumulative_gain(input, target, k=5, threshold=threshold)
    ndcg_1 = normalized_discounted_cumulative_gain(input, target, k=1, threshold=threshold)
    ndcg_3 = normalized_discounted_cumulative_gain(input, target, k=3, threshold=threshold)
    ndcg_5 = normalized_discounted_cumulative_gain(input, target, k=5, threshold=threshold)

    return (p_1, p_3, p_5, ap, map_), mrr, (dcg_1, dcg_3, dcg_5), (ndcg_1, ndcg_3, ndcg_5)


def eval_ranking_metrics_on_data_frame(
        id_left: typing.Any,
        y_true: typing.Union[list, np.array],
        y_pred: typing.Union[list, np.array]
):
    df = pd.DataFrame(
        data={
            'id': id_left,
            'true': y_true,
            'pred': y_pred
        }
    )

    metrics = df.groupby(by='id').apply(
        lambda x: get_ranking_metrics(input=x['pred'].values, target=x['true'].values)
    )
    p_1 = metrics.apply(lambda x: x[0][0]).mean()
    p_3 = metrics.apply(lambda x: x[0][1]).mean()
    p_5 = metrics.apply(lambda x: x[0][2]).mean()
    ap = metrics.apply(lambda x: x[0][3]).mean()
    map_ = metrics.apply(lambda x: x[0][4]).mean()
    mrr = metrics.apply(lambda x: x[1]).mean()
    dcg_1 = metrics.apply(lambda x: x[2][0]).mean()
    dcg_3 = metrics.apply(lambda x: x[2][1]).mean()
    dcg_5 = metrics.apply(lambda x: x[2][2]).mean()
    ndcg_1 = metrics.apply(lambda x: x[3][0]).mean()
    ndcg_3 = metrics.apply(lambda x: x[3][1]).mean()
    ndcg_5 = metrics.apply(lambda x: x[3][2]).mean()

    return (p_1, p_3, p_5, ap, map_), mrr, (dcg_1, dcg_3, dcg_5), (ndcg_1, ndcg_3, ndcg_5)


def evaluate(args, task_name, model, test_dataloader, timestamp, save_test):
    predictions = []
    for test_sample in test_dataloader:
        test_inputs, _ = test_sample
        with torch.no_grad():
            test_outputs = model(test_inputs).detach().cpu()
            predictions.append(test_outputs)
    y_pred = torch.cat(predictions, dim=0).numpy()
    y_true = test_dataloader.label
    id_left = test_dataloader.id_left
    (p_1, p_3, p_5, ap, map_), mrr, (dcg_1, dcg_3, dcg_5), (ndcg_1, ndcg_3, ndcg_5) = (
        eval_ranking_metrics_on_data_frame(id_left, y_true, y_pred.squeeze(axis=-1)))
    best_test_result = (
        f'{task_name}\t'
        f'Ranking:\t'
        f'{coloring("p@1", "red_bg")}:{round(p_1, 4)}\t'
        f'{coloring("p@3", "red_bg")}:{round(p_3, 4)}\t'
        f'{coloring("p@5", "red_bg")}:{round(p_5, 4)}\t'
        f'{coloring("ap", "green_bg")}:{round(ap, 4)}\t'
        f'{coloring("map", "yellow_bg")}:{round(map_, 4)}\t'
        f'{coloring("mrr", "blue_bg")}:{round(mrr, 4)}\t'
        f'dcg@1:{round(dcg_1, 4)}\t'
        f'dcg@3:{round(dcg_3, 4)}\t'
        f'dcg@5:{round(dcg_5, 4)}\t'
        f'{coloring("ndcg@1", "purple_bg")}:{round(ndcg_1, 4)}\t'
        f'{coloring("ndcg@3", "purple_bg")}:{round(ndcg_3, 4)}\t'
        f'{coloring("ndcg@5", "purple_bg")}:{round(ndcg_5, 4)}'
    )
    if args.is_train or save_test:
        best_test_tsv = f'./Result/Temp/{task_name}-{timestamp}/best_test.tsv'
        with open(mkdir(best_test_tsv), 'a' if os.path.exists(best_test_tsv) else 'w') as f:
            f.write(decoloring(best_test_result) + '\n')
    print(best_test_result)
    print(f'{p_1}\t{p_3}\t{p_5}\t{ap}\t{map_}\t{mrr}\t{dcg_1}\t{dcg_3}\t{dcg_5}\t{ndcg_1}\t{ndcg_3}\t{ndcg_5}')


def parse_args():
    parser = argparse.ArgumentParser(description='Our Model without Source Credibility')
    parser.add_argument('--task_name', nargs='?', default='OM_wo_SC',
                        help='Task name')

    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--bert_hidden_size', type=int, default=768,
                        help='Bert hidden size')
    parser.add_argument('--data_dir', nargs='?', default='/home/cuifulai/Projects/CQA/Data/StackExchange',
                        help='Data directory')
    parser.add_argument('--data_name', nargs='?', default='meta.stackoverflow.com',
                        help='Data name')
    parser.add_argument('--device', nargs='?', default='cuda:0',
                        help='Device')
    parser.add_argument('--dropout_prob', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--fold', type=int, default=1,
                        help='Fold')
    parser.add_argument('--finetuned_model_path', nargs='?', default='./FinetunedModel/Our_model-20241004_191930/best_model.pth',
                        help='Finetuned model path')
    parser.add_argument('--freeze', type=bool, default=False,
                        help='Freeze')
    parser.add_argument('--hidden_size', type=int, default=108,
                        help='Hidden size')
    parser.add_argument('--is_from_finetuned', type=bool, default=False,
                        help='Is from finetuned')
    parser.add_argument('--is_train', type=bool, default=True,
                        help='Is train')
    parser.add_argument('--limit', nargs='?', default=[0, 0, 0],
                        help='Limit')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Max length')
    parser.add_argument('--max_seq_length', type=int, default=32,
                        help='Max sequence length')
    parser.add_argument('--normalize', type=bool, default=True,
                        help='Normalize')
    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help='Number of attention heads')
    parser.add_argument('--num_dup', type=int, default=1,
                        help='Number of duplications')
    parser.add_argument('--num_neg', type=int, default=1,
                        help='Number of negative samples')
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
    parser.add_argument('--spacy_path', nargs='?', default='/data/cuifulai/Spacy/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1',
                        help='Spacy path')
    parser.add_argument('--step', type=int, default=1,
                        help='Step')
    parser.add_argument('--threshold', type=int, default=5,
                        help='Threshold')

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    argument_quality = ArgumentQuality(spacy.load(args.spacy_path))
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

    if args.is_train:
        train_dp = OurProcessor(
            data_name=args.data_name,
            stage='train',
            task='ranking',
            filtered=False,
            threshold=args.threshold,
            normalize=args.normalize,
            return_classes=False,
            limit=args.limit[0],
            max_length=args.max_length,
            max_seq_length=args.max_seq_length,
            mode='accept',
            fold=args.fold,
        ).get_train_examples(args.data_dir)
        train_dataset = OurDataset(
            argument_quality=argument_quality,
            tokenizer=tokenizer,
            data_pack=train_dp,
            mode='pair',
            num_dup=args.num_dup,
            num_neg=args.num_neg,
            batch_size=args.batch_size,
            resample=True,
            shuffle=True,
            max_length=args.max_length
        )
        train_dataloader = DataLoader(
            train_dataset,
            stage='train'
        )
    else:
        train_dataloader = None

    test_dp = OurProcessor(
        data_name=args.data_name,
        stage='test',
        task='ranking',
        filtered=True,
        threshold=args.threshold,
        normalize=args.normalize,
        return_classes=False,
        limit=args.limit[2],
        max_length=args.max_length,
        max_seq_length=args.max_seq_length,
        mode='accept',
        fold=args.fold,
    ).get_test_examples(args.data_dir)

    test_dataset = OurDataset(
        argument_quality=argument_quality,
        tokenizer=tokenizer,
        data_pack=test_dp,
        mode='point',
        batch_size=args.batch_size,
        resample=False,
        shuffle=False,
        max_length=args.max_length
    )

    test_dataloader = DataLoader(
        test_dataset,
        stage='test'
    )

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    model = OurModel(
        freeze=args.freeze,
        pretrained_model_name_or_path=args.pretrained_model_path,
        device=device,
        hidden_size=args.hidden_size,
        bert_hidden_size=args.bert_hidden_size,
        dropout_prob=args.dropout_prob,
        num_layers=args.num_layers,
    ).to(device)

    timestamp = None
    if args.is_from_finetuned:
        model.load_state_dict(torch.load(args.finetuned_model_path))
    if args.is_train:
        model, timestamp = train(args, args.task_name, model, train_dataloader, test_dataloader, args.epochs, args.lr, args.step)

    evaluate(args, args.task_name, model, test_dataloader, timestamp, args.save_test)

if __name__ == '__main__':
    main()

