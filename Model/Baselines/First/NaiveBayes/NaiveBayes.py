# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/10/8 14:32
import argparse
import typing

import numpy as np
import pandas as pd
import spacy
import torch
from sklearn.naive_bayes import MultinomialNB
from torch import nn
from tqdm import tqdm
from transformers import BertModel, set_seed, AutoTokenizer

from Model.DataLoader.DataLoader import DataLoader
from Model.DataLoader.DataProcessor import OurProcessor
from Model.DataLoader.Dataset import OurDataset
from Model.Our.Dimension.ArgumentQuality import ArgumentQuality

from warnings import simplefilter

from Model.Unit.cprint import coloring
from Model.Unit.function import ignore_warning
from Model.Unit.metrics import (
    precision, average_precision, mean_average_precision, mean_reciprocal_rank,
    discounted_cumulative_gain, normalized_discounted_cumulative_gain
)

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=DeprecationWarning)
ignore_warning(name="transformers")


class NaiveBayesModel:
    def __init__(
            self,
            freeze: bool = False,
            pretrained_model_name_or_path: str = 'bert-base-uncased',
            device: torch.device = torch.device('cuda:0'),
    ):
        self.device = device

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path).to(self.device)
        for p in self.bert.parameters():
            p.data = p.data.contiguous()

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.feature_norm = nn.BatchNorm1d(44).to(self.device)

        self.naive_bayes = MultinomialNB()

    def forward(self, inputs):
        text_left = torch.stack([x.to(self.device) for x in inputs['text_left']], dim=0)                # torch.Size([batch_size, max_length])
        text_right = torch.stack([x.to(self.device) for x in inputs['text_right']],dim=0)               # torch.Size([batch_size, max_length])
        feature = torch.stack([torch.tensor(x).to(self.device) for x in inputs['feature']], dim=0)      # torch.Size([batch_size, 44])
        try:
            feature = self.feature_norm(feature)                                                        # torch.Size([batch_size, 44])
        except ValueError:
            feature = feature

        bert_output_left = self.bert(text_left)['pooler_output']                                        # torch.Size([batch_size, bert_hidden_size])
        bert_output_right = self.bert(text_right)['pooler_output']                                      # torch.Size([batch_size, bert_hidden_size])

        # AQ
        # Relevancy
        relevancy = torch.nn.functional.cosine_similarity(
            bert_output_left, bert_output_right, dim=-1)                                                # torch.Size([batch_size, 1])

        argument_quality = torch.cat([relevancy.unsqueeze(-1), feature], dim=-1)                 # torch.Size([batch_size, 45])
        argument_quality = torch.clamp(argument_quality, min=0.0)

        return argument_quality.detach().cpu().numpy()

    def train(self, train_dataloader):
        all_features = []

        for train_sample in tqdm(train_dataloader):
            train_inputs, _, _ = train_sample

            features = self.forward(train_inputs)
            all_features.append(features)
        all_features = np.vstack(all_features)
        all_labels = train_dataloader.label

        self.naive_bayes.fit(all_features, all_labels)

    def predict(self, test_dataloader):
        all_features = []

        for test_sample in test_dataloader:
            test_inputs, _, _ = test_sample

            features = self.forward(test_inputs)
            all_features.append(features)
        all_features = np.vstack(all_features)

        y_pred = self.naive_bayes.predict(all_features)
        y_true = test_dataloader.label
        id_left = test_dataloader.id_left

        (p_1, p_3, p_5, ap, map_), mrr, (dcg_1, dcg_3, dcg_5), (ndcg_1, ndcg_3, ndcg_5) = (
            eval_ranking_metrics_on_data_frame(id_left, y_true, y_pred))
        best_test_result = (
            f'{'NB'}\t'
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
        print(best_test_result)
        print(f'{p_1}\t{p_3}\t{p_5}\t{ap}\t{map_}\t{mrr}\t{dcg_1}\t{dcg_3}\t{dcg_5}\t{ndcg_1}\t{ndcg_3}\t{ndcg_5}')


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


def parse_args():
    parser = argparse.ArgumentParser(description='Naive Bayes Model')
    parser.add_argument('--task_name', nargs='?', default='NB',
                        help='Task name')

    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Alpha')
    parser.add_argument('--batch_size', type=int, default=4,
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
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs')
    parser.add_argument('--finetuned_model_path', nargs='?', default='./FinetunedModel/XXX/best_model.pth',
                        help='Finetuned model path')
    parser.add_argument('--fold', type=int, default=1,
                        help='Fold')
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
    parser.add_argument('--limit', nargs='?', default=[0, 0, 0],
                        help='Limit')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--proportion', type=float, default=1.0,
                        help='Proportion')
    parser.add_argument('--margin', type=float, default=5,
                        help='Margin')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Max length')
    parser.add_argument('--max_seq_length', type=int, default=5,
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
            proportion=args.proportion,
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

    model = NaiveBayesModel(
        freeze=args.freeze,
        pretrained_model_name_or_path=args.pretrained_model_path,
        device=device,
    )

    model.train(train_dataloader)
    model.predict(test_dataloader)


if __name__ == '__main__':
    main()
