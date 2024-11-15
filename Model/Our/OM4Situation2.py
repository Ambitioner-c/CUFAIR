# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/10/8 14:32
import argparse
import re
import typing
from collections import Counter

import numpy as np
import pandas as pd
import spacy
import torch
from torch import nn
from transformers import BertModel, set_seed, AutoTokenizer

from Model.DataLoader.DataLoader import DataLoader
from Model.DataLoader.DataProcessor import OurProcessor
from Model.DataLoader.Dataset import OurDataset
from Model.LSTM.SQACILSTM import SQACILSTMModel
from Model.Our.Dimension.ArgumentQuality import ArgumentQuality

from warnings import simplefilter

from Model.Unit.function import ignore_warning

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
            num_attention_heads: int = 12,
            num_labels: int = 2,
            is_peephole: bool = False,
            ci_mode: str = 'all',
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

        self.credibility_layer = nn.Linear(hidden_size, 64)

        self.community_support_layer = nn.Linear(64, 1)

        self.usefulness_layer = nn.Linear(128, num_labels)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs: dict):
        text_left = torch.stack([x.to(self.device) for x in inputs['text_left']], dim=0)            # torch.Size([batch_size, max_length])
        rel_text_left = torch.stack([x.to(self.device) for x in inputs['rel_text_left']], dim=0)    # torch.Size([batch_size, max_length])
        text_right = torch.stack([x.to(self.device) for x in inputs['text_right']], dim=0)          # torch.Size([batch_size, max_length])
        comment = torch.stack([x.to(self.device) for x in inputs['comment']], dim=0)                # torch.Size([batch_size, max_sequence_length, max_length])
        ping = torch.stack([x.to(self.device) for x in inputs['ping']], dim=0)                      # torch.Size([batch_size, max_sequence_length])
        feature = torch.stack([torch.tensor(x).to(self.device) for x in inputs['feature']], dim=0)  # torch.Size([batch_size, 44])
        try:
            feature = self.feature_norm(feature)                                                        # torch.Size([batch_size, 44])
        except ValueError:
            feature = feature

        bert_output_left = self.reduction_layer(self.bert(text_left)['pooler_output'])              # torch.Size([batch_size, hidden_size])
        bert_output_rel_left = self.reduction_layer(self.bert(rel_text_left)['pooler_output'])      # torch.Size([batch_size, hidden_size])
        bert_output_right = self.reduction_layer(self.bert(text_right)['pooler_output'])            # torch.Size([batch_size, hidden_size])

        bert_output_comment = []
        for j in range(len(comment)):
            bert_output_comment.append(self.reduction_layer(self.bert(comment[j])['pooler_output']))
        bert_output_comment = torch.stack(bert_output_comment, dim=0)                               # torch.Size([batch_size, max_sequence_length, hidden_size])

        # AQ
        # Relevancy
        relevancy = self.relevancy_layer(
            torch.cat([bert_output_left, bert_output_right], dim=-1))                        # torch.Size([batch_size, 20])
        argument_quality = torch.cat([relevancy, feature], dim=-1)                           # torch.Size([batch_size, 64])

        # SC
        source_credibility = self.sqacilstm(bert_output_rel_left.unsqueeze(1), bert_output_right.unsqueeze(1), bert_output_comment, ping)[:, -1, :]     # torch.Size([batch_size, hidden_size])
        source_credibility = self.credibility_layer(source_credibility)                                             # torch.Size([batch_size, 64])

        usefulness = torch.cat([argument_quality, source_credibility], dim=-1)                               # torch.Size([batch_size, 128])
        outputs = self.usefulness_layer(usefulness)[:, 1].unsqueeze(1)                                              # torch.Size([batch_size, 1])

        return outputs, self.community_support_layer(source_credibility)


def get_top_1(
        id_lefts: typing.Any,
        y_preds: typing.Union[list, np.array],
):
    df = pd.DataFrame(
        data={
            'id': id_lefts,
            'pred': y_preds
        }
    )
    result = df.groupby(by='id')['pred'].idxmax().reset_index(name='idxmax')

    counts = dict(Counter(id_lefts))
    start_points = {}
    for idx in range(100):
        if idx == 0:
            start_points[f'L-{idx}'] = 0
        else:
            try:
                start_points[f'L-{idx}'] = start_points[f'L-{idx-1}'] + counts[f'L-{idx-1}']
            except KeyError:
                break

    result['idxmax'] = result.apply(lambda x: x['idxmax'] - start_points[x['id']], axis=1)

    return result


def evaluate(args, task_name, model, test_dataloader, timestamp):
    predictions = []
    for test_sample in test_dataloader:
        test_inputs, _, _ = test_sample
        with torch.no_grad():
            test_outputs = model(test_inputs)[0].detach().cpu()
            predictions.append(test_outputs)
    y_preds = torch.cat(predictions, dim=0).numpy()
    id_lefts = test_dataloader.id_left

    result = get_top_1(id_lefts, y_preds.squeeze(axis=-1))

    if args.save_test:
        result.to_csv(f'./Result/Situation2/{task_name}-{timestamp}.csv', index=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Our Model for Situation 2')
    parser.add_argument('--task_name', nargs='?', default='OM4Situation2',
                        help='Task name')

    parser.add_argument('--alpha', type=float, default=0.8,
                        help='Alpha')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--bert_hidden_size', type=int, default=768,
                        help='Bert hidden size')
    parser.add_argument('--ci_mode', nargs='?', default='all',
                        help='CI Mode')
    parser.add_argument('--data_dir', nargs='?', default='/home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Situation2',
                        help='Data directory')
    parser.add_argument('--data_name', nargs='?', default='meta.stackoverflow.com',
                        help='Data name')
    parser.add_argument('--device', nargs='?', default='cuda:1',
                        help='Device')
    parser.add_argument('--dropout_prob', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--finetuned_model_path', nargs='?', default='./FinetunedModel/OM-20241031_122542/best_model.pth',
                        help='Finetuned model path')
    parser.add_argument('--fold', type=int, default=1,
                        help='Fold')
    parser.add_argument('--freeze', type=bool, default=False,
                        help='Freeze')
    parser.add_argument('--hidden_size', type=int, default=108,
                        help='Hidden size')
    parser.add_argument('--is_from_finetuned', type=bool, default=True,
                        help='Is from finetuned')
    parser.add_argument('--is_peephole', type=bool, default=False,
                        help='Is peephole')
    parser.add_argument('--is_train', type=bool, default=False,
                        help='Is train')
    parser.add_argument('--limit', nargs='?', default=[0, 0, 0],
                        help='Limit')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='Learning rate')
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
    parser.add_argument('--save_test', type=bool, default=True,
                        help='Save test')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--spacy_path', nargs='?', default='/data/cuifulai/Spacy/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1',
                        help='Spacy path')
    parser.add_argument('--step', type=int, default=1,
                        help='Step')
    parser.add_argument('--threshold', type=int, default=0,
                        help='Threshold')

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    argument_quality = ArgumentQuality(spacy.load(args.spacy_path))
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)

    all_dp = OurProcessor(
        data_name=args.data_name,
        stage='test',
        task='ranking',
        filtered=False,
        threshold=args.threshold,
        normalize=args.normalize,
        return_classes=False,
        limit=args.limit[2],
        max_length=args.max_length,
        max_seq_length=args.max_seq_length,
        mode='accept',
        fold=args.fold,
        situation=2,
    ).get_all_examples(args.data_dir)

    all_dataset = OurDataset(
        argument_quality=argument_quality,
        tokenizer=tokenizer,
        data_pack=all_dp,
        mode='point',
        batch_size=args.batch_size,
        resample=False,
        shuffle=False,
        max_length=args.max_length
    )

    all_dataloader = DataLoader(
        all_dataset,
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
        num_attention_heads=args.num_attention_heads,
        is_peephole=args.is_peephole,
        ci_mode=args.ci_mode,
    ).to(device)

    timestamp = re.findall(r'-(\d+_\d+)/', args.finetuned_model_path)[0]
    if args.is_from_finetuned:
        model.load_state_dict(torch.load(args.finetuned_model_path))

    evaluate(args, args.task_name, model, all_dataloader, timestamp)

if __name__ == '__main__':
    main()
