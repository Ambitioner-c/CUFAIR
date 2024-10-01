# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/29 19:05
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
from Model.LSTM.AttentionLSTM import AttentionLSTMModel
from Model.Our.Dimension.ArgumentQuality import ArgumentQuality

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)


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
    ):
        super(OurModel, self).__init__()
        self.device = device

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        for p in self.bert.parameters():
            p.data = p.data.contiguous()

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.relevancy_layer = nn.Linear(bert_hidden_size * 2, 20)

        self.attention_lstm = AttentionLSTMModel(
            attention_size=bert_hidden_size,
            input_size=bert_hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            output_size=hidden_size,
        )

        self.credibility_layer = nn.Linear(hidden_size, 64)

        self.usefulness_layer = nn.Linear(128, 1)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inputs: dict):
        text_left = torch.stack([x.to(self.device) for x in inputs['text_left']], dim=0)            # torch.Size([batch_size, max_length])
        text_right = torch.stack([x.to(self.device) for x in inputs['text_right']], dim=0)          # torch.Size([batch_size, max_length])
        comment = torch.stack([x.to(self.device) for x in inputs['comment']], dim=0)                # torch.Size([batch_size, max_sequence_length, max_length])
        feature = torch.stack([torch.tensor(x).to(self.device) for x in inputs['feature']], dim=0)  # torch.Size([batch_size, 44])

        bert_output_left = self.bert(text_left)['pooler_output']                                    # torch.Size([batch_size, hidden_size])
        bert_output_right = self.bert(text_right)['pooler_output']                                  # torch.Size([batch_size, hidden_size])

        bert_output_comment = []
        for j in range(len(comment)):
            bert_output_comment.append(self.bert(comment[j])['pooler_output'])
        bert_output_comment = torch.stack(bert_output_comment, dim=0)                               # torch.Size([batch_size, max_sequence_length, hidden_size])

        # AQ
        # Relevancy
        relevancy = self.relevancy_layer(
            torch.cat([bert_output_left, bert_output_right], dim=-1))                        # torch.Size([batch_size, 20])
        argument_quality = torch.cat([relevancy, feature], dim=-1)                           # torch.Size([batch_size, 64])

        # SC
        source_credibility = self.attention_lstm(bert_output_right.unsqueeze(1), bert_output_comment)[:, -1, :]     # torch.Size([batch_size, hidden_size])
        source_credibility = self.credibility_layer(source_credibility)                                             # torch.Size([batch_size, 64])

        usefulness = torch.cat([argument_quality, source_credibility], dim=-1)                               # torch.Size([batch_size, 128])
        outputs = self.usefulness_layer(usefulness)                                                                  # torch.Size([batch_size, 1])

        return outputs


def train(model, train_dataloader, dev_dataloader, epochs, lr, device, step):
    optimizer = Adam(model.parameters(), lr=lr)
    loss_function = RankHingeLoss(
        num_neg=1,
        margin=1.,
        reduction='mean'
    )

    best_loss = float('inf')

    n = 0
    for epoch in range(epochs):
        for train_sample in tqdm(train_dataloader):
            train_inputs, _ = train_sample

            optimizer.zero_grad()

            train_outputs = model(train_inputs)

            train_loss = loss_function(y_pred=train_outputs)
            train_loss.backward()

            optimizer.step()

            train_result = f"Epoch: {epoch + 1}, Loss: {train_loss.item()}"
            print(train_result)


def main():
    set_seed(2024)

    data_dir = '/home/cuifulai/Projects/CQA/Data/StackExchange'
    data_name = 'meta.stackoverflow.com'

    spacy_path = "/data/cuifulai/Spacy/en_core_web_sm-3.7.1/en_core_web_sm/en_core_web_sm-3.7.1"
    nlp = spacy.load(spacy_path)
    argument_quality = ArgumentQuality(nlp)

    pretrained_model_path = '/data/cuifulai/PretrainedModel/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    train_dp = OurProcessor(
        data_name=data_name,
        stage='train',
        task='ranking',
        filtered=False,
        threshold=5,
        normalize=True,
        return_classes=False,
        limit=100,
        max_length=256
    ).get_train_examples(data_dir)
    dev_dp = OurProcessor(
        data_name=data_name,
        stage='dev',
        task='ranking',
        filtered=True,
        threshold=5,
        normalize=True,
        return_classes=False,
        limit=100,
        max_length=256
    ).get_dev_examples(data_dir)
    test_dp = OurProcessor(
        data_name=data_name,
        stage='test',
        task='ranking',
        filtered=True,
        threshold=5,
        normalize=True,
        return_classes=False,
        limit=100,
        max_length=256
    ).get_test_examples(data_dir)

    train_dataset = OurDataset(
        argument_quality=argument_quality,
        tokenizer=tokenizer,
        data_pack=train_dp,
        mode='pair',
        num_dup=1,
        num_neg=1,
        batch_size=2,
        resample=True,
        shuffle=True,
        sort=False,
        max_length=256
    )
    dev_dataset = OurDataset(
        argument_quality=argument_quality,
        tokenizer=tokenizer,
        data_pack=dev_dp,
        mode='point',
        num_dup=1,
        num_neg=1,
        batch_size=2,
        resample=False,
        shuffle=False,
        sort=False,
        max_length=256
    )
    test_dataset = OurDataset(
        argument_quality=argument_quality,
        tokenizer=tokenizer,
        data_pack=test_dp,
        mode='point',
        num_dup=1,
        num_neg=1,
        batch_size=2,
        resample=False,
        shuffle=False,
        sort=False,
        max_length=256
    )

    train_dataloader = DataLoader(
        train_dataset,
        stage='train'
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        stage='dev'
    )
    test_dataloader = DataLoader(
        test_dataset,
        stage='test'
    )

    model = OurModel(
        freeze=False,
        pretrained_model_name_or_path=pretrained_model_path,
        device=torch.device('cuda:0'),
        hidden_size=108,
        bert_hidden_size=768,
        dropout_prob=0.1,
        num_layers=1,
    ).to(torch.device('cuda:0'))

    train(model, train_dataloader, dev_dataloader, epochs=5, lr=5e-5, device=torch.device('cuda:0'), step=1)


if __name__ == '__main__':
    main()
