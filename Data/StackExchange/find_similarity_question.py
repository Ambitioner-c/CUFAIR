# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/11/4 9:31
import xml.etree.ElementTree as ElementTree
from typing import Optional

import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset
from tqdm import tqdm, trange
from transformers import PreTrainedTokenizer, AutoTokenizer, BertModel

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)
simplefilter(action='ignore', category=DeprecationWarning)

from Model.Unit.function import ignore_warning
ignore_warning(name="transformers")


class Split:
    def __init__(
            self,
            data_name: str = 'meta.stackoverflow.com',
            num_answers: int = 2,
            num_comments: int = 5,
            limit: int = 0,
            filtered: bool = True,
    ):
        self.data_name = data_name
        self.num_answers = num_answers
        self.num_comments = num_comments
        self.limit = limit
        self.filtered = filtered

    def run(self):
        original_elems = []
        candidate_elems = []
        for elem in tqdm(self.iterparse(f'./{self.data_name}/{self.data_name}.xml', limit=self.limit),
                         desc=f"Splitting {self.data_name} XML file"):
            # Answers
            answers = elem.findall('Answer')

            switch = True
            if self.filtered:
                for answer in answers:
                    if answer.attrib['ACCEPTED_ANSWER'] == 'Yes':
                        switch = False
                        break
                    if int(answer.attrib['COMMENT_COUNT']) < self.num_comments:
                        switch = False
                        break
            if len(answers) == self.num_answers and switch:
                original_elems.append(elem)
            else:
                candidate_elems.append(elem)

        return original_elems, candidate_elems

    @staticmethod
    def iterparse(filepath: str, limit: int):
        with open(filepath, 'r', encoding='utf-8') as f:
            context = ElementTree.iterparse(f, events=('end',))
            _, root = next(context)

            n = 0
            for event, elem in context:
                if elem.tag == 'Thread':
                    if limit:
                        yield elem
                        root.clear()
                        n += 1

                        if n == limit:
                            break
                    else:
                        yield elem
                        root.clear()


class OurProcessor:
    def __init__(
            self,
            original_elems: list,
            candidate_elems: list,
    ):
        super(OurProcessor, self).__init__()

        self.original_elems = original_elems
        self.candidate_elems = candidate_elems

    def get_original_examples(self):
        text_left_ids: list = []
        text_lefts: list = []
        for elem in self.original_elems:
            # Question
            question = elem.find('Question')
            q_id: str = question.attrib['ID']
            q_body: str = question.find('QBody').text

            text_left_ids.append(q_id)
            text_lefts.append(q_body)

        assert len(text_left_ids) == len(text_lefts)

        return pd.DataFrame({
            'left_id': text_left_ids,
            'text_left': text_lefts,
        })


    def get_candidate_examples(self):
        text_left_ids: list = []
        text_lefts: list = []
        for elem in self.candidate_elems:
            # Question
            question = elem.find('Question')
            q_id: str = question.attrib['ID']
            q_title: str = question.find('QTitle').text

            text_left_ids.append(q_id)
            text_lefts.append(q_title)

        return pd.DataFrame({
            'left_id': text_left_ids,
            'text_left': text_lefts,
        })


class OurDataset(Dataset):
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            original_df: pd.DataFrame,
            candidate_df: pd.DataFrame,
            max_length: Optional[int] = None,
    ):
        super(OurDataset, self).__init__()

        self.original_left_ids = original_df['left_id'].tolist()
        self.original_text_left_features = self.convert_examples_to_features(original_df, tokenizer, max_length)
        self.candidate_left_ids = candidate_df['left_id'].tolist()
        self.candidate_text_left_features = self.convert_examples_to_features(candidate_df, tokenizer, max_length)

    @staticmethod
    def convert_examples_to_features(
            examples: pd.DataFrame,
            tokenizer: PreTrainedTokenizer,
            max_length: Optional[int] = None,
    ) -> [Tensor]:
        if max_length is None:
            max_length = tokenizer.model_max_length

        text_left_features = tokenizer(
            [x if x is not None else '' for x in examples['text_left'].tolist()],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )['input_ids']

        return text_left_features


class BERTModel(nn.Module):
    def __init__(
            self,
            freeze: bool = True,
            pretrained_model_name_or_path: str = 'bert-base-uncased',
            device: torch.device = torch.device('cuda:0'),
    ):
        super(BERTModel, self).__init__()

        self.device = device

        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path)
        for p in self.bert.parameters():
            p.data = p.data.contiguous()

        if freeze:
            for p in self.bert.parameters():
                p.requires_grad = False

    def forward(self, original_samples, candidate_samples):
        original_left_ids, original_text_left_features = original_samples
        candidate_left_ids, candidate_text_left_features = candidate_samples

        original_text_left_features = original_text_left_features.to(self.device)
        candidate_text_left_features = candidate_text_left_features.to(self.device)

        length = original_text_left_features.shape[0]
        length = 100 if length > 100 else length

        original_outputs = self.bert(original_text_left_features[:length])['pooler_output']

        outputs = []
        batch_size = 5000
        for i in trange(length, desc="Calculating cosine similarity"):
            temp_outputs = []
            for j in range(0, candidate_text_left_features.shape[0], batch_size):
                candidate_outputs = self.bert(candidate_text_left_features[j:j + batch_size])['pooler_output']
                temp_outputs.append(torch.nn.functional.cosine_similarity(original_outputs[i].unsqueeze(0), candidate_outputs, dim=1))
            topk_list = torch.cat(temp_outputs, dim=0).topk(10)
            outputs.append(topk_list)

        return outputs


def main():
    data_name = 'meta.stackoverflow.com'
    num_answers = 2
    num_comments = 5
    limit = 0

    original_elems, candidate_elems = Split(data_name, num_answers, num_comments, limit, filtered=True).run()

    processor = OurProcessor(original_elems, candidate_elems)
    original_df = processor.get_original_examples()
    candidate_df = processor.get_candidate_examples()

    pretrained_model_path = '/data/cuifulai/PretrainedModel/bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    dataset = OurDataset(tokenizer, original_df, candidate_df, max_length=256)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = BERTModel(
        freeze=True,
        pretrained_model_name_or_path=pretrained_model_path,
        device=device
    ).to(device)

    outputs = model((dataset.original_left_ids, dataset.original_text_left_features), (dataset.candidate_left_ids, dataset.candidate_text_left_features))

    with open(f'./{data_name}/Situation2/related_questions_{str(num_answers)}.txt', 'w', encoding='utf-8') as f:
        for i, output in enumerate(outputs):
            for j in range(output.indices.shape[0]):
                f.write(f"{dataset.original_left_ids[i]}\t"
                        f"{dataset.candidate_left_ids[output.indices[j]]}\t"
                        f"{output.values[j]}\n")


if __name__ == '__main__':
    main()
