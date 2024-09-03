# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/1 10:28
import argparse
import configparser
import json
import os
from random import randint
from time import sleep
from typing import *

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tqdm import trange
from transformers import set_seed

from Model.Baselines.Ablation.StackExchange.DataLoader.DataProcessor import SEProcessor
from Model.Unit.cprint import *


class Relation(BaseModel):
    type: str
    subtype: str
    description: str


class Detail(BaseModel):
    type: str
    content: str
    explanation: str


class Annotation(BaseModel):
    relation: Relation
    left: Detail
    right: Detail
    category: str


class Annotator:
    def __init__(
            self,
            configs: dict
    ):
        self.client: OpenAI = self.get_client(api_key=configs["api_key"], base_url=configs["base_url"])

    @staticmethod
    def get_client(api_key: str, base_url: str) -> OpenAI:
        return OpenAI(api_key=api_key, base_url=base_url)

    def get_response(
            self,
            idx: int,
            prompt: str,
            content: str,
            model_name: str = "gpt-4o-2024-08-06",
            temperature: float = 0.0,
            seed: int = 2024,
            max_tokens: int = 256
    ) -> str:
        try:
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a Rhetorical Structure Theory (RST) expert."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                response_format={"type": "json_object"},
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,
            )
            response = completion.choices[0].message

            if response.content:
                return response.content
            elif response.refusal:
                # handle refusal
                print(f"{coloring('Refusal')}: {response.refusal}")
                exit(1)
        except Exception as e:
            print(f"{coloring('Error', 'yellow_bg')}: {coloring(str(idx), 'red')}")

            # Handle edge cases
            if type(e) == openai.LengthFinishReasonError:
                # Retry with a higher max tokens
                print(f"{coloring('LengthFinishReasonError')}")
                exit(2)
            else:
                # Handle other exceptions
                print(f"{coloring('Exception')}: {e}")
                exit(3)


def get_configs(config_path: str) -> dict:
    parser = configparser.RawConfigParser()
    parser.read(config_path)
    return {
        'api_key': parser.get('OpenAI API', 'api_key'),
        'base_url': parser.get('OpenAI API', 'base_url')
    }


def get_prompt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_content(left: str, right: str):
    return json.dumps({
        "left": left,
        "right": right
    })


def get_sampled_rows(args: argparse.Namespace, error: Optional[int] = False) -> dict:
    df = SEProcessor(
        data_name=args.data_name,
        limit=args.limit,
        show=False,
        save=None,
        threshold=args.threshold
    ).get_all_examples(args.data_dir)

    sampled_df = df.sample(n=args.sample_size, random_state=args.seed)

    rows = dict()
    n = 0
    for _, row in sampled_df.iterrows():
        if error:
            if n >= error:
                rows[n] = get_content(left=row['left'], right=row['right'])
        else:
            rows[n] = get_content(left=row['left'], right=row['right'])
        n += 1

    return rows


def mkdir(file_dir: str) -> str:
    os.makedirs(os.path.dirname(file_dir), exist_ok=True)
    return file_dir


def write(args: argparse.Namespace, annotation: Annotation) -> bool:
    file_path = mkdir(os.path.join(args.output_dir, f"{args.model_name}")) + '/rows.txt'

    with open(file_path, 'a' if os.path.exists(file_path) else 'w', encoding='utf-8') as f:
        f.write(json.dumps(annotation.model_dump()) + '\n')
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='GPT generated annotations')

    parser.add_argument('--task_name', nargs='?', default='GPT_generated_annotations',
                        help='Task name')
    parser.add_argument('--data_dir', nargs='?', default='/home/cuifulai/Projects/CQA/Data/StackExchange',
                        help='Data directory')
    parser.add_argument('--data_name', nargs='?', default='meta.stackoverflow.com',
                        help='Data name')
    parser.add_argument('--prompt_path', nargs='?', default='/home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Annotation/prompt.txt',
                        help='Prompt path')
    parser.add_argument('--config_path', nargs='?', default='/home/cuifulai/Projects/CQA/config.ini',
                        help='Config path')
    parser.add_argument('--output_dir', nargs='?', default='/home/cuifulai/Projects/CQA/Data/StackExchange/meta.stackoverflow.com/Annotation',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--limit', type=int, default=0,
                       help='Limit')
    parser.add_argument('--threshold', type=float, default=-1.0,
                        help='Threshold')
    parser.add_argument('--model_name', nargs='?', default="gpt-4o-2024-08-06",
                        help='Model name')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature')
    parser.add_argument('--max_tokens', type=int, default=1024,
                        help='Max tokens')
    parser.add_argument('--sample_size', type=int, default=200,
                        help='Sample size')
    parser.add_argument('--error', type=Optional[int], default=117,
                        help='Error')

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    prompt = get_prompt(file_path=args.prompt_path)

    contents = get_sampled_rows(args, args.error)

    annotator = Annotator(configs=get_configs(args.config_path))

    finish = 0
    for idx in trange(args.sample_size):
        if idx not in contents:
            continue

        response: str = annotator.get_response(
            idx=idx,
            prompt=prompt,
            content=contents[idx],
            model_name=args.model_name,
            temperature=args.temperature,
            seed=args.seed,
            max_tokens=args.max_tokens
        )

        sleep(randint(1, 3))

        try:
            row: json = json.loads(response)
        except json.decoder.JSONDecodeError:
            try:
                row: json = json.loads(response.replace(r'\\', '\\').replace(r'\"', '"'))
            except json.decoder.JSONDecodeError:
                print(f"{coloring('Error', 'yellow_bg')}: {coloring(str(idx), 'red')}")
                print(f"{coloring('json.decoder.JSONDecodeError')}")
                print(coloring(response, 'purple'))
                exit(4)

        try:
            annotation = Annotation.model_validate(row)
        except ValidationError:
            try:
                try:
                    row = json.loads(row['messages'][0]['content'])
                except json.decoder.JSONDecodeError:
                    print(f"{coloring('Error', 'yellow_bg')}: {coloring(str(idx), 'red')}")
                    print(f"{coloring('json.decoder.JSONDecodeError')}")
                    print(coloring(response, 'purple'))
                    exit(5)
                annotation = Annotation.model_validate(row)
            except ValidationError:
                print(f"{coloring('Error', 'yellow_bg')}: {coloring(str(idx), 'red')}")
                print(f"{coloring('ValidationError')}")
                print(coloring(response, 'purple'))
                exit(6)

        if write(args, annotation):
            finish = idx
            print(f"{coloring('Success', 'green')}: {coloring(str(idx), 'blue')}")

    print(f"{coloring('Error', 'green')}: {coloring(str(finish + 1), 'blue')}")


if __name__ == '__main__':
    main()
