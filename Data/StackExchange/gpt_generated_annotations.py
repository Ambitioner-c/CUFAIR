# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/1 10:28
import argparse
import configparser
import json

import openai
from openai import OpenAI
from pydantic import BaseModel
from transformers import set_seed


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
                print(response.refusal)
        except Exception as e:
            # Handle edge cases
            if type(e) == openai.LengthFinishReasonError:
                # Retry with a higher max tokens
                print("Too many tokens: ", e)
                pass
            else:
                # Handle other exceptions
                print(e)
                pass


def get_configs(config_path: str) -> dict:
    parser = configparser.RawConfigParser()
    parser.read(config_path)
    return {
        'api_key': parser.get('OpenAI API', 'api_key'),
        'base_url': parser.get('OpenAI API', 'base_url')
    }


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
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed')
    parser.add_argument('--limit', type=int, default=0,
                       help='Limit')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold')
    parser.add_argument('--model_name', nargs='?', default="gpt-4o-2024-08-06",
                        help='Model name')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Temperature')
    parser.add_argument('--max_tokens', type=int, default=1024,
                        help='Max tokens')

    return parser.parse_args()


def get_prompt(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_content(left: str, right: str):
    return json.dumps({
        "left": left,
        "right": right
    })


def main():
    args = parse_args()
    configs = get_configs(args.config_path)

    set_seed(args.seed)

    left = "In this case, I should get more of those."
    right = "@bfavaretto Caching, always caching. Wait a few minutes."

    prompt = get_prompt(file_path=args.prompt_path)
    content = get_content(left=left, right=right)

    annotator = Annotator(configs=configs)
    response: str = annotator.get_response(
        prompt=prompt,
        content=content,
        model_name=args.model_name,
        temperature=args.temperature,
        seed=args.seed,
        max_tokens=args.max_tokens
    )

    try:
        response = json.loads(response)
    except json.decoder.JSONDecodeError as e:
        print(e)

    annotation = Annotation.model_validate(response)
    print(annotation)


if __name__ == '__main__':
    main()
