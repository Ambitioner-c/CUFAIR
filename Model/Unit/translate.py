# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/5 16:34
import configparser
import json
import random
from hashlib import md5

import requests


class BaiduTranslate:
    def __init__(self, config_path: str):
        self.url = self.get_url()
        self.configs = self.get_configs(config_path)

    def translate(
            self,
            query: str,
            from_lang: str = 'en',
            to_lang: str = 'zh'
    ) -> str:
        query = query.replace('\n', ' ')
        headers, payload = self.build_request(query, from_lang, to_lang)
        response: json = requests.post(self.url, params=payload, headers=headers).json()
        return response['trans_result'][0]['dst']


    @staticmethod
    def get_url():
        endpoint = 'http://api.fanyi.baidu.com'
        path = '/api/trans/vip/translate'
        return endpoint + path

    @staticmethod
    def make_md5(s: str, encoding: str = 'utf-8') -> str:
        return md5(s.encode(encoding)).hexdigest()

    def get_sign(self, query: str, salt) -> str:
        return self.make_md5(self.configs['app_id'] + query + str(salt) + self.configs['app_key'])

    def build_request(self, query: str, from_lang: str, to_lang: str):
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}

        salt = random.randint(32768, 65536)
        payload = {
            'appid': self.configs['app_id'],
            'q': query,
            'from': from_lang,
            'to': to_lang,
            'salt': salt,
            'sign': self.get_sign(query, salt)
        }
        return headers, payload

    @staticmethod
    def get_configs(config_path: str) -> dict:
        parser = configparser.RawConfigParser()
        parser.read(config_path)
        return {
            'app_id': parser.get('Baidu Text transAPI', 'app_id'),
            'app_key': parser.get('Baidu Text transAPI', 'app_key')
        }


def main():
    config_path = '/home/cuifulai/Projects/CQA/config.ini'
    query = 'Hello World! This is 1st paragraph.\nThis is 2nd paragraph.'

    translate = BaiduTranslate(config_path)
    response = translate.translate(query, from_lang='en', to_lang='zh')
    print(response)


if __name__ == '__main__':
    main()
