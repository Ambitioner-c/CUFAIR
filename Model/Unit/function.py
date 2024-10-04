# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/5 20:35
import json
import os


def mkdir(file_dir: str) -> str:
    os.makedirs(os.path.dirname(file_dir), exist_ok=True)
    return file_dir


def save_args_to_file(args, file_path: str):
    args_dict = vars(args)

    with open(file_path, 'w') as f:
        json.dump(args_dict, f, indent=4)


def main():
    pass


if __name__ == '__main__':
    main()
