# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/3 10:24
import random


def coloring(content: str, color: str=None) -> str:
    """

    :param content: string to be colored
    :param color: red, green, yellow, blue, purple, cyan, white, red_bg, green_bg, yellow_bg, blue_bg, purple_bg, cyan_bg, white_bg
    :return:
    """
    if color == 'None':
        return content

    colors = {
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'purple': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'red_bg': '\033[41m',
        'green_bg': '\033[42m',
        'yellow_bg': '\033[43m',
        'blue_bg': '\033[44m',
        'purple_bg': '\033[45m',
        'cyan_bg': '\033[46m',
        'white_bg': '\033[47m',
    }
    if color is None:
        while True:
            color = random.choice(list(colors.keys()))
            if color.endswith('_bg'):
                continue
            else:
                break
    return f'{colors[color]}{content}\033[0m'


def main():
    print(coloring('Hello, World!', 'red'))


if __name__ == '__main__':
    main()
