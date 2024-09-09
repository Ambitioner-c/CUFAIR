# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/6 8:59
import json
import os.path
import tkinter as tk
from datetime import datetime
from tkinter import scrolledtext

try:
    from typing import Literal, Optional
except ImportError:
    from typing_extensions import Literal, Optional

import logging


class GUIAnnotation:
    def __init__(
            self,
            sample_path: str,
    ):
        self.sample_path = sample_path

        self.page: int = 0
        self.selected_option: tk.IntVar() = None

        self.objs: [json] = self.get_objs()
        self.length: int = len(self.objs)

        self.root = tk.Tk()

    def run(self):
        self.create_window()

        if self.page == 0:
            self.place(self.objs[self.page])
            logging.info(f"[Page] {self.page}")

        self.draw_navigation_buttons()

        self.root.mainloop()

    def draw_radio_buttons(self):
        self.selected_option = tk.IntVar()

        correct_button = tk.Radiobutton(self.root, text="Correct", variable=self.selected_option, value=0, bg='red', command=lambda: logging.info(f"[Select] {self.page if self.page >= 0 else self.page + self.length}:{self.selected_option.get()}"))
        correct_button.grid(row=5, column=0, padx=10, pady=5)
        incorrect_button = tk.Radiobutton(self.root, text="Incorrect", variable=self.selected_option, value=1, bg='green', command=lambda: logging.info(f"[Select] {self.page if self.page >= 0 else self.page + self.length}:{self.selected_option.get()}"))
        incorrect_button.grid(row=5, column=1, padx=10, pady=5)

    def draw_navigation_buttons(self):
        def next_page():
            logging.info(f"[Decision] {self.page if self.page >= 0 else self.page + self.length}:{self.selected_option.get()}")
            self.page += 1
            try:
                self.place(self.objs[self.page])
            except IndexError:
                logging.info("[Info] End of the file: forward")
                exit(0)
            logging.info(f"[Action] {self.page-1 if self.page-1 >= 0 else self.page-1 + self.length}->{self.page if self.page >= 0 else self.page + self.length}")
            logging.info(f"[Page] {self.page if self.page >= 0 else self.page + self.length}")

        def previous_page():
            logging.info(f"[Decision] {self.page if self.page >= 0 else self.page + self.length}:{self.selected_option.get()}")
            self.page -= 1
            try:
                self.place(self.objs[self.page])
            except IndexError:
                logging.info("[Info] End of the file: back forward")
                exit(0)
            logging.info(f"[Action] {self.page+1 if self.page+1 >= 0 else self.page+1 + self.length}->{self.page if self.page >= 0 else self.page + self.length}")
            logging.info(f"[Page] {self.page if self.page >= 0 else self.page + self.length}")

        previous_button = tk.Button(self.root, text="Previous", command=previous_page)
        previous_button.grid(row=6, column=0, padx=10, pady=5)
        next_button = tk.Button(self.root, text="Next", command=next_page)
        next_button.grid(row=6, column=1, padx=10, pady=5)

    def clear(self):
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame) or isinstance(widget, tk.Radiobutton):
                widget.destroy()

    def place(self, obj: json):
        self.clear()

        # Prompt
        prompt_origi_frame = self.get_frame(row=0, column=0, anchor="w", bg='#deebf7')
        prompt_trans_frame = self.get_frame(row=0, column=1, anchor="e", bg='#deebf7')
        self.fix_grid(prompt_origi_frame, **{"1": 30})
        self.fix_grid(prompt_trans_frame, **{"1": 30})
        self.get_prompt(
            prompt_origi_frame,
            'Task Description',
            'origi',
            "Here are two interacting comments from a community Q&A platform (Stack Overflow): Left.Content and Right.Content, where Right is a reply to Left (either explicitly or implicitly mentioning other user @name).\nWe aim to assess the relative importance of the two comments based on Rhetorical Structure Theory (RST), a method used to describe potential structural relations between two texts (Nucleus-Satellite (N-S, Left is importance than Right), Satellite-Nucleus (S-N, Right is importance than Left), Nucleus-Nucleus (N-N, Left is as important as Right), where the Nucleus is considered more important than the Satellite).\nUsing prompt engineering, we employed the gpt-4o-2024-08-06 model to generate possible structural relations, including the types and explanations of both comments, the type, subtype, and description of the relation, as well as the final relation category.\n You are required to first provide your own assessment of the relative importance of the two texts without external help (without clicking Show), and then optionally view the judgment given by GPT-4o. You need to check and judge whether the relation category is reasonable."
        )
        self.get_prompt(
            prompt_trans_frame,
            'Task Description',
            'trans',
            "这里展示了社区问答平台上两个存在交互的评论：Left.Content和Right.Content，其中，Right是Left的回复（显式或隐式的@了被回复者）。\n我们试图根据修辞结构理论（rhetorical structure theory, RST），一种描述两个文本可能的结构关系（Nucleus-Satellite (N-S，Left比Right更重要), Satellite-Nucleus (S-N，Right比Left更重要), Nucleus-Nucleus (N-N，Left和Right都重要)，其中Nucleus（核）相对Satellite（卫星）更重要）的方法，来判断两个评论的相对重要性。\n我们通过提示工程的方式，使用gpt-4o-2024-08-06模型生成了可能的结构关系，其中包括两个评论的类型和解释，关系的类型、子类型和描述以及最终的分类判断。\n您需要首先在无外界帮助（不点击Show）的情况下给出您自认为的两个文本的相对重要性，然后可选择性的查看GPT-4o给出的判断。您最终需要查看并判断关系分类是否合理。"
        )

        # Left
        left_origi_frame = self.get_frame(row=1, column=0, anchor="w", bg='#e2f0d9')
        left_trans_frame = self.get_frame(row=1, column=1, anchor="e", bg='#e2f0d9')
        self.fix_grid(left_origi_frame, **{"1": 30, "3": 70})
        self.fix_grid(left_trans_frame, **{"1": 30, "3": 70})
        self.get_component(left_origi_frame, obj['左节点(left)'], 'Left', 'origi')
        self.get_component(left_trans_frame, obj['左节点(left)'], 'Left', 'trans')

        # Right
        right_origi_frame = self.get_frame(row=2, column=0, anchor="w", bg='#fff2cc')
        right_trans_frame = self.get_frame(row=2, column=1, anchor="e", bg='#fff2cc')
        self.fix_grid(right_origi_frame, **{"1": 30, "3": 70})
        self.fix_grid(right_trans_frame, **{"1": 30, "3": 70})
        self.get_component(right_origi_frame, obj['右节点(right)'], 'Right', 'origi')
        self.get_component(right_trans_frame, obj['右节点(right)'], 'Right', 'trans')

        # Relation
        relation_origi_frame = self.get_frame(row=3, column=0, anchor="w", bg='#dae3f3')
        relation_trans_frame = self.get_frame(row=3, column=1, anchor="e", bg='#dae3f3')
        self.fix_grid(relation_origi_frame, **{"1": 30, "2": 30, "3": 70})
        self.fix_grid(relation_trans_frame, **{"1": 30, "2": 30, "3": 70})
        self.get_component(relation_origi_frame, obj['关系(relation)'], 'Relation', 'origi')
        self.get_component(relation_trans_frame, obj['关系(relation)'], 'Relation', 'trans')

        # Category
        category_frame = self.get_frame(row=4, column=0, anchor="w", bg='#fbe5d6', columnspan=2)
        self.fix_grid(category_frame, **{"1": 30})
        self.get_category(category_frame, 'Category', obj['类别(category)'])

        # Radio buttons
        self.draw_radio_buttons()


    def get_frame(
            self,
            row: int,
            column: int,
            anchor: Literal["nw", "n", "ne", "w", "center", "e", "sw", "s", "se"],
            bg: str,
            columnspan: Optional[int]=None
    ):
        frame = tk.Frame(self.root, bg=bg)
        if columnspan:
            frame.grid(row=row, column=column, padx=10, pady=5, columnspan=columnspan)
        else:
            frame.grid(row=row, column=column, padx=10, pady=5, sticky=anchor)
        return frame

    @staticmethod
    def get_prompt(
            frame: tk.Frame,
            span: str,
            mode: Literal['origi', 'trans'],
            text: str
    ):
        head = tk.Label(frame, text=span, font=("Arial", 10, "bold"))
        head.grid(row=0, column=0, padx=5, columnspan=2)

        if mode == 'origi':
            attrs = {'label': 'Detail','foot': 'Arial'}
        elif mode == 'trans':
            attrs = {'label': '详情', 'foot': '黑体'}
        else:
            raise ValueError(f"Invalid mode: {mode}")

        prompt_label = tk.Label(frame, text=attrs['label'], font=(attrs['foot'], 10))
        prompt_label.grid(row=1, column=0, padx=5)

        prompt_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=50, height=5)
        prompt_text.insert(tk.INSERT, text)
        prompt_text.config(state=tk.DISABLED)
        prompt_text.grid(row=1, column=1, padx=5)

    def get_category(
            self,
            frame: tk.Frame,
            span: str,
            text: str
    ):
        head = tk.Label(frame, text=span, font=("Arial", 10, "bold"))
        head.grid(row=0, column=0, padx=5, columnspan=2)

        category_label = tk.Label(frame, text='Category', font=("Arial", 10))
        category_label.grid(row=1, column=0, padx=5)

        category_text = tk.Text(frame, wrap=tk.WORD, width=50, height=1)
        category_text.insert(tk.INSERT, text)
        category_text.config(state=tk.DISABLED)
        category_text.grid(row=1, column=1, pady=5)
        category_text.grid_remove()

        button = tk.Button(
            frame,
            text="Hide/Show",
            command=lambda: self.toggle_category(
                category_text, 1, 1)
        )
        button.grid(row=2, column=0, padx=5, columnspan=2)

    def get_component(
            self,
            frame: tk.Frame,
            obj: json,
            span: Literal['Left', 'Right', 'Relation'],
            mode: Literal['origi', 'trans']
    ):
        head = tk.Label(frame, text=span, font=("Arial", 10, "bold"))
        head.grid(row=0, column=0, padx=5, columnspan=2)

        if span == 'Relation':
            if mode == 'origi':
                attrs = {'type': 'Type', 'subtype': 'Subtype', 'description': 'Description', 'foot': 'Arial'}
            elif mode == 'trans':
                attrs = {'type': '类型', 'subtype': '子类型', 'description': '描述', 'foot': '黑体'}
            else:
                raise ValueError(f"Invalid mode: {mode}")
        else:
            if mode == 'origi':
                attrs = {'type': 'Type', 'content': 'Content', 'explanation': 'Explanation', 'foot': 'Arial'}
            elif mode == 'trans':
                attrs = {'type': '类型', 'content': '内容', 'explanation': '解释', 'foot': '黑体'}
            else:
                raise ValueError(f"Invalid mode: {mode}")

        type_label = tk.Label(frame, text=attrs['type'], font=(attrs['foot'], 10))
        type_label.grid(row=1, column=0, padx=5)

        type_text = tk.Text(frame, wrap=tk.WORD, width=50, height=1)
        type_text.insert(tk.INSERT, obj[attrs['type'].lower()])
        type_text.config(state=tk.DISABLED)
        type_text.grid(row=1, column=1, pady=5)
        type_text.grid_remove()

        content_or_subtype_label = tk.Label(frame, text=attrs['content' if span != 'Relation' else 'subtype'], font=(attrs['foot'], 10))
        content_or_subtype_label.grid(row=2, column=0, padx=5)

        content_or_subtype_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=50, height=5)
        content_or_subtype_text.insert(tk.INSERT, obj[attrs['content' if span != 'Relation' else 'subtype'].lower()])
        content_or_subtype_text.config(state=tk.DISABLED)
        content_or_subtype_text.grid(row=2, column=1, pady=5)
        if span == 'Relation':
            content_or_subtype_text.grid_remove()

        explanation_or_description_label = tk.Label(frame, text=attrs['explanation' if span != 'Relation' else 'description'], font=(attrs['foot'], 10))
        explanation_or_description_label.grid(row=3, column=0, padx=5)

        explanation_or_description_text = scrolledtext.ScrolledText(frame, wrap=tk.WORD, width=50, height=3)
        explanation_or_description_text.insert(tk.INSERT, obj[attrs['explanation' if span != 'Relation' else 'description'].lower()])
        explanation_or_description_text.config(state=tk.DISABLED)
        explanation_or_description_text.grid(row=3, column=1, pady=5)
        explanation_or_description_text.grid_remove()

        button = tk.Button(
            frame,
            text="Hide/Show",
            command=lambda: self.toggle_type_and_explanation(
                span,
                mode,
                type_text, 1, 1,
                content_or_subtype_text, 2, 1,
                explanation_or_description_text, 3, 1)
        )
        button.grid(row=4, column=0, padx=5, columnspan=2)

    @staticmethod
    def fix_grid(frame: tk.Frame, **kwargs):
        for key, value in kwargs.items():
            frame.grid_rowconfigure(int(key), minsize=value)

    @staticmethod
    def toggle_category(
            category_text: tk.Text,
            category_row: int,
            category_column: int
    ):
        if category_text.winfo_viewable():
            category_text.grid_remove()
            logging.info("[Hide] Category")
        else:
            category_text.grid(row=category_row, column=category_column, pady=5)
            logging.info("[Show] Category")

    @staticmethod
    def toggle_type_and_explanation(
            span: Literal['Left', 'Right', 'Relation'],
            mode: Literal['origi', 'trans'],
            type_text: tk.Text,
            type_row: int,
            type_column:int,
            content_or_subtype_text: tk.Text,
            content_or_subtype_row: int,
            content_or_subtype_column: int,
            explanation_or_description_text: tk.Text,
            explanation_or_description_row: int,
            explanation_or_description_column: int
    ):
        if type_text.winfo_viewable():
            type_text.grid_remove()
            if span == 'Relation':
                content_or_subtype_text.grid_remove()
            explanation_or_description_text.grid_remove()
            logging.info(f"[Hide] {span}-{mode}")
        else:
            type_text.grid(row=type_row, column=type_column, pady=5)
            if span == 'Relation':
                content_or_subtype_text.grid(row=content_or_subtype_row, column=content_or_subtype_column, pady=5)
            explanation_or_description_text.grid(row=explanation_or_description_row, column=explanation_or_description_column, pady=5)
            logging.info(f"[Show] {span}-{mode}")


    def get_objs(self) -> [json]:
        objs = []
        with open(self.sample_path, 'r', encoding='utf-8') as f:
            content = f.read()
            blocks = content.split('}\n{')
            for idx, block in enumerate(blocks):
                if idx == 0:
                    block += '}'
                elif idx == len(blocks) - 1:
                    block = '{' + block
                else:
                    block = '{' + block + '}'

                try:
                    obj = json.loads(block)
                    objs.append(obj)
                except json.JSONDecodeError:
                    print(f'Error: {block}')
        return objs

    def create_window(self):
        self.root.title("Discourse Analysis Example")
        self.root.geometry("950x1000")
        logging.info("[Info] Window created")


def configure_logging():
    file_path = os.path.abspath(f'./meta.stackoverflow.com/Annotation/gpt-4o-2024-08-06/logs/{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log')
    template = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        filename=file_path,
        level=logging.INFO,
        format=template
    )


def main():
    configure_logging()

    sample_path = os.path.abspath('./meta.stackoverflow.com/Annotation/gpt-4o-2024-08-06/samples.txt')
    gui = GUIAnnotation(sample_path)
    gui.run()


if __name__ == '__main__':
    main()
