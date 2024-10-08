# coding=utf-8
# @Author: Fulai Cui (cuifulai@mail.hfut.edu.cn)
# @Time: 2024/9/30 8:55
import functools
import typing

import numpy as np
import pandas as pd
import inspect

import torch


def _convert_to_list_index(
        index: typing.Union[int, slice, np.array],
        length: int
):
    if isinstance(index, int):
        index = [index]
    elif isinstance(index, slice):
        index = list(range(*index.indices(length)))
    return index


class DataPack:
    def __init__(
            self,
            relation: pd.DataFrame,
            left: pd.DataFrame,
            right_id: pd.DataFrame,
            right: pd.DataFrame,
            comment: pd.DataFrame,
            ping: pd.DataFrame,
            extend: pd.DataFrame,
            feature: pd.DataFrame,
            max_length: int,
            max_seq_length: int,
    ):
        self._relation = relation
        self._left = left
        self._right_id = right_id
        self._right = right
        self._comment = comment
        self._ping = ping
        self._extend = extend
        self._feature = feature
        self._max_length = max_length
        self._max_seq_length = max_seq_length

    @property
    def has_label(self) -> bool:
        return 'label' in self._relation.columns

    def __len__(self) -> int:
        return self._relation.shape[0]

    @property
    def frame(self) -> 'DataPack.FrameView':
        return DataPack.FrameView(self)

    def unpack(self) -> typing.Tuple[typing.Dict[str, np.array], typing.Optional[np.array]]:
        frame = self.frame()

        columns = list(frame.columns)
        if self.has_label:
            columns.remove('label')
            y = np.vstack(np.asarray(frame['label']))
        else:
            y = None

        x = frame[columns].to_dict(orient='list')

        max_seq_length = max(max([len(comment) if comment is not None else 0 for comment in x['comment']]), 1)

        for key, val in x.items():
            if key == 'comment':
                comments = []
                for comment in val:
                    seq_length = len(comment) if comment is not None else 0
                    if seq_length == 0:
                        comment = torch.tensor([[101, 102] + [0] * (self._max_length - 2)] * max_seq_length, dtype=torch.long)
                    else:
                        comment = torch.cat((comment, torch.tensor([[101, 102] + [0] * (self._max_length - 2)] * (max_seq_length - seq_length), dtype=torch.long)), dim=0)
                    comments.append(comment[: self._max_seq_length])
                x[key] = comments
            elif key == 'ping':
                pings = []
                for ping in val:
                    seq_length = len(ping) if ping is not None else 0
                    if seq_length == 0:
                        ping = [0] * max_seq_length
                    else:
                        ping = ping + [0] * (max_seq_length - seq_length)
                    pings.append(ping[: self._max_seq_length])
                x[key] = pings
            elif key == 'right_id' or key == 'extend':
                continue
            else:
                x[key] = val

        return x, y

    def __getitem__(self, index: typing.Union[int, slice, np.array]) -> 'DataPack':
        index = _convert_to_list_index(index, len(self))
        relation = self._relation.loc[index].reset_index(drop=True)
        left = self._left.loc[relation['id_left'].unique()]
        right_id = self._right_id.loc[relation['id_right'].unique()]
        right = self._right.loc[relation['id_right'].unique()]
        comment = self._comment.loc[relation['id_right'].unique()]
        ping = self._ping.loc[relation['id_right'].unique()]
        extend = self._extend.loc[relation['id_left'].unique()]
        feature = self._feature.loc[relation['id_right'].unique()]
        return DataPack(
            relation=relation.copy(),
            left=left.copy(),
            right_id=right_id.copy(),
            right=right.copy(),
            comment=comment.copy(),
            ping=ping.copy(),
            extend=extend.copy(),
            feature=feature.copy(),
            max_length=self._max_length,
            max_seq_length=self._max_seq_length
        )

    @property
    def relation(self) -> pd.DataFrame:
        return self._relation

    @relation.setter
    def relation(self, value):
        self._relation = value

    @property
    def left(self) -> pd.DataFrame:
        return self._left

    @property
    def right_id(self) -> pd.DataFrame:
        return self._right_id

    @right_id.setter
    def right_id(self, value):
        self._right_id = value

    @property
    def right(self) -> pd.DataFrame:
        return self._right

    @property
    def comment(self) -> pd.DataFrame:
        return self._comment

    @property
    def ping(self) -> pd.DataFrame:
        return self._ping

    @property
    def extend(self) -> pd.DataFrame:
        return self._extend

    @extend.setter
    def extend(self, value):
        self._extend = value

    @property
    def feature(self) -> pd.DataFrame:
        return self._feature

    @feature.setter
    def feature(self, value):
        self._feature = value

    def copy(self) -> 'DataPack':
        return DataPack(
            relation=self._relation.copy(),
            left=self._left.copy(),
            right_id=self._right_id.copy(),
            right=self._right.copy(),
            comment=self._comment.copy(),
            ping=self._ping.copy(),
            extend=self._extend.copy(),
            feature=self._feature.copy(),
            max_length=self._max_length,
            max_seq_length=self._max_seq_length
        )

    @staticmethod
    def _optional_inplace(func):
        """
        Decorator that adds `inplace` key word argument to a method.

        Decorate any method that modifies inplace to make that inplace change
        optional.
        """
        doc = ":param inplace: `True` to modify inplace, `False` to return " \
              "a modified copy. (default: `False`)"

        def _clean(s):
            return s.replace(' ', '').replace('\n', '')

        if _clean(doc) not in _clean(inspect.getdoc(func)):
            raise NotImplementedError(
                f"`inplace` parameter of {func} not documented.\n"
                f"Please add the following line to its documentation:\n{doc}")

        @functools.wraps(func)
        def wrapper(
                self, *args, inplace: bool = False, **kwargs
        ) -> typing.Optional['DataPack']:

            if inplace:
                target = self
            else:
                target = self.copy()

            func(target, *args, **kwargs)

            if not inplace:
                return target

        return wrapper

    @_optional_inplace
    def drop_empty(self, inplace: bool = True):
        """
        Process empty data by removing corresponding rows.

        :param inplace: `True` to modify inplace, `False` to return a modified
            copy. (default: `False`)
        """
        empty_left_id = self._left[
            self._left['length_left'] == 0].index.tolist()
        empty_right_id = self._right[
            self._right['length_right'] == 0].index.tolist()
        empty_id = self._relation[
            self._relation['id_left'].isin(empty_left_id) | self._relation[
                'id_right'].isin(empty_right_id)
        ].index.tolist()

        self._relation = self._relation.drop(empty_id)
        self._left = self._left.drop(empty_left_id)
        self._right_id = self._right_id.drop(empty_right_id)
        self._right = self._right.drop(empty_right_id)
        self._comment = self._comment.drop(empty_right_id)
        self._ping = self._ping.drop(empty_right_id)
        self._extend = self._extend.drop(empty_left_id)
        self._feature = self._feature.drop(empty_right_id)
        self._relation.reset_index(drop=True, inplace=inplace)

    @_optional_inplace
    def shuffle(self, inplace: bool = True):
        """
        Shuffle the data pack by shuffling the relation column.

        :param inplace: `True` to modify inplace, `False` to return a modified
            copy. (default: `False`)
        """
        self._relation = self._relation.sample(frac=1)
        self.relation.reset_index(drop=True, inplace=inplace)

    @_optional_inplace
    def drop_label(self, inplace: bool = False):
        """
        Remove `label` column from the data pack.

        :param inplace: `True` to modify inplace, `False` to return a modified
            copy. (default: `False`)
        """
        self._relation = self._relation.drop(columns='label', inplace=inplace)

    def append_text_length(self):
        """
        Append `length_left` and `length_right` columns.

        """
        self.apply_on_text(len, rename=('length_left', 'length_right'))

    def apply_on_text(
            self,
            func: typing.Callable,
            mode: str = 'both',
            rename: typing.Optional[tuple] = None,
    ):
        """
        Apply `func` to text columns based on `mode`.

        :param func: The function to apply.
        :param mode: One of "both", "left" and "right".
        :param rename: If set, use new names for results instead of replacing
            the original columns. To set `rename` in "both" mode, use a tuple
            of `str`, e.g. ("text_left_new_name", "text_right_new_name").
        """
        if mode == 'both':
            self._apply_on_text_both(func, rename)
        elif mode == 'left':
            self._apply_on_text_left(func, rename)
        elif mode == 'right':
            self._apply_on_text_right(func, rename)
        else:
            raise ValueError(f"{mode} is not a valid mode type."
                             f"Must be one of `left` `right` `both`.")

    def _apply_on_text_right(self, func, rename):
        name = rename or 'text_right'
        self._right[name] = self._right['text_right'].apply(func)

    def _apply_on_text_left(self, func, rename):
        name = rename or 'text_left'
        self._left[name] = self._left['text_left'].apply(func)

    def _apply_on_text_both(self, func, rename):
        left_name, right_name = rename or ('text_left', 'text_right')
        self._apply_on_text_left(func, rename=left_name)
        self._apply_on_text_right(func, rename=right_name)

    class FrameView:
        def __init__(self, data_pack: 'DataPack'):
            self._data_pack = data_pack

        def __getitem__(self, index: typing.Union[int, slice, np.array]) -> pd.DataFrame:
            dp = self._data_pack
            index = _convert_to_list_index(index, len(dp))
            left_df = dp.left.loc[dp.relation['id_left'][index]].reset_index()
            right_id_df = dp.right_id.loc[dp.relation['id_right'][index]].reset_index()
            right_df = dp.right.loc[dp.relation['id_right'][index]].reset_index()
            comment_df = dp.comment.loc[dp.relation['id_right'][index]].reset_index()
            ping_df = dp.ping.loc[dp.relation['id_right'][index]].reset_index()
            extend_df = dp.extend.loc[dp.relation['id_left'][index]].reset_index()
            feature_df = dp.feature.loc[dp.relation['id_right'][index]].reset_index()
            joined_table = left_df.join(right_df)
            for column in dp.relation.columns:
                if column not in ['id_left', 'id_right']:
                    labels = dp.relation[column][index].to_frame()
                    labels = labels.reset_index(drop=True)
                    joined_table = joined_table.join(labels)
            joined_table = joined_table.join(right_id_df, rsuffix='_right_id')
            joined_table.drop(columns=['id_right_right_id'], inplace=True)
            joined_table = joined_table.join(comment_df, rsuffix='_comment')
            joined_table.drop(columns=['id_right_comment'], inplace=True)
            joined_table = joined_table.join(ping_df, rsuffix='_ping')
            joined_table.drop(columns=['id_right_ping'], inplace=True)
            joined_table = joined_table.join(extend_df, rsuffix='_extend')
            joined_table.drop(columns=['id_left_extend'], inplace=True)
            joined_table = joined_table.join(feature_df, rsuffix='_feature')
            joined_table.drop(columns=['id_right_feature'], inplace=True)
            return joined_table

        def __call__(self):
            return self[:]



def main():
    pass


if __name__ == '__main__':
    main()
