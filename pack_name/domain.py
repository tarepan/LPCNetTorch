"""Domain"""


from typing import List, Tuple

from torch import Tensor # pyright: ignore [reportUnknownVariableType] ; because of PyTorch ; pylint: disable=no-name-in-module


"""
(delele here when template is used)

[Design Notes - Data type]
    Data is finally transformed by collate_fn in DataLoader, then consumed by x_step of the Model (Network consumes some of them).
    Both data-side and model-side depends on the data type.
    For this reason, the data type is separated as domain.
"""


# Data batch

## :: (Batch=b, T=t, 1) - hoge hoge
HogeBatched = Tensor
## :: (Batch=b, T=t, 1) - fuga fuga
FugaBatched = Tensor
## :: (L=b,)            - Non-padded length of items in the FugaBatched
LenFuga = List[int]

## the batch
HogeFugaBatch = Tuple[HogeBatched, FugaBatched, LenFuga]
