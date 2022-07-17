"""The Child sub-module"""


from typing import List
from dataclasses import dataclass

from torch import Tensor
import torch.nn as nn
from omegaconf import MISSING


@dataclass
class ConfChild:
    """Configuration of the Child sub-module.
    Args:
        dim_i: Dimension size of input
        dim_o: Dimension size of output
        dropout: Dropout rate (0 means No-dropout)
    """
    dim_i: int = MISSING
    dim_o: int = MISSING
    dropout: float = MISSING

class Child(nn.Module):
    """The Network.
    """
    def __init__(self, conf: ConfChild):
        super().__init__()

        layers: List[nn.Module] = []
        layers += [nn.Linear(conf.dim_i, conf.dim_o), nn.ReLU()]
        layers += [nn.Dropout(conf.dropout)] if conf.dropout > 0. else []
        self.fc1 = nn.Sequential(*layers)

    # Typing of PyTorch forward API is poor.
    def forward(self, i_pred: Tensor) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward a batch.

        Arguments:
            i_pred :: (Batch, T, Feat=dim_i) - Input
        Returns:
            o_pred :: (Batch, T, Feat=dim_o) - Prediction
        """
        # :: (Batch, T, Feat=dim_i) -> (Batch, T, Feat=dim_o)
        o_pred = self.fc1(i_pred)

        return o_pred

    def generate(self, i_pred: Tensor) -> Tensor:
        """Run inference with a batch.

        Arguments:
            i_pred :: (Batch, T, Feat=dim_i) - Input
        Returns:
            o_pred :: (Batch, T, Feat=dim_o) - Prediction
        """

        # :: (Batch, T, Feat=dim_i) -> (Batch, T, Feat=dim_o)
        o_pred = self.fc1(i_pred)

        return o_pred
