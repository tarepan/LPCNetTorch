"""Differential Embedding"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from torch import nn, FloatTensor, int32, from_numpy, clamp, floor # pyright: ignore [reportUnknownVariableType]; pylint: disable=no-name-in-module
from omegaconf import MISSING


def proportional_emb_init(emb: nn.Embedding) -> nn.Embedding:
    """Initialize nn.Embedding with idx-proportional random initialization."""

    gain: float = .1
    n_idx = emb.weight.size()[0]

    # Random Init ~ U(-√3, +√3)
    nn.init.uniform_(emb.weight, a=-math.sqrt(3), b=math.sqrt(3))

    # Idx proportional bias :: (Idx, Emb) + (Idx) -> (Idx, Emb)
    #                                              [-127.5/256, -126.5/256, ..., +127.5/256 (+127.6 is cut)]
    bias: NDArray[np.float32] = 2 * math.sqrt(3) * np.arange(-.5*n_idx + .5, .5*n_idx -.4, dtype=np.float32) / n_idx # pyright: ignore [reportUnknownMemberType]
    emb.weight = nn.parameter.Parameter(gain * (emb.weight + from_numpy(bias).unsqueeze(-1)))
    # idx0 is 0.1 * (U(-√3, +√3) - ~0.5*2√3), so about [-0.346,      0]
    # idxL is 0.1 * (U(-√3, +√3) + ~0.5*2√3), so about [     0, +0.346]
    return emb


@dataclass
class ConfDifferentialEmbedding:
    """Configuration of DifferentialEmbedding.
    Args:
        codebook_size - The size of codebook (dictionary size)
        ndim_emb - The size of embedding dimension (embedding vector size)
        ndims_i - The number of input's dimension
    """
    codebook_size: int = MISSING
    ndim_emb: int = MISSING
    ndims_i: int = MISSING

class DifferentialEmbedding(nn.Module):
    """Differential Embedding with interpolation."""

    def __init__(self, conf: ConfDifferentialEmbedding):
        """Init."""
        super().__init__()

        # Dimension size of weight
        self._dim_weight = [-1 for _ in range(conf.ndims_i)] + [conf.ndim_emb]

        self._max_idx = conf.codebook_size - 1
        self.emb = proportional_emb_init(nn.Embedding(conf.codebook_size, conf.ndim_emb))

    # Typing of PyTorch forward API is poor.
    def forward(self, continuous_idx: FloatTensor) -> FloatTensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """(PT API) Forward a batch.

        Args:
            continuous_idx :: (...) - Input tensor, last dimension is continous index (embedding target)
        """
        # :: (...) -> (..., 1) -> (..., Emb)
        weight = (continuous_idx - floor(continuous_idx)).unsqueeze(-1).expand(self._dim_weight)
        idx_d = continuous_idx.to(int32)
        return (1-weight) * self.emb(idx_d) + weight * self.emb(clamp(idx_d+1, 0, self._max_idx))
