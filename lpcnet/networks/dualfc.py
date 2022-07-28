"""DualFC module."""

from dataclasses import dataclass

from torch import nn, Tensor, ones            # pylint: disable=no-name-in-module
from torch.nn.functional import tanh, sigmoid # pyright: ignore [reportUnknownVariableType]
from omegaconf import MISSING


@dataclass
class ConfDualFC:
    """Configuration of DualFC.
    Args:
        ndim_i_feat - The size of input's  feature dimension (last dim)
        ndim_o_feat - The size of output's feature dimension (last dim)
    """
    ndim_i_feat: int = MISSING
    ndim_o_feat: int = MISSING

class DualFC(nn.Module):
    """DualFC = sigmoid(a_1 ○ tanh(W_1 x) + a_2 ○ tanh(W_2 x))."""

    def __init__(self, conf: ConfDualFC):
        """Init."""

        super().__init__()
        self._ndim_o_feat = conf.ndim_o_feat

        # Calculation hack: 'a_1 ○ tanh(W_1 x) + a_2 ○ tanh(W_2 x)' -> '(a_1_2 ○ (tanh(W_1_2 x)).split()'

        ndim_o_fc = conf.ndim_o_feat*2
        self.w_1_2 = nn.Linear(conf.ndim_i_feat, ndim_o_fc, bias=True)
        # todo: Is xavier init affected by calculation hack ...? ('fout' changed)
        nn.init.xavier_uniform_(self.w_1_2.weight)
        nn.init.zeros_(self.w_1_2.bias)
        self.a_1_2 = nn.parameter.Parameter(ones(ndim_o_fc))

    # Typing of PyTorch forward API is poor.
    def forward(self, ipt: Tensor) -> Tensor: # pyright: ignore [reportIncompatibleMethodOverride]
        """
        Args:
            ipt :: (..., Feat=i_feat) - Input,  last dimension is feature
        Returns:
            opt :: (..., Feat=o_feat) - Output, last dimension is feature
        """

        # Activation 'a_1_2 ○ (tanh(W_1_2 x))'
        #     :: (..., Feat=i_feat) -> (..., Feat=2*o_feat)
        act_1_2: Tensor = self.a_1_2 * tanh(self.w_1_2(ipt))

        # Split act_1_2 -> (act1, act2) :: (..., Feat=2*o_feat) -> ((..., Feat=o_feat), (..., Feat=o_feat))
        fc1, fc2 = act_1_2.split((self._ndim_o_feat, self._ndim_o_feat), dim=-1) # pyright: ignore [reportUnknownVariableType, reportUnknownMemberType]

        # Sum+Act :: ((..., Feat=o_feat), (..., Feat=o_feat)) -> (..., Feat=o_feat)
        opt = sigmoid(fc1 + fc2)

        return opt
