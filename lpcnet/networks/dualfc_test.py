"""Test DualFC"""

from torch import nn, tensor, equal           # pylint: disable=no-name-in-module
from torch.nn.functional import tanh, sigmoid # pyright: ignore [reportUnknownVariableType]

from .dualfc import DualFC, ConfDualFC


def test_dualfc_output():
    """Test calculation correctness of DualFC.
    """

    # sigmoid(a_1 ○ tanh(W_1  x ) + a_2 ○ tanh( W_2  x ))
    # sigmoid(1.0 * tanh(2.0*1.0) + 2.0 * tanh(-0.5*1.0))

    ipt = tensor([1.0])
    w_1, w_2 = [2.0], [-0.5]
    a_1, a_2 = 1.0, 2.0

    o_gt = sigmoid(tensor([a_1]) * tanh(tensor(w_1) * ipt) + tensor([a_2]) * tanh(tensor(w_2) * ipt))

    dualfc = DualFC(ConfDualFC(1, 1))
    dualfc.w_1_2.weight = nn.parameter.Parameter(tensor([w_1, w_2]))
    dualfc.a_1_2 = nn.parameter.Parameter(tensor([a_1, a_2]))

    o_estim = dualfc(ipt)
    assert tuple(o_estim.size()) == (1,)
    assert equal(o_estim, o_gt)
