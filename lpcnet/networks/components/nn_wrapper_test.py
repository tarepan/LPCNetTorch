"""Test wrappers"""

from torch import tensor, equal # pylint: disable=no-name-in-module

from .nn_wrapper import TransposeLast


def test_transposewrapper():
    """Test `TransposeLast`."""

    trans = TransposeLast()

    # (B, T, F)
    ipt = tensor([
        # batch0
        [
            # F0   F1
            [ 1.,  2.,], # Time0
            [ 3.,  4.,], # Time1
            [ 5.,  6.,], # Time2
        ],
        # batch1
        [
            [ 7.,  8.,], # Time0
            [ 9., 10.,], # Time1
            [11., 12.,], # Time2
        ]
    ])

    opt_gt = tensor([
        # batch0
        [
            # T0  T1   T2
            [ 1.,  3.,  5.,], # Feat0
            [ 2.,  4.,  6.,], # Feat1
        ],
        # batch1
        [
            [ 7.,  9., 11.,], # Feat0
            [ 8., 10., 12.,], # Feat1
        ]
    ])

    # Transpose
    assert equal(trans(ipt), opt_gt)

    # T(T) == E
    assert equal(trans(trans(ipt)), ipt)
