"""Test tree sampling"""

from torch import Tensor, tensor, float64, abs, all, equal, log # pylint: disable=no-name-in-module,redefined-builtin

from .tree_sampling import _cprobs_to_cdist, tree_to_pdf, tree_to_logpdf # pyright: ignore [reportPrivateUsage]


def test_cprobs_to_cdist():
    """Test `_cprobs_to_cdist` function."""

    bit_depth = 3

    # P(bit_k|bit_<k) :: (Batch, Time, Bit)
    bit_cond_probs: Tensor = tensor([
        [ # batch_0
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,], # t_0
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,], # t_1
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,], # t_2
        #   | - | L1 |    L2   |        L3         |
        ],
        [ # batch_1
            [1.0, 1.0, 1.0, 1.0, 0.1, 0.2, 0.3, 0.4,], # t_0
            [1.0, 1.0, 1.0, 1.0, 0.5, 0.6, 0.7, 0.8,], # t_1
            [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,], # t_2
        #   | - | L1 |    L2   |        L3         |
        ],
    ], dtype=float64)

    # (B=2, T_s=3, Prob=4)
    layer = 3
    assert 2**(layer-1) == 4, "Prerequisites"
    assert 2**layer == 8, "Prerequisites"
    l3: Tensor = bit_cond_probs[:, :, 2**(layer-1) : 2**layer]

    l3_gt: Tensor = tensor([
        [ # batch_0
            [1.0-1.0, 1.0, 1.0-1.0, 1.0, 1.0-1.0, 1.0, 1.0-1.0, 1.0,], # t_0
            [1.0-1.0, 1.0, 1.0-1.0, 1.0, 1.0-1.0, 1.0, 1.0-1.0, 1.0,], # t_1
            [1.0-1.0, 1.0, 1.0-1.0, 1.0, 1.0-1.0, 1.0, 1.0-1.0, 1.0,], # t_2
        #   |   L      H      L      H      L      H      L      H  |
        ],
        [ # batch_1
            [1.0-0.1, 0.1, 1.0-0.2, 0.2, 1.0-0.3, 0.3, 1.0-0.4, 0.4,], # t_0
            [1.0-0.5, 0.5, 1.0-0.6, 0.6, 1.0-0.7, 0.7, 1.0-0.8, 0.8,], # t_1
            [1.0-0.0, 0.0, 1.0-0.0, 0.0, 1.0-0.0, 0.0, 1.0-0.0, 0.0,], # t_2
        #   |   L      H      L      H      L      H      L      H  |
        ],
    ], dtype=float64)

    l3_cond_probs = _cprobs_to_cdist(l3, bit_depth, layer)

    assert equal(l3_cond_probs, l3_gt)


def test_tree_to_pdf():
    """Test `tree_to_pdf` function."""

    # P(bit_k|bit_<k) :: (B=1, T_s=1, Cond=2**3)
    bit_cond_probs_all: Tensor = tensor([
        [ # batch_1
            [1.0, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4,], # t_0
        #   | - | L1 |    L2   |        L3         |
        ],
    ], dtype=float64)

    # P(level) :: (B=1, T_s=1, Dist=2**3)
    joint_dist_gt: Tensor = tensor([
        [ # batch_0
            [ # time_0
                (1.0-0.3) * (1.0-0.4) * (1.0-0.1), # 000
                (1.0-0.3) * (1.0-0.4) *      0.1 , # 001
                (1.0-0.3) *      0.4  * (1.0-0.2), # 010
                (1.0-0.3) *      0.4  *      0.2 , # 011
                     0.3  * (1.0-0.5) * (1.0-0.3), # 100
                     0.3  * (1.0-0.5) *      0.3 , # 101
                     0.3  *      0.5  * (1.0-0.4), # 110
                     0.3  *      0.5  *      0.4 , # 111
        ],
    ]], dtype=float64)

    joint_dist_estim = tree_to_pdf(bit_cond_probs_all)

    assert equal(joint_dist_estim, joint_dist_gt)


def test_tree_to_logpdf():
    """Test `tree_to_logpdf` function."""

    # P(bit_k|bit_<k) :: (B=1, T_s=1, Cond=2**3)
    bit_cond_probs_all: Tensor = tensor([
        [ # batch_1
            [1.0, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4,], # t_0
        #   | - | L1 |    L2   |        L3         |
        ],
    ], dtype=float64)

    # P(level) :: (B=1, T_s=1, Dist=2**3)
    logp_gt: Tensor = log(tensor([
        [ # batch_0
            [ # time_0
                (1.0-0.3) * (1.0-0.4) * (1.0-0.1), # 000
                (1.0-0.3) * (1.0-0.4) *      0.1 , # 001
                (1.0-0.3) *      0.4  * (1.0-0.2), # 010
                (1.0-0.3) *      0.4  *      0.2 , # 011
                     0.3  * (1.0-0.5) * (1.0-0.3), # 100
                     0.3  * (1.0-0.5) *      0.3 , # 101
                     0.3  *      0.5  * (1.0-0.4), # 110
                     0.3  *      0.5  *      0.4 , # 111
        ],
    ]], dtype=float64))

    logp_estim = tree_to_logpdf(bit_cond_probs_all)

    # There are tiny numerical difference between 'logΠp' and 'Σlogp'
    assert all(abs(logp_estim - logp_gt) < 0.0000000001)
