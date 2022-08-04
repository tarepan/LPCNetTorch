"""Test diffemb"""

from torch import nn, tensor, equal, sum # pylint: disable=no-name-in-module,redefined-builtin

from .diffemb import DifferentialEmbedding, ConfDifferentialEmbedding


def init(diff_emb: DifferentialEmbedding) -> DifferentialEmbedding:
    """Init for test."""
    diff_emb.emb.weight = nn.parameter.Parameter(tensor([
        [  0.,   0.,   0.],
        [  1.,   1.,   1.],
        [  2.,   2.,   2.],
        [  3.,   3.,   3.],
        [100., 100., 100.],
    ]))
    return diff_emb


def test_diff_emb_discrete_idx():
    """Test `DifferentialEmbedding` with discrete index."""

    diff_emb = init(DifferentialEmbedding(ConfDifferentialEmbedding(5, 3, ndims_i=1)))
    idx_d = tensor([1.])
    emb_gt = tensor([[1., 1., 1.]])
    emb_estim = diff_emb(idx_d)
    assert equal(emb_estim, emb_gt)

def test_diff_emb_continuous_idx():
    """Test `DifferentialEmbedding` with continuous index."""

    diff_emb = init(DifferentialEmbedding(ConfDifferentialEmbedding(5, 3, ndims_i=1)))
    idx_c = tensor([2.5], requires_grad=True)
    emb_gt = tensor([[2.5, 2.5, 2.5]])
    emb_estim = diff_emb(idx_c)
    assert equal(emb_estim, emb_gt)

    sum(emb_estim).backward() # pyright: ignore [reportUnknownMemberType]

def test_diff_emb_high_dim():
    """Test `DifferentialEmbedding` with high dimensional input."""

    diff_emb = init(DifferentialEmbedding(ConfDifferentialEmbedding(5, 3, ndims_i=3)))
    idx_d = tensor([
        # batch0
        [#   i0  i1  i2
            [1. , 2. , 3. ,], # t0
            [1.7, 0.1, 3.0,], # t1
        ]
    ])
    emb_gt = tensor([
        # batch0
        [
            # t0
            [
                [1. , 1. , 1. ], # Emb(i0)
                [2. , 2. , 2. ], # Emb(i1)
                [3. , 3. , 3. ], # Emb(i2)
            ],
            # t1
            [
                [1.7, 1.7, 1.7], # Emb(i0)
                [0.1, 0.1, 0.1], # Emb(i1)
                [3.0, 3.0, 3.0], # Emb(i2)
            ],
        ]
    ])
    emb_estim = diff_emb(idx_d)
    print(emb_estim)
    assert equal(emb_estim, emb_gt)
