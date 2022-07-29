"""Test diffemb"""

from torch import nn, tensor, equal, sum # pylint: disable=no-name-in-module,redefined-builtin

from .diffemb import DifferentialEmbedding, ConfDifferentialEmbedding


diff_emb = DifferentialEmbedding(ConfDifferentialEmbedding(5, 3))
diff_emb.emb.weight = nn.parameter.Parameter(tensor([
    [  0.,   0.,   0.],
    [  1.,   1.,   1.],
    [  2.,   2.,   2.],
    [  3.,   3.,   3.],
    [100., 100., 100.],
]))

def test_diff_emb_discrete_idx():
    """Test `DifferentialEmbedding` with discrete index."""

    idx_d = tensor([1.])
    emb_gt = tensor([[1., 1., 1.]])
    emb_estim = diff_emb(idx_d)
    print(diff_emb.emb.weight)
    assert equal(emb_estim, emb_gt)

def test_diff_emb_continuous_idx():
    """Test `DifferentialEmbedding` with continuous index."""

    idx_c = tensor([2.5], requires_grad=True)
    emb_gt = tensor([[2.5, 2.5, 2.5]])
    emb_estim = diff_emb(idx_c)
    assert equal(emb_estim, emb_gt)

    sum(emb_estim).backward() # pyright: ignore [reportUnknownMemberType]
