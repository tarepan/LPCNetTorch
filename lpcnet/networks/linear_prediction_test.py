"""Test of linear_prediction."""

from torch import tensor, equal                  # pylint: disable=no-name-in-module

from .linear_prediction import linear_prediction


def test_linear_prediction():
    """Test `linear_prediction`."""
    order = 4
    # (T=5, Order=4)
    coeff_series = tensor([
        [1.0, 0.5, 1.0, 0.1],
        [1.0, 0.5, 1.0, 0.1],
        [0.5, 1.0, 1.0, 0.1],
        [0.5, 1.0, 0.5, 0.1],
        [0.5, 1.0, 1.0, 0.1],
    ])
    # (B=1, T=5)
    s_t_1_series = tensor([[1., 2., 3., 4., 5.,]])
    p_t_series_gt = tensor([[
        1.0 * 1.,
        1.0 * 2. + 0.5 * 1.,
        0.5 * 3. + 1.0 * 2. + 1.0 * 1.,
        0.5 * 4. + 1.0 * 3. + 0.5 * 2. + 0.1 * 1.,
        0.5 * 5. + 1.0 * 4. + 1.0 * 3. + 0.1 * 2.,
    ]])

    # Prerequisites
    assert coeff_series.size()[1] == order, "Order not matched."
    assert s_t_1_series.size()[1] == coeff_series.size()[0], "T not matched"

    # Test
    p_t_series_pred = linear_prediction(s_t_1_series, coeff_series, order)
    assert equal(p_t_series_pred, p_t_series_gt)
