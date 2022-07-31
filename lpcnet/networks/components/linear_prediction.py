"""Linear Prediction"""

from torch import Tensor, cat, sum # pylint: disable=no-name-in-module,redefined-builtin
from torch.nn.functional import pad


def linear_prediction(s_t_n: Tensor, coeffs: Tensor) -> Tensor:
    """
    Args:
        s_t_n  :: (B, Order) - sample t-1 ~ t-ORDER
        coeffs :: (Order)    - LP coefficients
    Returns:
        p_t    :: (B,)       - LinearPrediction
    """

    return sum(s_t_n * coeffs, dim=1)


def linear_prediction_series(s_t_1_series: Tensor, coeff_series: Tensor, order: int) -> Tensor:
    """
    Args:
        s_t_1_series :: (B,     T=t_s) - Lagged/Delayed sample series (waveform)
        coeff_series :: (T=t_s, Order) - LPcoefficient series
        order                          - Order of Linear Prediction (=='Order')
    Returns:
        p_t_series   :: (B,     T=t_s) - LinearPrediction series
    """

    # [Design Notes - LP by Lagged series sum]
    #
    #     p_t_series      p1  p2 ... p15  p16  p17
    #     ----------------------------------------
    #     s_t_1_series    s0  s1 ... s14  s15  s16
    #     s_t_2_series     0  s0 ... s13  s14  s15
    #     ...
    #     s_t_16_series    0   0 ...   0   s0   s1

    # s_t_1 ~ s_t_16 :: (B, T=t) -> (B, T=t, 1)[] -> (B, T=t, Delay=n_coeff)
    s_t_1_series = s_t_1_series.unsqueeze(-1)
    s_t_n_series = cat([s_t_1_series] + [pad(s_t_1_series[:, :-i], (0, 0, i, 0)) for i in range(1, order)], dim=-1)

    # p_t = Σi=1 (a_i*s_{t-i}) :: (B, T, Delay) -> (B, T)
    p_t_series = sum(coeff_series * s_t_n_series, dim=-1)

    return p_t_series
