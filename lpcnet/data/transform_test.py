import numpy as np
from numpy.typing import NDArray

from .transform import emulate_noisy_sample_frame


def test_emulate_noisy_sample_frame():
    """Test `emulate_noisy_sample_frame` function."""


    """
    t           -2   -1        0        1        2         3
    s_t_clean    0    0     +100     +200     +270    +20000
    p_t_noisy    -    -        0.0   +145.~   +288.~     386.~
    e_t_ideal    -    -     +100.0    +54.~    -18.~  +19613.~
    e_q_t_idl    -    -      +13       +8       -3      +116
    noise        -    -       +0       +0       +0        +0
    s_t_noisy    0.   0.     +97.~   +198.~   +270.~  +19818.~
    s_t_clean    0    0     +100     +200     +270    +20000
    """
    coeffs            : NDArray[np.float32] = np.array([+1.5, -0.1,],                                                        dtype=np.float32) # pyright: ignore [reportUnknownMemberType]; because of numpy
    s_m1m2_noisy_ls16c: NDArray[np.float32] = np.array([0.0, 0.0,],                                                          dtype=np.float32) # pyright: ignore [reportUnknownMemberType]; because of numpy
    s_t_clean_ls16q   : NDArray[np.int16]   = np.array([+100,          +200,          +270,           +20000,],              dtype=np.int16)   # pyright: ignore [reportUnknownMemberType]; because of numpy
    s_t_noisy_gt_ls16c: NDArray[np.float32] = np.array([ +97.17988546, +198.99708314, +270.943024464, +19818.389962832003,], dtype=np.float32) # pyright: ignore [reportUnknownMemberType]; because of numpy
    noise_ms8q        : NDArray[np.int8]    = np.array([   0,             0,             0,                0,],              dtype=np.uint8)   # pyright: ignore [reportUnknownMemberType]; because of numpy

    s_t_noisy_estim = emulate_noisy_sample_frame(s_t_clean_ls16q, coeffs, noise_ms8q, s_m1m2_noisy_ls16c)

    np.testing.assert_allclose(s_t_noisy_gt_ls16c, s_t_noisy_estim) # pyright: ignore [reportUnknownMemberType]; because of numpy
