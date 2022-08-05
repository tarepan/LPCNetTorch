"""Test μ-law conversion"""

from torch import tensor, float32, int32, int64, equal # pylint: disable=no-name-in-module,redefined-builtin

from .mulaw import lin2mlaw, lin2mlawpcm, linear_s16pcm, mlaw2lin


def test_linear_pcm():
    """Test `linear_pcm`."""

    i_toomin, i_toomin_gt = tensor(-32768.1, dtype=float32), tensor(-32768, dtype=int32)
    i_min   , i_min_gt    = tensor(-32768. , dtype=float32), tensor(-32768, dtype=int32)
    i_small , i_small_gt  = tensor(-32767. , dtype=float32), tensor(-32767, dtype=int32)
    i_zero  , i_zero_gt   = tensor(     0. , dtype=float32), tensor(     0, dtype=int32)
    i_rounds, i_rounds_gt = tensor(     0.1, dtype=float32), tensor(     0, dtype=int32)
    i_roundl, i_roundl_gt = tensor(     0.7, dtype=float32), tensor(     1, dtype=int32)
    i_big   , i_big_gt    = tensor(+31266. , dtype=float32), tensor(+31266, dtype=int32)
    i_max   , i_max_gt    = tensor(+32767. , dtype=float32), tensor(+32767, dtype=int32)
    i_toomax, i_toomax_gt = tensor(+32767.1, dtype=float32), tensor(+32767, dtype=int32)

    assert equal(linear_s16pcm(i_toomin), i_toomin_gt)
    assert equal(linear_s16pcm(i_min)   , i_min_gt)
    assert equal(linear_s16pcm(i_small) , i_small_gt)
    assert equal(linear_s16pcm(i_zero)  , i_zero_gt)
    assert equal(linear_s16pcm(i_rounds), i_rounds_gt)
    assert equal(linear_s16pcm(i_roundl), i_roundl_gt)
    assert equal(linear_s16pcm(i_big)   , i_big_gt)
    assert equal(linear_s16pcm(i_max)   , i_max_gt)
    assert equal(linear_s16pcm(i_toomax), i_toomax_gt)


def test_lin2mlaw():
    """Test `lin2mlaw`."""

    i_toomin, i_toomin_gt = tensor(-32768.1, dtype=float32), tensor(  0.          , dtype=float32)
    i_min   , i_min_gt    = tensor(-32768. , dtype=float32), tensor(  0.          , dtype=float32)
    i_small , i_small_gt  = tensor(-32767. , dtype=float32), tensor(  0.0007019043, dtype=float32)
    i_zero  , i_zero_gt   = tensor(     0. , dtype=float32), tensor(128.          , dtype=float32)
    i_big   , i_big_gt    = tensor(+31266. , dtype=float32), tensor(254.92125     , dtype=float32)
    i_max   , i_max_gt    = tensor(+32767. , dtype=float32), tensor(255.          , dtype=float32)
    i_toomax, i_toomax_gt = tensor(+32767.1, dtype=float32), tensor(255.          , dtype=float32)

    assert equal(lin2mlaw(i_toomin), i_toomin_gt)
    assert equal(lin2mlaw(i_min)   , i_min_gt)
    assert equal(lin2mlaw(i_small) , i_small_gt)
    assert equal(lin2mlaw(i_zero)  , i_zero_gt)
    assert equal(lin2mlaw(i_big)   , i_big_gt)
    assert equal(lin2mlaw(i_max)   , i_max_gt)
    assert equal(lin2mlaw(i_toomax), i_toomax_gt)


def test_lin2mlawpcm():
    """Test `lin2mlawpcm`."""

    i_toomin, i_toomin_gt = tensor(-32768.1, dtype=float32), tensor(  0, dtype=int64)
    i_min   , i_min_gt    = tensor(-32768. , dtype=float32), tensor(  0, dtype=int64)
    i_small , i_small_gt  = tensor(-32767. , dtype=float32), tensor(  0, dtype=int64)
    i_zero  , i_zero_gt   = tensor(     0. , dtype=float32), tensor(128, dtype=int64)
    i_big   , i_big_gt    = tensor(+31266. , dtype=float32), tensor(255, dtype=int64)
    i_max   , i_max_gt    = tensor(+32767. , dtype=float32), tensor(255, dtype=int64)
    i_toomax, i_toomax_gt = tensor(+32767.1, dtype=float32), tensor(255, dtype=int64)

    assert equal(lin2mlawpcm(i_toomin), i_toomin_gt)
    assert equal(lin2mlawpcm(i_min)   , i_min_gt)
    assert equal(lin2mlawpcm(i_small) , i_small_gt)
    assert equal(lin2mlawpcm(i_zero)  , i_zero_gt)
    assert equal(lin2mlawpcm(i_big)   , i_big_gt)
    assert equal(lin2mlawpcm(i_max)   , i_max_gt)
    assert equal(lin2mlawpcm(i_toomax), i_toomax_gt)


def test_mlaw2lin():
    """Test `mlaw2lin`."""

    i_toomin , i_toomin_gt  = tensor( -0.1     , dtype=float32), tensor(-32768.     , dtype=float32)
    i_min    , i_min_gt     = tensor(  0.      , dtype=float32), tensor(-32768.     , dtype=float32)
    i_zero   , i_zero_gt    = tensor(128.      , dtype=float32), tensor(     0.     , dtype=float32)
    i_max    , i_max_gt     = tensor(255.      , dtype=float32), tensor(+31373.2969 , dtype=float32)
    i_toomax , i_toomax_gt  = tensor(256.      , dtype=float32), tensor(+32767.     , dtype=float32)

    assert equal(mlaw2lin(i_toomin),  i_toomin_gt)
    assert equal(mlaw2lin(i_min)   ,  i_min_gt)
    assert equal(mlaw2lin(i_zero)  ,  i_zero_gt)
    assert equal(mlaw2lin(i_max)   ,  i_max_gt)
    assert equal(mlaw2lin(i_toomax),  i_toomax_gt)


def test_mlaw_conversion():
    """Test linear <-> mulaw."""

    i_min            = tensor(-32768., dtype=float32)
    i_zero           = tensor(     0., dtype=float32)
    # 255] clipping prevent '+32767' int16 max
    i_max,  i_max_gt = tensor(+32767., dtype=float32), tensor(+31373.2969 , dtype=float32)

    assert equal(mlaw2lin(lin2mlaw(i_min)),  i_min)
    assert equal(mlaw2lin(lin2mlaw(i_zero)), i_zero)
    assert equal(mlaw2lin(lin2mlaw(i_max)),  i_max_gt)
