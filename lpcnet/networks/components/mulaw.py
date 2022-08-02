"""MuLaw conversion"""

from torch import Tensor, FloatTensor, LongTensor, tensor, float32, int64, abs, clamp, log, round, sgn # pylint: disable=no-name-in-module,redefined-builtin


# lin/mlaw     - 'Linear' or 'μ-law' scale
# (non)/s16/u8 - Range [-1, 1] | [-32768, +32767] | [0, 255]
# (non)/pcm    - Continuous | Discretized

def lin2mlaw(linear_s16: Tensor) -> FloatTensor:
    """Encode linear sint16-scale signals into μ-law uint8-scale (just scale, not discretized).

    I/O is FloatTensor, so differential.

    Args:
        linear_s16 :: (...) - Linear sint16-scale [-32768., +32767.] signal
    Returns:
        mlaw_u8    :: (...) - μ-law uint8-scale [0., +255.] continuous signal
    """

    scale_s16 = 32768.0
    mu8bit = tensor(255.0, dtype=float32) # pylint: disable=invalid-name

    # Scaling :: [-32768, +32767] -> [-1, +1)
    linear = linear_s16 / scale_s16

    # linear-to-μlaw
    mlaw = sgn(linear) * log(1. + mu8bit * abs(linear))/ log(1. + mu8bit)

    # Rescaling :: [-1, +1) -> [0, +2) -> [0, 255]
    mlaw_u8 = clamp(128 * (mlaw + 1), min=0, max=255)

    return mlaw_u8 # type: ignore


def lin2mlawpcm(linear_s16: Tensor) -> LongTensor:
    """Convert linear int16-scale signals into μ-law 8bit PCM (discrete) signals.

    Signal is discretized, so un-differential.

    Args:
        linear_s16  :: (...) - Linear sint16-scale [-32768., +32767.] signal
    Returns:
        mulaw_u8pcm :: (...) - μ-law uint8-scale [0, +255] PCM (discrete) signal
    """
    mlaw_u8 = lin2mlaw(linear_s16)
    mlaw_u8pcm = round(mlaw_u8).to(int64)
    return mlaw_u8pcm # type: ignore
