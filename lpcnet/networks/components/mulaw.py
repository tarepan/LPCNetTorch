"""MuLaw conversion"""

from torch import Tensor, tensor, float32, abs, clamp, log, sgn # pylint: disable=no-name-in-module,redefined-builtin


def lin2mlaw(linear_int: Tensor) -> Tensor:
    """Convert linear int16 signals into μ-law int8 scale (just scale, not discretized).

    I/O is FloatTensor, so differential.

    Args:
        linear_int :: (...) - Linear PCM int16-scale [-32768, +32767] signal
    Returns:
        mulaw_int  :: (...) - μ-law int8-scale [-256, +255] signal
    """

    scale = 32768.0
    mu = tensor(255.0, dtype=float32) # pylint: disable=invalid-name

    # Scaling :: [-32768, +32767] -> [-1, +1)
    linear = linear_int / scale

    # linear-to-μlaw
    ulaw = sgn(linear) * log(1. + mu * abs(linear))/ log(1. + mu)

    # Rescaling :: [-1, +1) -> [0, +2) -> [0, 255]
    ulaw_int = clamp(128 * (ulaw + 1), min=0, max=255)

    return ulaw_int
