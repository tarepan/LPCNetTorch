"""Conv receptive field"""

def calc_rf(n_layer: int, kernel_size: int, stride: int = 1) -> int:
    """Calculate receptive field of repetitive-Conv.

    r_0 is receptive field on input.
    r_l is receptive field on the layer (on top of l-layer conv).
    """

    r_l_1, r_l = 1, 1
    for _ in range(n_layer, 0, -1):
        r_l_1 = (r_l - 1) * stride + kernel_size
        r_l = r_l_1

    return r_l_1 # == r_0
