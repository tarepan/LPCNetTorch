"""Test receptive_field calculators."""


from .receptive_field import calc_rf


def test_calc_r():
    """Test `calc_r`."""

    receptive_field = calc_rf(n_layer=2, kernel_size=3, stride=1)
    assert receptive_field == 5

    receptive_field = calc_rf(n_layer=4, kernel_size=3, stride=1)
    assert receptive_field == 9

    receptive_field = calc_rf(n_layer=4, kernel_size=3, stride=2)
    assert receptive_field == 31
