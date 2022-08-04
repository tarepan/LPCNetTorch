"""Test config"""


from .config import load_conf


def test_config_load():
    """Test `load_conf` function."""

    # If validation failed, error will be thrown.
    load_conf()
    assert True
