"""``download_data`` must actually provision a usable data dir and accept
both ``str`` and ``pathlib.Path`` (the autouse ``initialize_empyrean`` fixture
ensures/skips on data availability)."""

from pathlib import Path

import empyrean


def test_download_data_provisions():
    d = empyrean.download_data()
    assert (Path(d) / "de440.bsp").exists()


def test_download_data_accepts_path_argument():
    # Regression: an explicit pathlib.Path must not raise TypeError at the
    # Option<&str> FFI boundary. Coercion previously ran only on the
    # data_dir=None branch, so download_data(data_dir=Path(...)) raised.
    default = Path(empyrean.default_data_dir())
    result = empyrean.download_data(data_dir=default)  # a real Path, not str
    assert Path(result) == default
    assert (default / "de440.bsp").exists()
