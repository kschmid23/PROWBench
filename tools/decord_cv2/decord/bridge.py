"""decord.bridge: which array type VideoReader.get_batch returns ("native", "numpy" or "torch")."""

_MODES = ("native", "numpy", "torch")
_current = "native"


def set_bridge(new_bridge: str) -> None:
    global _current
    assert new_bridge in _MODES, f"unknown bridge {new_bridge!r}, expected one of {_MODES}"
    _current = new_bridge


def get_bridge() -> str:
    return _current
