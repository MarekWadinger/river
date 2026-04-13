"""Decomposition."""

from __future__ import annotations

from .odmd import OnlineDMD, OnlineDMDwC
from .opca import OnlinePCA
from .osvd import OnlineSVD, OnlineSVDZhang
try:
    from .rust_rolling_dmd import RustRollingDMD, RustRollingDMDwC
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "OnlineSVD",
    "OnlineSVDZhang",
    "OnlineDMD",
    "OnlineDMDwC",
    "OnlinePCA",
    "RustRollingDMD",
    "RustRollingDMDwC",
]
