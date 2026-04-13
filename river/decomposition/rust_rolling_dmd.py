"""Rust-accelerated Rolling DMD and Rolling DMDwC wrappers.

These classes wrap the PyO3 Rust implementations of the fused
Rolling + OnlineDMD pipeline for 50-100x speedup on small matrix
operations where numpy dispatch overhead dominates.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from river.decomposition._rust_decomp import (
    RustRollingDMD as _RustRollingDMD,
    RustRollingDMDwC as _RustRollingDMDwC,
)


def _as_complex(arr: np.ndarray) -> np.ndarray:
    """Ensure numpy array (Rust returns complex arrays directly)."""
    return np.asarray(arr)


def _build_svd_namespace(inner: object) -> SimpleNamespace | None:
    """Build a Python ``_svd``-style namespace exposing ``_U``, ``_S``, ``_Vt``.

    Returns ``None`` when the backing OnlineSVDZhang has never been fit
    (matches Python's ``hasattr(self._svd, "_U")`` being False in that case).
    """
    u = inner.svd_u
    if u is None:
        return None
    return SimpleNamespace(_U=u, _S=inner.svd_s, _Vt=inner.svd_vt)


class RustRollingDMD:
    """Rust-accelerated Rolling Online DMD.

    This is a drop-in replacement for ``Rolling(OnlineDMD(...), window_size)``
    with the rolling window and DMD update/revert fused into a single Rust
    implementation.

    Args:
        r: Number of modes to keep. If 0, all modes are kept.
        w: Weighting factor in (0, 1].
        window_size: Rolling window size.
        initialize: Number of snapshot pairs for batch initialization.
        exponential_weighting: Use exponential weighting in revert.
        eig_rtol: Tolerance for eigenvalue convergence check.
        seed: Random seed.
    """

    def __init__(
        self,
        r: int = 0,
        w: float = 1.0,
        window_size: int = 301,
        initialize: int = 1,
        exponential_weighting: bool = False,
        eig_rtol: float | None = None,
        seed: int | None = None,
    ) -> None:
        self._inner = _RustRollingDMD(
            r, w, window_size, initialize, exponential_weighting, eig_rtol, seed
        )

    def update(
        self,
        x: dict | np.ndarray,
        y: dict | np.ndarray | None = None,
        **kwargs,  # noqa: ANN003, ARG002
    ) -> None:
        """Update with a new sample."""
        self._inner.update(x, y)

    def learn_one(
        self,
        x: dict,
        y: dict | None = None,
        **kwargs,  # noqa: ANN003, ARG002
    ) -> None:
        """Alias for update."""
        self._inner.update(x, y)

    def learn_many(
        self,
        X: np.ndarray | pd.DataFrame,
        Y: np.ndarray | pd.DataFrame | None = None,
    ) -> None:
        """Batch initialization."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.DataFrame):
            Y = Y.values
        self._inner.learn_many(X, Y)

    @property
    def eig(self) -> tuple[np.ndarray, np.ndarray]:
        """Eigenvalues and eigenvectors."""
        evals_raw, evecs_raw = self._inner.eig
        return _as_complex(evals_raw), _as_complex(evecs_raw)

    @property
    def modes(self) -> np.ndarray:
        """DMD modes."""
        return _as_complex(self._inner.modes)

    @property
    def A(self) -> np.ndarray:
        """DMD matrix A."""
        return np.asarray(self._inner.a)

    @property
    def A_allclose(self) -> bool:
        """Check if A has converged."""
        return self._inner.a_allclose

    @property
    def n_seen(self) -> int:
        """Number of samples seen."""
        return self._inner.n_seen

    @property
    def r(self) -> int:
        """Number of modes."""
        return self._inner.r

    @property
    def m(self) -> int:
        """State dimension."""
        return self._inner.m

    @property
    def window_size(self) -> int:
        """Window size."""
        return self._inner.window_size

    @property
    def obj(self) -> RustRollingDMD:
        """DMDChangeDetector compatibility."""
        return self

    @property
    def _svd(self) -> SimpleNamespace | None:
        """Backing OnlineSVDZhang exposing ``_U``, ``_S``, ``_Vt``.

        Mirrors Python ``OnlineDMD._svd``: returns ``None`` until the SVD has
        been fit (which only happens when ``r < m``).
        """
        return _build_svd_namespace(self._inner)

    @property
    def _Y(self) -> np.ndarray:
        """Buffer of past ``y`` snapshots in the rolling window, shape ``(n, m)``.

        Mirrors Python ``OnlineDMD._Y`` post-initialization. Reconstructed from
        the rolling ``(x, y)`` window kept inside the Rust struct.
        """
        return np.asarray(self._inner.y_buffer)

    def transform_one(self, x: dict | np.ndarray) -> dict | np.ndarray:
        r"""Reconstruct x via DMD mode subspace: :math:`\hat{x} = \Phi \Phi^T x`.

        Returns an m-dim real vector in the original feature space. Before the
        model is fitted, returns zeros with the input's shape/keys.
        """
        result = np.asarray(self._inner.transform_one(x))
        if isinstance(x, dict):
            return dict(zip(x.keys(), result))
        return result

    def transform_many(
        self, X: np.ndarray | pd.DataFrame
    ) -> np.ndarray | pd.DataFrame:
        r"""Reconstruct X via DMD mode subspace: :math:`\hat{X} = X \Phi \Phi^T`.

        Returns an ``(n, m)`` real array in the original feature space. Before
        the model is fitted, returns a copy of the input.
        """
        if isinstance(X, pd.DataFrame):
            result = np.asarray(self._inner.transform_many(X.values))
            return pd.DataFrame(result, index=X.index, columns=X.columns)
        return np.asarray(self._inner.transform_many(X))


class RustRollingDMDwC:
    """Rust-accelerated Rolling Online DMD with Control.

    This is a drop-in replacement for ``Rolling(OnlineDMDwC(...), window_size)``
    with the rolling window and DMDwC update/revert fused into a single Rust
    implementation.

    Args:
        p: State truncation. If 0, compute exact DMD.
        q: Control truncation. If 0, compute exact DMD.
        w: Weighting factor in (0, 1].
        window_size: Rolling window size.
        initialize: Number of snapshot pairs for batch initialization.
        exponential_weighting: Use exponential weighting in revert.
        eig_rtol: Tolerance for eigenvalue convergence check.
        seed: Random seed.
    """

    def __init__(
        self,
        p: int = 0,
        q: int = 0,
        w: float = 1.0,
        window_size: int = 301,
        initialize: int = 1,
        exponential_weighting: bool = False,
        eig_rtol: float | None = None,
        seed: int | None = None,
    ) -> None:
        self._inner = _RustRollingDMDwC(
            p, q, w, window_size, initialize, exponential_weighting, eig_rtol, seed
        )

    def update(
        self,
        x: dict | np.ndarray,
        y: dict | np.ndarray | None = None,
        u: dict | np.ndarray | None = None,
        **kwargs,  # noqa: ANN003, ARG002
    ) -> None:
        """Update with a new sample."""
        self._inner.update(x, y, u)

    def learn_one(
        self,
        x: dict,
        y: dict | None = None,
        u: dict | np.ndarray | None = None,
        **kwargs,  # noqa: ANN003, ARG002
    ) -> None:
        """Alias for update."""
        self._inner.update(x, y, u)

    def learn_many(
        self,
        X: np.ndarray | pd.DataFrame,
        Y: np.ndarray | pd.DataFrame | None = None,
        U: np.ndarray | pd.DataFrame | None = None,
    ) -> None:
        """Batch initialization."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.DataFrame):
            Y = Y.values
        if isinstance(U, pd.DataFrame):
            U = U.values
        self._inner.learn_many(X, Y, U)

    @property
    def eig(self) -> tuple[np.ndarray, np.ndarray]:
        """Eigenvalues and eigenvectors."""
        evals_raw, evecs_raw = self._inner.eig
        return _as_complex(evals_raw), _as_complex(evecs_raw)

    @property
    def modes(self) -> np.ndarray:
        """DMD modes."""
        return _as_complex(self._inner.modes)

    @property
    def A(self) -> np.ndarray:
        """DMD matrix A."""
        return np.asarray(self._inner.a)

    @property
    def B(self) -> np.ndarray:
        """Control matrix B."""
        return np.asarray(self._inner.b)

    @property
    def A_allclose(self) -> bool:
        """Check if A has converged."""
        return self._inner.a_allclose

    @property
    def n_seen(self) -> int:
        """Number of samples seen."""
        return self._inner.n_seen

    @property
    def r(self) -> int:
        """Number of modes."""
        return self._inner.r

    @property
    def m(self) -> int:
        """State dimension."""
        return self._inner.m

    @property
    def window_size(self) -> int:
        """Window size."""
        return self._inner.window_size

    @property
    def obj(self) -> RustRollingDMDwC:
        """DMDChangeDetector compatibility."""
        return self

    @property
    def _svd(self) -> SimpleNamespace | None:
        """Backing OnlineSVDZhang exposing ``_U``, ``_S``, ``_Vt``."""
        return _build_svd_namespace(self._inner)

    @property
    def _Y(self) -> np.ndarray:
        """Buffer of past ``y`` snapshots in the rolling window, shape ``(n, m)``."""
        return np.asarray(self._inner.y_buffer)

    def predict_one(
        self,
        x: dict | np.ndarray,
        u: dict | np.ndarray | None = None,
    ) -> dict | np.ndarray:
        """Predict next state."""
        if u is None:
            msg = "Control input u is required for DMDwC prediction"
            raise ValueError(msg)
        result = np.asarray(self._inner.predict_one(x, u))
        if isinstance(x, dict):
            return dict(zip(x.keys(), result))
        return result

    def predict_horizon(
        self,
        x: dict | np.ndarray,
        horizon: int,
        U: np.ndarray | pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Predict multiple future states."""
        if U is None:
            msg = "Control matrix U is required for DMDwC prediction"
            raise ValueError(msg)
        if isinstance(U, pd.DataFrame):
            U = U.values
        return np.asarray(self._inner.predict_horizon(x, horizon, U))

    def transform_one(self, x: dict | np.ndarray) -> dict | np.ndarray:
        r"""Reconstruct x via DMD mode subspace: :math:`\hat{x} = \Phi \Phi^T x`.

        Returns an m-dim real vector in the state space. Before fitting,
        returns zeros with the input's shape/keys.
        """
        result = np.asarray(self._inner.transform_one(x))
        if isinstance(x, dict):
            return dict(zip(x.keys(), result))
        return result

    def transform_many(
        self, X: np.ndarray | pd.DataFrame
    ) -> np.ndarray | pd.DataFrame:
        r"""Reconstruct X via DMD mode subspace: :math:`\hat{X} = X \Phi \Phi^T`.

        Returns an ``(n, m)`` real array in the state space. Before fitting,
        returns a copy of the input.
        """
        if isinstance(X, pd.DataFrame):
            result = np.asarray(self._inner.transform_many(X.values))
            return pd.DataFrame(result, index=X.index, columns=X.columns)
        return np.asarray(self._inner.transform_many(X))
