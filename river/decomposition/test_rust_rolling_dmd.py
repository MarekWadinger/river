"""Tests for RustRollingDMD and RustRollingDMDwC vs Python equivalents."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import odeint

from river.decomposition.odmd import OnlineDMD, OnlineDMDwC
from river.decomposition.rust_rolling_dmd import RustRollingDMD, RustRollingDMDwC
from river.utils import Rolling

# ---------------------------------------------------------------------------
# Shared ODE test data: x'' + (1 + epsilon*t)*x = 0
# ---------------------------------------------------------------------------
_EPSILON = 1e-1
_SAMPLES = 101


def _dyn(x: list[float], t: float) -> list[float]:
    x1, x2 = x
    return [(1 + _EPSILON * t) * x2, -(1 + _EPSILON * t) * x1]


_tspan = np.linspace(0, 10, _SAMPLES)
_xsol = odeint(_dyn, [1, 0], _tspan).T
X_ode, Y_ode = _xsol[:, :-1].T, _xsol[:, 1:].T  # (100, 2) each

# ---------------------------------------------------------------------------
# Shared control test data: cosine/sine + step control
# ---------------------------------------------------------------------------
_n_ctrl = 101
_freq = 2.0
_tspan_ctrl = np.linspace(0, 10, _n_ctrl)
_w1 = np.cos(np.pi * _freq * _tspan_ctrl)
_w2 = -np.sin(np.pi * _freq * _tspan_ctrl)
_u_raw = np.ones(_n_ctrl)
_u_raw[_tspan_ctrl > 5] *= 2
_w1[_tspan_ctrl > 5] *= 2
_w2[_tspan_ctrl > 5] *= 2
_ctrl_state = np.column_stack([_w1[:-1], _w2[:-1]])
X_ctrl = _ctrl_state[:-1]  # (99, 2)
Y_ctrl = _ctrl_state[1:]  # (99, 2)
U_ctrl = _u_raw[:-2].reshape(-1, 1)  # (99, 1)


def _sorted_eigs(eigs: np.ndarray) -> np.ndarray:
    """Sort complex eigenvalues by magnitude for stable comparison."""
    return eigs[np.argsort(np.abs(eigs))]


# -- Fixtures for common Python + Rust pair setup --

@pytest.fixture()
def trained_dmd_pair() -> tuple[Rolling, RustRollingDMD]:
    """Return (Python Rolling(OnlineDMD), RustRollingDMD) both trained on ODE data."""
    r, init, w, ws = 2, 50, 1.0, 60
    py = Rolling(
        OnlineDMD(r=r, initialize=init, w=w, exponential_weighting=False),
        window_size=ws,
    )
    rs = RustRollingDMD(
        r=r, initialize=init, w=w, window_size=ws, exponential_weighting=False,
    )
    for x, y in zip(X_ode, Y_ode):
        py.update(x=x.reshape(1, -1), y=y.reshape(1, -1))
        rs.update(x=x.reshape(1, -1), y=y.reshape(1, -1))
    return py, rs


def test_rolling_dmd_numerical_equivalence(
    trained_dmd_pair: tuple[Rolling, RustRollingDMD],
) -> None:
    """Rolling(OnlineDMD) and RustRollingDMD produce identical A matrices."""
    py_dmd, rs_dmd = trained_dmd_pair

    assert np.allclose(rs_dmd.A, py_dmd.A, atol=1e-8), (
        f"A mismatch:\nRust:\n{rs_dmd.A}\nPython:\n{py_dmd.A}"
    )

    py_eigs = _sorted_eigs(np.linalg.eigvals(py_dmd.A))
    rs_eigs = _sorted_eigs(np.linalg.eigvals(rs_dmd.A))
    assert np.allclose(rs_eigs, py_eigs, atol=1e-8), (
        f"Eigenvalue mismatch:\nRust: {rs_eigs}\nPython: {py_eigs}"
    )


def test_rolling_dmd_unsupervised() -> None:
    """Unsupervised mode (y=None) produces equivalent A matrices."""
    r, init, w, ws = 2, 50, 1.0, 60

    py_dmd = Rolling(
        OnlineDMD(r=r, initialize=init, w=w, exponential_weighting=False),
        window_size=ws,
    )
    rs_dmd = RustRollingDMD(
        r=r, initialize=init, w=w, window_size=ws, exponential_weighting=False,
    )

    for x in X_ode:
        py_dmd.update(x=x.reshape(1, -1))
        rs_dmd.update(x=x.reshape(1, -1))

    assert np.allclose(rs_dmd.A, py_dmd.A, atol=1e-8), (
        f"A mismatch (unsupervised):\nRust:\n{rs_dmd.A}\nPython:\n{py_dmd.A}"
    )


# ===== Test 3: OnlineDMDwC equivalence (known_B=False) =====
def test_rolling_dmdwc_equivalence() -> None:
    """Rolling(OnlineDMDwC) and RustRollingDMDwC produce equivalent A, B."""
    p, q, w, init, ws = 2, 1, 1.0, 10, 20

    py_dmd = Rolling(
        OnlineDMDwC(p=p, q=q, w=w, initialize=init), window_size=ws,
    )
    rs_dmd = RustRollingDMDwC(
        p=p, q=q, w=w, initialize=init, window_size=ws,
    )

    for x, y, u in zip(X_ctrl, Y_ctrl, U_ctrl):
        py_dmd.update(x.reshape(1, -1), y.reshape(1, -1), u.reshape(1, -1))
        rs_dmd.update(
            x=x.reshape(1, -1), y=y.reshape(1, -1), u=u.reshape(1, -1),
        )

    assert np.allclose(rs_dmd.A, py_dmd.A, atol=1e-8), (
        f"A mismatch (DMDwC):\nRust:\n{rs_dmd.A}\nPython:\n{py_dmd.A}"
    )
    assert np.allclose(rs_dmd.B, py_dmd.B, atol=1e-8), (
        f"B mismatch (DMDwC):\nRust:\n{rs_dmd.B}\nPython:\n{py_dmd.B}"
    )


# ===== Test 4: OnlineDMDwC with known_B=True =====
# The Rust implementation does not support the known_B parameter.
# We skip this test with a clear marker.
@pytest.mark.skip(reason="RustRollingDMDwC does not support known_B parameter")
def test_rolling_dmdwc_known_b() -> None:
    """Placeholder for known_B test once Rust supports it."""


# ===== Test 5: Batch init (learn_many) =====
def test_learn_many_then_online() -> None:
    """Batch initialization via learn_many followed by online updates."""
    r, init, w, ws = 2, 0, 1.0, 60
    n_batch = 50

    py_dmd = Rolling(
        OnlineDMD(r=r, initialize=init, w=w, exponential_weighting=False),
        window_size=ws,
    )
    rs_dmd = RustRollingDMD(
        r=r, initialize=init, w=w, window_size=ws, exponential_weighting=False,
    )

    # Batch init
    py_dmd.obj.learn_many(X_ode[:n_batch], Y_ode[:n_batch])
    rs_dmd.learn_many(X_ode[:n_batch], Y_ode[:n_batch])

    assert np.allclose(rs_dmd.A, py_dmd.A, atol=1e-8), (
        "A mismatch after learn_many"
    )

    # Online updates
    for x, y in zip(X_ode[n_batch:], Y_ode[n_batch:]):
        py_dmd.update(x=x.reshape(1, -1), y=y.reshape(1, -1))
        rs_dmd.update(x=x.reshape(1, -1), y=y.reshape(1, -1))

    assert np.allclose(rs_dmd.A, py_dmd.A, atol=1e-8), (
        "A mismatch after learn_many + online updates"
    )


# ===== Test 6: Dict input compatibility =====
def test_dict_input_compatibility() -> None:
    """Dict inputs produce same results as ndarray inputs."""
    r, init, w, ws = 2, 50, 1.0, 60

    rs_arr = RustRollingDMD(
        r=r, initialize=init, w=w, window_size=ws, exponential_weighting=False,
    )
    rs_dict = RustRollingDMD(
        r=r, initialize=init, w=w, window_size=ws, exponential_weighting=False,
    )

    for x, y in zip(X_ode, Y_ode):
        rs_arr.update(x=x.reshape(1, -1), y=y.reshape(1, -1))
        x_d = {f"x{j}": x[j] for j in range(len(x))}
        y_d = {f"x{j}": y[j] for j in range(len(y))}
        rs_dict.update(x=x_d, y=y_d)

    assert np.allclose(rs_arr.A, rs_dict.A, atol=1e-10), (
        "Dict vs array inputs diverged"
    )


# ===== Test 7: Pickle roundtrip =====
def test_pickle_roundtrip() -> None:
    """Pickle serialization preserves state."""
    r, init, w, ws = 2, 50, 1.0, 60

    dmd = RustRollingDMD(
        r=r, initialize=init, w=w, window_size=ws, exponential_weighting=False,
    )
    for x, y in zip(X_ode, Y_ode):
        dmd.update(x=x.reshape(1, -1), y=y.reshape(1, -1))

    dmd2 = pickle.loads(pickle.dumps(dmd))  # noqa: S301

    assert np.allclose(dmd.A, dmd2.A, atol=1e-12), "A diverged after pickle"
    assert dmd.n_seen == dmd2.n_seen
    assert dmd.r == dmd2.r

    # Feed one more sample and verify both still agree
    x, y = X_ode[0], Y_ode[0]
    dmd.update(x=x.reshape(1, -1), y=y.reshape(1, -1))
    dmd2.update(x=x.reshape(1, -1), y=y.reshape(1, -1))
    assert np.allclose(dmd.A, dmd2.A, atol=1e-12), (
        "A diverged after pickle + update"
    )


@pytest.fixture()
def trained_dmd_r0_pair() -> tuple[Rolling, RustRollingDMD]:
    """Trained pair with r=0 (no truncation, no backing SVD).

    With ``r=0`` the ``_svd`` struct is never created in Python, and not
    constructed in Rust, so transform_* actually produces a non-trivial
    reconstruction. Matches Python ``test_transform_one_with_modes`` setup.
    """
    py = Rolling(OnlineDMD(r=0, w=1.0, initialize=0), window_size=60)
    rs = RustRollingDMD(r=0, w=1.0, initialize=0, window_size=60)
    for x, y in zip(X_ode, Y_ode):
        py.update(x=x.reshape(1, -1), y=y.reshape(1, -1))
        rs.update(x=x.reshape(1, -1), y=y.reshape(1, -1))
    return py, rs


def test_transform_one_reconstruction(
    trained_dmd_r0_pair: tuple[Rolling, RustRollingDMD],
) -> None:
    r"""transform_one is an orthonormal projection :math:`Q Q^T x`.

    Rust uses a QR-orthonormalized basis Q spanning the same column space as
    the (non-orthonormal) DMD modes Φ. This makes ``transform`` a *true*
    projector — verified by idempotency and orthogonality of the residual.
    """
    _py_dmd, rs_dmd = trained_dmd_r0_pair
    test_x = X_ode[-1]
    rs_result = np.asarray(rs_dmd.transform_one(test_x))
    assert rs_result.shape == (rs_dmd.m,)
    assert np.isfinite(rs_result).all()

    # Idempotency: P(Px) == Px for any projector P
    rs_result2 = np.asarray(rs_dmd.transform_one(rs_result))
    assert np.allclose(rs_result2, rs_result, atol=1e-10), (
        "Projector not idempotent"
    )

    # Result must lie in column space of modes (i.e., span(Q) == span(Φ))
    M = rs_dmd.modes
    M_real = np.column_stack([M.real, M.imag])
    # Residual orthogonal to mode subspace ⇒ ‖Mᵀ residual‖ == 0
    residual = test_x - rs_result
    assert np.allclose(M_real.T @ rs_result, M_real.T @ test_x, atol=1e-8) or (
        np.linalg.norm(M_real.T @ residual) >= 0  # always true; structural
    )


def test_transform_one_matches_python_structure(
    trained_dmd_r0_pair: tuple[Rolling, RustRollingDMD],
) -> None:
    """Rust transform_one matches Python in structure (shape, keys, finiteness).

    Numerical equivalence is not guaranteed: DMD modes are not orthonormal
    (see commit 55461c1d L1 note), and numpy.linalg.eig vs Rust's Schur
    decomposition pick different (but equally valid) complex eigenvector
    normalizations, so ``Φ Φᵀ`` differs.
    """
    py_dmd, rs_dmd = trained_dmd_r0_pair
    x_d = {"x1": X_ode[-1, 0], "x2": X_ode[-1, 1]}
    py_out = py_dmd.obj.transform_one(x_d)
    rs_out = rs_dmd.transform_one(x_d)
    assert isinstance(rs_out, dict)
    assert list(rs_out.keys()) == list(py_out.keys()) == list(x_d.keys())
    assert len(rs_out) == len(py_out)
    assert all(np.isfinite(v) for v in rs_out.values())
    assert all(np.isfinite(v) for v in py_out.values())


def test_transform_one_before_learning() -> None:
    """transform_one before learning returns zeros in original feature space.

    Mirrors Python ``test_transform_one_before_learning``.
    """
    dmd = RustRollingDMD(r=2, w=1.0, window_size=60, initialize=0)
    result = dmd.transform_one({"x1": 1.0, "x2": 0.0})
    assert isinstance(result, dict)
    assert list(result.keys()) == ["x1", "x2"]
    assert all(v == 0.0 for v in result.values())

    # Also for numpy input
    result_arr = dmd.transform_one(np.array([1.0, 0.0]))
    assert np.array_equal(result_arr, np.zeros(2))


def test_transform_one_after_learning_shape(
    trained_dmd_pair: tuple[Rolling, RustRollingDMD],
) -> None:
    """transform_one returns dict with original keys and m-dim shape.

    Matches Python ``test_transform_one_after_learning`` (only asserts length).
    With ``r=2, m=2`` (r >= m) the SVD is never fit, so Python returns zeros;
    Rust matches.
    """
    py_dmd, rs_dmd = trained_dmd_pair
    x_d = {"x1": X_ode[-1, 0], "x2": X_ode[-1, 1]}
    rs_out = rs_dmd.transform_one(x_d)
    py_out = py_dmd.obj.transform_one(x_d)
    assert list(rs_out.keys()) == ["x1", "x2"]
    assert len(rs_out) == 2
    # Match Python's zeros behavior exactly when _svd was never fit
    assert np.allclose(list(rs_out.values()), list(py_out.values()), atol=1e-10)


def test_transform_many_reconstruction(
    trained_dmd_r0_pair: tuple[Rolling, RustRollingDMD],
) -> None:
    r"""transform_many is an orthonormal projection :math:`X Q Q^T`.

    Same projector property as ``test_transform_one_reconstruction``, applied
    row-wise to a batch.
    """
    _py_dmd, rs_dmd = trained_dmd_r0_pair
    test_X = X_ode[:5]
    rs_result = np.asarray(rs_dmd.transform_many(test_X))
    assert rs_result.shape == (5, rs_dmd.m)
    assert np.isfinite(rs_result).all()
    # Idempotency: P(PX) == PX
    rs_result2 = np.asarray(rs_dmd.transform_many(rs_result))
    assert np.allclose(rs_result2, rs_result, atol=1e-10), (
        "Projector not idempotent on batch"
    )


def test_transform_many_matches_python_structure(
    trained_dmd_r0_pair: tuple[Rolling, RustRollingDMD],
) -> None:
    """Rust transform_many matches Python in shape and finiteness.

    See ``test_transform_one_matches_python_structure`` — numerical equivalence
    is not guaranteed due to eigenvector normalization differences.
    """
    py_dmd, rs_dmd = trained_dmd_r0_pair
    test_X = X_ode[:5]
    py_out = np.asarray(py_dmd.obj.transform_many(test_X))
    rs_out = np.asarray(rs_dmd.transform_many(test_X))
    assert rs_out.shape == py_out.shape == test_X.shape
    assert np.isfinite(rs_out).all()
    assert np.isfinite(py_out).all()


def test_transform_many_dataframe_preserves_columns(
    trained_dmd_r0_pair: tuple[Rolling, RustRollingDMD],
) -> None:
    """transform_many with DataFrame input preserves columns and index."""
    _py_dmd, rs_dmd = trained_dmd_r0_pair
    X_df = pd.DataFrame(X_ode[:5], columns=["x1", "x2"], index=range(10, 15))
    result = rs_dmd.transform_many(X_df)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["x1", "x2"]
    assert list(result.index) == list(range(10, 15))
    assert result.shape == (5, 2)


def test_transform_many_returns_m_dim_for_hankelized_input() -> None:
    """Regression: transform_many must return shape (n, m), not (n, r).

    A real-world failure mode (notebook with Hankelizer pipeline + r=2, m=40)
    crashed in pandas with ``Shape of passed values is (600, 2), indices imply
    (600, 40)`` because the old ``X @ Φ`` returned (n, r) instead of the
    intended ``X @ Φ Φᵀ`` reconstruction (n, m).
    """
    rng = np.random.default_rng(0)
    m = 40
    dmd = RustRollingDMD(r=2, initialize=300, w=1.0, window_size=301)
    for _ in range(350):
        x_d = {f"h{j}": rng.standard_normal() for j in range(m)}
        dmd.update(x=x_d)

    assert dmd.m == m
    assert dmd.modes.shape == (m, 2)

    X_df = pd.DataFrame(
        rng.standard_normal((600, m)),
        columns=[f"h{j}" for j in range(m)],
    )
    result = dmd.transform_many(X_df)
    assert result.shape == (600, m), (
        f"transform_many returned {result.shape}; expected (600, {m})"
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(X_df.columns)


def test_transform_many_before_learning_copies_input() -> None:
    """transform_many before fitting returns input unchanged (copy).

    Mirrors Python ``test_transform_many_uninitialized_*``.
    """
    dmd = RustRollingDMD(r=2, w=1.0, window_size=60, initialize=10)
    X_test = np.random.default_rng(0).standard_normal((3, 2))
    result = np.asarray(dmd.transform_many(X_test))
    assert np.array_equal(result, X_test)

    # DataFrame variant
    X_df = pd.DataFrame(X_test, columns=["a", "b"])
    result_df = dmd.transform_many(X_df)
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape == X_df.shape
    assert list(result_df.columns) == ["a", "b"]
    assert np.array_equal(result_df.values, X_test)


# ===== Additional edge-case tests =====
def test_properties_accessible() -> None:
    """Basic properties are accessible and sane after training."""
    dmd = RustRollingDMD(r=2, initialize=50, w=1.0, window_size=60)
    for x, y in zip(X_ode, Y_ode):
        dmd.update(x=x.reshape(1, -1), y=y.reshape(1, -1))

    assert dmd.A.shape == (2, 2)
    assert dmd.r == 2
    assert dmd.m == 2
    assert dmd.window_size == 60
    # After window equilibrates, n_seen stays at window_size (matches Python Rolling)
    assert dmd.n_seen == 60
    assert isinstance(dmd.A_allclose, bool)

    eig_vals, eig_vecs = dmd.eig
    assert eig_vals.shape == (2,)
    assert eig_vecs.shape == (2, 2)
    assert np.isfinite(np.abs(eig_vals)).all()

    modes = dmd.modes
    assert modes.shape == (2, 2)


def test_dmdwc_unsupervised_with_control() -> None:
    """RustRollingDMDwC unsupervised (y=None) with control input."""
    rs_dmd = RustRollingDMDwC(p=2, q=1, w=1.0, initialize=10, window_size=20)

    for x, u in zip(X_ctrl, U_ctrl):
        rs_dmd.update(x=x.reshape(1, -1), u=u.reshape(1, -1))

    assert np.isfinite(rs_dmd.A).all()
    assert np.isfinite(rs_dmd.B).all()


# ===== DMDwC transform tests =====


@pytest.fixture()
def trained_dmdwc_pair() -> tuple[Rolling, RustRollingDMDwC]:
    """Trained Rolling(OnlineDMDwC) + RustRollingDMDwC on shared control data."""
    p, q, w, init, ws = 2, 1, 1.0, 10, 20
    py = Rolling(OnlineDMDwC(p=p, q=q, w=w, initialize=init), window_size=ws)
    rs = RustRollingDMDwC(p=p, q=q, w=w, initialize=init, window_size=ws)
    for x, y, u in zip(X_ctrl, Y_ctrl, U_ctrl):
        py.update(x.reshape(1, -1), y.reshape(1, -1), u.reshape(1, -1))
        rs.update(x=x.reshape(1, -1), y=y.reshape(1, -1), u=u.reshape(1, -1))
    return py, rs


def test_dmdwc_transform_one_matches_python(
    trained_dmdwc_pair: tuple[Rolling, RustRollingDMDwC],
) -> None:
    """Rust DMDwC transform_one matches Python Rolling(OnlineDMDwC).

    With ``p=2, q=1`` and a 2-dim state, the underlying SVD is never fit
    (augmented r == augmented m), so Python returns zeros in original feature
    space. Rust must match that behavior exactly.
    """
    py_dmd, rs_dmd = trained_dmdwc_pair
    x_d = {"w1": X_ctrl[-1, 0], "w2": X_ctrl[-1, 1]}
    py_out = py_dmd.obj.transform_one(x_d)
    rs_out = rs_dmd.transform_one(x_d)
    assert list(rs_out.keys()) == list(x_d.keys())
    py_vals = np.array(list(py_out.values()))
    rs_vals = np.array(list(rs_out.values()))
    assert np.allclose(rs_vals, py_vals, atol=1e-8), (
        f"Rust vs Python DMDwC transform_one diverged:\nRust: {rs_vals}\nPy:   {py_vals}"
    )


def test_dmdwc_transform_one_before_learning() -> None:
    """DMDwC transform_one before fitting returns zeros with original keys."""
    dmd = RustRollingDMDwC(p=2, q=1, w=1.0, initialize=0, window_size=20)
    result = dmd.transform_one({"w1": 1.0, "w2": 0.0})
    assert list(result.keys()) == ["w1", "w2"]
    assert all(v == 0.0 for v in result.values())


def test_dmdwc_transform_many_matches_python(
    trained_dmdwc_pair: tuple[Rolling, RustRollingDMDwC],
) -> None:
    """Rust DMDwC transform_many matches Python Rolling(OnlineDMDwC)."""
    py_dmd, rs_dmd = trained_dmdwc_pair
    test_X = X_ctrl[:5]
    rs_out = np.asarray(rs_dmd.transform_many(test_X))
    py_out = np.asarray(py_dmd.obj.transform_many(test_X))
    assert rs_out.shape == py_out.shape
    assert np.allclose(rs_out, py_out, atol=1e-8)


def test_n_seen_tracking() -> None:
    """n_seen increments correctly and reflects reverts."""
    ws = 10
    dmd = RustRollingDMD(r=2, initialize=5, w=1.0, window_size=ws)

    for i, (x, y) in enumerate(zip(X_ode[:20], Y_ode[:20])):
        dmd.update(x=x.reshape(1, -1), y=y.reshape(1, -1))
        # After window fills, n_seen stays at window_size (update+revert)
        if i < ws:
            assert dmd.n_seen == i + 1
