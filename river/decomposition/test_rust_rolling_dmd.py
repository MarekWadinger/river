"""Tests for RustRollingDMD and RustRollingDMDwC vs Python equivalents."""

from __future__ import annotations

import pickle

import numpy as np
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


def test_transform_one(
    trained_dmd_pair: tuple[Rolling, RustRollingDMD],
) -> None:
    """transform_one computes x @ modes."""
    _py_dmd, rs_dmd = trained_dmd_pair

    test_x = X_ode[-1]
    rs_result = np.asarray(rs_dmd.transform_one(test_x))
    expected = test_x @ rs_dmd.modes

    assert np.allclose(rs_result, expected, atol=1e-10), (
        f"transform_one mismatch:\nGot: {rs_result}\nExpected (x @ modes): {expected}"
    )


def test_transform_many(
    trained_dmd_pair: tuple[Rolling, RustRollingDMD],
) -> None:
    """transform_many computes X @ modes."""
    _py_dmd, rs_dmd = trained_dmd_pair

    test_X = X_ode[:5]
    rs_result = np.asarray(rs_dmd.transform_many(test_X))
    expected = test_X @ rs_dmd.modes

    assert np.allclose(rs_result, expected, atol=1e-10), (
        f"transform_many mismatch:\nGot: {rs_result}\nExpected (X @ modes): {expected}"
    )


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


def test_n_seen_tracking() -> None:
    """n_seen increments correctly and reflects reverts."""
    ws = 10
    dmd = RustRollingDMD(r=2, initialize=5, w=1.0, window_size=ws)

    for i, (x, y) in enumerate(zip(X_ode[:20], Y_ode[:20])):
        dmd.update(x=x.reshape(1, -1), y=y.reshape(1, -1))
        # After window fills, n_seen stays at window_size (update+revert)
        if i < ws:
            assert dmd.n_seen == i + 1
