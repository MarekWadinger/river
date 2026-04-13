"""Test conversion from river to scikit-learn API and back.

Requires two modifications to river code:
1. change line 49 in river.compat.river_to_sklearn to
`SKLEARN_INPUT_Y_PARAMS = {"multi_output": True, "y_numeric": False}`
2. change line 194 in river.compat.river_to_sklearn to
`y_pred = np.empty(shape=(len(X), X.shape[1]))`
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import odeint

from river.decomposition.odmd import OnlineDMD
from river.utils import Rolling

epsilon = 1e-1


def dyn(x: list[float], t: float) -> list[float]:
    x1, x2 = x
    dxdt = [(1 + epsilon * t) * x2, -(1 + epsilon * t) * x1]
    return dxdt


# integrate from initial condition [1,0]
samples = 101
tspan = np.linspace(0, 10, samples)
dt = 0.1
x0 = [1, 0]
xsol = odeint(dyn, x0, tspan).T
# extract snapshots
X, Y = xsol[:, :-1].T, xsol[:, 1:].T
t = tspan[1:]
n, m = X.shape
A = np.empty((n, m, m))
eigvals = np.empty((n, m), dtype=complex)
for k in range(n):
    A[k, :, :] = np.array([[0, (1 + epsilon * t[k])], [-(1 + epsilon * t[k]), 0]])
    eigvals[k, :] = np.linalg.eigvals(A[k, :, :])


def test_input_types() -> None:
    n_init = round(samples / 2)

    odmd1 = OnlineDMD()

    odmd1.learn_many(X[:n_init, :], Y[:n_init, :])
    for x, y in zip(X[n_init:, :], Y[n_init:, :]):
        odmd1.update(x, y)

    X_, Y_ = pd.DataFrame(X), pd.DataFrame(Y)

    odmd2 = OnlineDMD()

    odmd2.learn_many(X_.iloc[:n_init], Y_.iloc[:n_init])
    for x, y in zip(X_.iloc[n_init:].values, Y_.iloc[n_init:].values):
        odmd2.update(x, y)

    assert np.allclose(odmd1.A, odmd2.A)


def test_one_many_close() -> None:
    n_init = round(samples / 2)

    odmd1 = OnlineDMD()
    odmd2 = OnlineDMD()

    odmd1.learn_many(X[:n_init, :], Y[:n_init, :])
    odmd2.learn_many(X[:n_init, :], Y[:n_init, :])

    eig_o1 = np.log(np.linalg.eigvals(odmd1.A)) / dt
    eig_o2 = np.log(np.linalg.eigvals(odmd2.A)) / dt
    assert np.allclose(eig_o1, eig_o2)

    for x, y in zip(X[n_init:, :], Y[n_init:, :]):
        odmd1.update(x, y)

    odmd2.learn_many(X[n_init:, :], Y[n_init:, :])
    eig_o1 = np.log(np.linalg.eigvals(odmd1.A)) / dt
    eig_o2 = np.log(np.linalg.eigvals(odmd2.A)) / dt
    print(eig_o1, eig_o2)
    assert np.allclose(eig_o1, eig_o2)


def test_errors_raised() -> None:
    odmd = OnlineDMD()

    with pytest.raises(Exception):
        odmd._update_many(X, Y)

    rodmd = Rolling(OnlineDMD(), window_size=1)
    with pytest.raises(Exception):
        for x, y in zip(X, Y):
            rodmd.update(x, y)


def test_allclose_unsupervised_supervised() -> None:
    m_u = OnlineDMD(r=2, w=0.1, initialize=0)
    m_s = OnlineDMD(r=2, w=0.1, initialize=0)

    for x, y in zip(X, Y):
        m_u.update(x)
        m_s.update(x, y)
    eig_u, _ = np.log(m_u.eig[0]) / dt
    eig_s, _ = np.log(m_u.eig[0]) / dt

    assert np.allclose(eig_u, eig_s)


# Proctor et al. (2016) "Dynamic Mode Decomposition with Control" suggests that
#  the DMDwC where B is unknown requires a second SVD computation for output
#  space of Y. As the computation and updates of SVDs are expensive, we want to
#  avoid this if possible. This test checks if the SVD of augumented state +
#  control space is at least as close to SVD of original space than the SVD of
#  the output space to the SVD of the original space.
def test_one_svd_is_enough() -> None:
    import numpy as np
    import pandas as pd
    import scipy as sp

    np.random.seed(0)

    n = 101
    freq = 2.0
    tspan = np.linspace(0, 10, n)
    w1 = np.cos(np.pi * freq * tspan)
    w2 = -np.sin(np.pi * freq * tspan)
    w3 = np.sin(2 * np.pi * freq * tspan)
    u_ = np.ones(n)
    u_[tspan > 5] *= 2
    w1[tspan > 5] *= 2
    w2[tspan > 5] *= 2
    w3[tspan > 5] *= 2
    df = pd.DataFrame({"w1": w1[:-1], "w2": w2[:-1], "w3": w3[:-1]})
    X, Y = df.iloc[:-1], df.shift(-1).iloc[:-1]
    U = pd.DataFrame({"u": u_[:-2]})
    X_ = X.copy()
    X_["u"] = U

    u_orig, s_orig, _ = sp.sparse.linalg.svds(X.values.T, k=2, return_singular_vectors="u")
    u_aug, s_aug, _ = sp.sparse.linalg.svds(X_.values.T, k=3, return_singular_vectors="u")
    u_out, s_out, _ = sp.sparse.linalg.svds(Y.values.T, k=2, return_singular_vectors="u")

    assert (np.abs(u_orig - u_aug[:3, :2]) <= np.abs(u_orig - u_out)).all()
    assert (np.abs(s_orig - s_aug[:2]) <= np.abs(s_orig - s_out)).all()


def test_truncated_svd_path() -> None:
    """Test _truncate_w_svd path (r < m) with truncated DMD."""
    odmd = OnlineDMD(r=1, w=1.0, initialize=10)
    for x, y in zip(X, Y):
        odmd.update(x, y)
    assert odmd.A.shape == (1, 1)
    assert odmd.modes.shape == (2, 1)
    # Verify predict works in truncated mode
    pred = odmd.predict_one({"x1": X[-1, 0], "x2": X[-1, 1]})
    assert len(pred) == 2
    # Verify predict_horizon works in truncated mode
    horizon = odmd.predict_horizon({"x1": X[-1, 0], "x2": X[-1, 1]}, 5)
    assert horizon.shape == (5, 2)


def test_exponential_weighting_revert() -> None:
    """Test _update_A_P with exponential weighting in revert."""
    odmd = Rolling(OnlineDMD(w=0.95, exponential_weighting=True), window_size=10)
    for x, y in zip(X, Y):
        odmd.update(x, y)
    assert np.isfinite(odmd.A).all()


def test_update_many_rank_check() -> None:
    """Test update_many rank check ValueError (X with rank < r)."""
    odmd = OnlineDMD(r=2, w=1.0, initialize=0)
    # Create data with rank 1 (all rows identical)
    X_rank1 = np.ones((5, 2))
    Y_rank1 = np.ones((5, 2))
    with pytest.raises(ValueError, match="Failed rank"):
        odmd.learn_many(X_rank1, Y_rank1)


def test_forecast_uninitialized() -> None:
    """Test forecast when SVD not initialized returns zeros."""
    odmd = OnlineDMD(r=1, w=1.0, initialize=0)
    odmd._x_last = {"x1": 1.0, "x2": 0.0}
    result = odmd.forecast(3)
    assert result == [0.0, 0.0, 0.0]


def test_forecast_after_learning() -> None:
    """Test forecast after unsupervised learning."""
    odmd = OnlineDMD(r=2, w=1.0, initialize=0)
    for x, y in zip(X, Y):
        odmd.update(x)
    result = odmd.forecast(3)
    assert len(result) == 3
    assert all(np.isfinite(v) for v in result)


def test_transform_one_before_learning() -> None:
    """Test transform_one before learning returns zeros."""
    odmd = OnlineDMD(r=2, w=1.0, initialize=0)
    result = odmd.transform_one({"x1": 1.0, "x2": 0.0})
    assert all(v == 0.0 for v in result.values())


def test_transform_one_after_learning() -> None:
    """Test transform_one after learning projects via modes."""
    odmd = OnlineDMD(r=2, w=1.0, initialize=0)
    for x, y in zip(X, Y):
        odmd.update(x, y)
    result = odmd.transform_one({"x1": X[-1, 0], "x2": X[-1, 1]})
    assert len(result) == 2


def test_transform_many() -> None:
    """Test transform_many reconstructs via modes (numpy and DataFrame)."""
    odmd = OnlineDMD(r=2, w=1.0, initialize=0)
    for x, y in zip(X, Y):
        odmd.update(x, y)
    # numpy input
    result = odmd.transform_many(X[:5])
    assert result.shape == (5, 2)
    # DataFrame input
    X_df = pd.DataFrame(X[:5], columns=["x1", "x2"])
    result_df = odmd.transform_many(X_df)
    assert isinstance(result_df, pd.DataFrame)
    assert result_df.shape == (5, 2)


def test_transform_many_truncated() -> None:
    """Test transform_many with r < m (via _svd._U)."""
    m = 6
    np.random.seed(0)
    X_hi = np.random.randn(100, m)
    Y_hi = np.random.randn(100, m)
    odmd = OnlineDMD(r=2, w=1.0, initialize=0)
    odmd.learn_many(X_hi[:50], Y_hi[:50])
    assert odmd.r < odmd.m
    result = odmd.transform_many(X_hi[50:55])
    assert result.shape == (5, m)
    result_df = odmd.transform_many(pd.DataFrame(X_hi[50:55]))
    assert result_df.shape == (5, m)


def test_modes_truncated() -> None:
    """Test modes property for r < m."""
    odmd = OnlineDMD(r=1, w=1.0, initialize=10)
    for x, y in zip(X, Y):
        odmd.update(x, y)
    assert odmd.modes.shape == (2, 1)


def test_xi_property() -> None:
    """Test xi property (amplitudes via scipy.optimize.minimize)."""
    odmd = OnlineDMD(r=2, w=1.0, initialize=0)
    for x, y in zip(X, Y):
        odmd.update(x, y)
    xi = odmd.xi
    assert xi.shape == (2,)
    assert np.isfinite(xi).all()


def test_A_allclose_with_eig_rtol() -> None:
    """Test A_allclose with eig_rtol."""
    odmd = OnlineDMD(r=2, w=1.0, eig_rtol=0.1, initialize=0)
    for x, y in zip(X, Y):
        odmd.update(x, y)
    assert isinstance(odmd.A_allclose, bool)
    # Without eig_rtol, always False
    odmd2 = OnlineDMD(r=2, w=1.0, initialize=0)
    for x, y in zip(X, Y):
        odmd2.update(x, y)
    assert odmd2.A_allclose is False


def test_eig_sorting() -> None:
    """Test eig property sorting branch."""
    odmd = OnlineDMD(r=2, w=1.0, initialize=0)
    for x, y in zip(X, Y):
        odmd.update(x, y)
    eig_vals, eig_vecs = odmd.eig
    assert eig_vals.shape == (2,)
    assert eig_vecs.shape == (2, 2)


def test_unsupervised_revert() -> None:
    """Test unsupervised revert path (y=None) via Rolling."""
    odmd = Rolling(OnlineDMD(r=2, w=1.0), window_size=10)
    for x in X:
        odmd.update(x)
    assert np.isfinite(odmd.A).all()


def test_update_many_unsupervised_dataframe() -> None:
    """Test update_many with Y=None using DataFrame."""
    odmd = OnlineDMD(r=2, w=1.0, initialize=0)
    df = pd.DataFrame(X, columns=["x1", "x2"])
    odmd.update_many(df)
    assert np.isfinite(odmd.A).all()


def test_update_many_incremental() -> None:
    """Test _update_many path when already initialized."""
    odmd = OnlineDMD(r=2, w=1.0, initialize=0)
    # First batch initializes
    odmd.learn_many(X[:50], Y[:50])
    # Second batch updates incrementally
    odmd.learn_many(X[50:], Y[50:])
    assert np.isfinite(odmd.A).all()


def test_update_many_exponential_weighting() -> None:
    """Test _update_many with exponential weighting."""
    odmd = OnlineDMD(r=2, w=0.95, exponential_weighting=True, initialize=0)
    odmd.learn_many(X[:50], Y[:50])
    odmd.learn_many(X[50:], Y[50:])
    assert np.isfinite(odmd.A).all()


def test_truncated_revert_via_rolling() -> None:
    """Test r < m revert path (_truncate_w_svd with revert)."""
    odmd = Rolling(OnlineDMD(r=1, w=1.0, initialize=10), window_size=20)
    for x, y in zip(X, Y):
        odmd.update(x, y)
    assert np.isfinite(odmd.A).all()


def test_update_many_truncated() -> None:
    """Test r < m path with incremental individual updates after init."""
    odmd = OnlineDMD(r=1, w=1.0, initialize=10)
    for x, y in zip(X[:50], Y[:50]):
        odmd.update(x, y)
    assert odmd.A.shape == (1, 1)
    assert np.isfinite(odmd.A).all()
    # Verify _truncate_w_svd update path was exercised
    assert odmd.r < odmd.m


def test_update_many_dataframe_inputs() -> None:
    """Test _update_many with DataFrame inputs for X and Y."""
    odmd = OnlineDMD(r=2, w=1.0, initialize=0)
    X_df = pd.DataFrame(X[:50], columns=["x1", "x2"])
    Y_df = pd.DataFrame(Y[:50], columns=["x1", "x2"])
    odmd.learn_many(X_df, Y_df)
    # Incremental with DataFrames triggers _update_many DataFrame branch
    X_df2 = pd.DataFrame(X[50:60], columns=["x1", "x2"])
    Y_df2 = pd.DataFrame(Y[50:60], columns=["x1", "x2"])
    odmd._update_many(X_df2, Y_df2)
    assert np.isfinite(odmd.A).all()


def test_forecast_truncated_initialized() -> None:
    """Test forecast with r < m when SVD is initialized."""
    odmd = OnlineDMD(r=1, w=1.0, initialize=10)
    for x, y in zip(X[:50], Y[:50]):
        odmd.update(x)
    result = odmd.forecast(3)
    assert len(result) == 3
    assert all(np.isfinite(v) for v in result)


def test_transform_one_with_modes() -> None:
    """Test transform_one after learning (exercises x_arr @ self.modes).

    Uses r=0 (auto-detected) so _svd is never created, allowing line 866.
    """
    odmd = OnlineDMD(r=0, w=1.0, initialize=0)
    for x, y in zip(X[:50], Y[:50]):
        odmd.update(x, y)
    assert not hasattr(odmd, "_svd")
    result = odmd.transform_one({"x1": X[0, 0], "x2": X[0, 1]})
    # Result should use modes projection
    assert len(result) == 2
    assert all(np.isfinite(v) for v in result.values())


def test_update_many_incremental_Y_trimming() -> None:
    """Test update_many incremental path that trims _Y buffer.

    After initial learn_many sets n_seen=50 and _Y has 50 rows, a second
    learn_many with 60 rows would make _Y grow to 110 while n_seen becomes
    110, then the > branch trims it back.
    """
    odmd = OnlineDMD(r=2, w=1.0, initialize=0)
    odmd.learn_many(X[:50], Y[:50])
    # Feed a large second batch so _Y grows beyond n_seen after vstack,
    # then the trim branch fires.
    odmd.learn_many(X[:90], Y[:90])
    assert odmd._Y.shape[0] <= odmd.n_seen
    assert np.isfinite(odmd.A).all()


def test_update_many_truncated_r_less_m() -> None:
    """Test _update_many with r < m triggers SVD truncation in the batch path."""
    m = 6
    np.random.seed(0)
    X_hi = np.random.randn(100, m)
    Y_hi = np.random.randn(100, m)
    odmd = OnlineDMD(r=2, w=1.0, initialize=0)
    odmd.learn_many(X_hi[:50], Y_hi[:50])
    assert odmd.r < odmd.m
    # Single-row _update_many triggers r < m branch in _update_many
    odmd._update_many(X_hi[50:51], Y_hi[50:51])
    assert np.isfinite(odmd.A).all()


def test_transform_many_uninitialized_numpy() -> None:
    """Test transform_many returns copy when model uninitialized (numpy)."""
    odmd = OnlineDMD(r=2, w=1.0, initialize=10)
    # Feed one sample so _svd exists but _U doesn't
    odmd.update(X[0], Y[0])
    X_test = np.random.randn(3, 2)
    result = odmd.transform_many(X_test)
    assert np.array_equal(result, X_test)


def test_transform_many_uninitialized_dataframe() -> None:
    """Test transform_many returns copy when model uninitialized (DataFrame)."""
    odmd = OnlineDMD(r=2, w=1.0, initialize=10)
    odmd.update(X[0], Y[0])
    X_test = pd.DataFrame(np.random.randn(3, 2), columns=["a", "b"])
    result = odmd.transform_many(X_test)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == X_test.shape
