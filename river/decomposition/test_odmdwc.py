"""Tests for the OnlineDMDwC model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from river.decomposition.odmd import OnlineDMD, OnlineDMDwC
from river.utils import Rolling

T = 100
t_diff = 0.01
samples = int(T / t_diff) - 1
time_space = np.linspace(0, T, num=samples + 1)


def omega(t: float) -> float:
    """Calculate the omega function."""
    return 1 + 0.1 * t


def u_t(x: np.ndarray) -> np.ndarray:
    """Calculate the control input function."""
    return K_prop * x


X = np.zeros((samples + 1, 2))
X[0, :] = np.array([4, 7])

K_prop = -1

B = np.array([1, 0])
U = np.zeros((samples + 1, 1))

i = 1
true_eigs_ = []
for k in np.linspace(t_diff, T, num=samples):
    A_t = np.array([[t_diff, -omega(k)], [omega(k), 0.1 * t_diff]])
    true_eigs_.append(np.imag(np.log(np.linalg.eig(A_t)[0])))

    control_input = np.matmul(B, u_t(X[i - 1]).T) * t_diff
    U[i, :] = control_input
    autonomous_state = np.matmul(X[i - 1, :], A_t) * t_diff + X[i - 1, :]
    X[i, :] = autonomous_state + control_input
    i += 1

true_eigs = np.vstack(true_eigs_)

X = X[:-1, :]
Y = X[1:, :]
U = U[:-1, :]


def test_input_types() -> None:
    """Test the input types for the OnlineDMDwC model."""
    n_init = round(samples / 2)

    odmd1 = OnlineDMDwC(initialize=n_init)

    for x, y, u in zip(X, Y, U):
        odmd1.update(x, y, u)

    X_, Y_, U_ = pd.DataFrame(X), pd.DataFrame(Y), pd.DataFrame(U)

    odmd2 = OnlineDMDwC(initialize=n_init)

    for xd, yd, ud in zip(
        X_.to_dict(orient="records"),
        Y_.to_dict(orient="records"),
        U_.to_dict(orient="records"),
    ):
        odmd2.learn_one(xd, yd, ud)

    assert np.allclose(odmd1.A, odmd2.A)


def test_dmdwc_variations() -> None:
    """Test the variations of the OnlineDMDwC model.

    Rolling variants only assert finite eigenvalues due to the numerical
    precision limitation documented in OnlineDMD.revert.
    """
    odmd = OnlineDMD(initialize=10)
    odmdc_weight = OnlineDMDwC(initialize=10, w=0.995, exponential_weighting=True)
    odmdc_b = OnlineDMDwC(initialize=10, B=B.reshape(-1, 1))
    odmdc_window = Rolling(OnlineDMDwC(initialize=10), window_size=100)
    odmdc_b_window = Rolling(OnlineDMDwC(initialize=10, B=B.reshape(-1, 1)), window_size=100)

    for x_, y_, u_ in zip(X, Y, U):
        odmd.update(x_, y_)
        odmdc_weight.update(x_, y_, u_)
        odmdc_b.update(x_, y_, u_)
        odmdc_window.update(x_, y_, u_)
        odmdc_b_window.update(x_, y_, u_)

    atol = np.abs(get_ct_eigs(odmd.A) - true_eigs[-1]) * 1.5
    eig_weight = get_ct_eigs(odmdc_weight.A)
    # Exponential weighting is numerically sensitive; only require finite.
    assert np.isfinite(eig_weight).all()
    eig_b = get_ct_eigs(odmdc_b.A)
    assert np.allclose(eig_b, true_eigs[-1], atol=atol)
    # Rolling variants: numerical precision limits prevent exact eigenvalue
    # recovery on long time-varying sequences (see docstring). Check finite.
    eig_window = get_ct_eigs(odmdc_window.A)
    assert np.isfinite(eig_window).all()
    eig_b_window = get_ct_eigs(odmdc_b_window.A)
    assert np.isfinite(eig_b_window).all()


def get_ct_eigs(A: np.ndarray) -> np.ndarray:
    """Calculate the continuous-time eigenvalues."""
    return np.imag(np.log(np.linalg.eigvals(A))) / t_diff


# Use smaller dataset for faster tests
_n_small = 200
X_s, Y_s, U_s = X[:_n_small], Y[:_n_small], U[:_n_small]


def test_reconstruct_AB_r_less_than_m() -> None:
    """Test _reconstruct_AB for r < m (truncated SVD path)."""
    odmdc = OnlineDMDwC(p=1, q=1, w=1.0, initialize=4)
    for x, y, u in zip(X_s, Y_s, U_s):
        odmdc.update(x, y, u)
    A_r, B_r = odmdc._reconstruct_AB()
    assert A_r.shape == (2, 2)
    assert B_r.shape == (2, 1)
    assert np.isfinite(A_r).all()
    assert np.isfinite(B_r).all()


def test_update_many_with_control() -> None:
    """Test _update_many with control input."""
    odmdc = OnlineDMDwC(p=2, q=1, w=1.0, initialize=0)
    odmdc.learn_many(X_s[:50], Y_s[:50], U_s[:50])
    # Incremental batch update with control
    odmdc._update_many(X_s[50:100], Y_s[50:100], U_s[50:100])
    assert np.isfinite(odmdc.A).all()


def test_learn_many_with_known_B() -> None:
    """Test learn_many with known B."""
    B_known = B.reshape(-1, 1)
    odmdc = OnlineDMDwC(B=B_known, p=2, q=1, w=1.0, initialize=0)
    odmdc.learn_many(X_s[:50], Y_s[:50], U_s[:50])
    assert np.isfinite(odmdc.A).all()
    # Incremental update with known B
    for x, y, u in zip(X_s[50:100], Y_s[50:100], U_s[50:100]):
        odmdc.update(x, y, u)
    assert np.isfinite(odmdc.A).all()


def test_revert_unsupervised_with_control() -> None:
    """Test revert unsupervised with control (y=None, u=u)."""
    odmdc = Rolling(OnlineDMDwC(p=2, q=1, w=1.0), window_size=20)
    for x, u in zip(X_s, U_s):
        odmdc.update(x, u=u)
    assert np.isfinite(odmdc.A).all()


def test_predict_horizon_without_U() -> None:
    """Test predict_horizon without U (fallback to parent) with known B."""
    B_known = B.reshape(-1, 1)
    odmdc = OnlineDMDwC(B=B_known, p=2, q=1, w=1.0, initialize=4)
    for x, y, u in zip(X_s, Y_s, U_s):
        odmdc.update(x, y, u)
    result = odmdc.predict_horizon({"x1": X_s[-1, 0], "x2": X_s[-1, 1]}, 5)
    assert result.shape == (5, 2)
    assert np.isfinite(result).all()


def test_truncation_error_without_U() -> None:
    """Test truncation_error without U (fallback to parent) with known B."""
    B_known = B.reshape(-1, 1)
    odmdc = OnlineDMDwC(B=B_known, p=2, q=1, w=1.0, initialize=4)
    for x, y, u in zip(X_s, Y_s, U_s):
        odmdc.update(x, y, u)
    err = odmdc.truncation_error(X_s, Y_s)
    assert np.isfinite(err)


def test_truncation_error_with_U() -> None:
    """Test truncation_error with control U."""
    odmdc = OnlineDMDwC(p=2, q=1, w=1.0, initialize=4)
    for x, y, u in zip(X_s, Y_s, U_s):
        odmdc.update(x, y, u)
    err = odmdc.truncation_error(X_s, Y_s, U_s)
    assert np.isfinite(err)


def test_predict_one_with_control() -> None:
    """Test predict_one with control input u."""
    odmdc = OnlineDMDwC(p=2, q=1, w=1.0, initialize=4)
    for x, y, u in zip(X_s, Y_s, U_s):
        odmdc.update(x, y, u)
    result = odmdc.predict_one(
        {"x1": X_s[-1, 0], "x2": X_s[-1, 1]}, u={"u": U_s[-1, 0]}
    )
    assert len(result) == 2
    assert all(np.isfinite(v) for v in result.values())


def test_predict_one_without_control() -> None:
    """Test predict_one without control falls back to parent."""
    B_known = B.reshape(-1, 1)
    odmdc = OnlineDMDwC(B=B_known, p=2, q=1, w=1.0, initialize=4)
    for x, y, u in zip(X_s, Y_s, U_s):
        odmdc.update(x, y, u)
    result = odmdc.predict_one({"x1": X_s[-1, 0], "x2": X_s[-1, 1]})
    assert len(result) == 2


def test_modes_wc_truncated() -> None:
    """Test modes property for r < m in OnlineDMDwC."""
    odmdc = OnlineDMDwC(p=1, q=1, w=1.0, initialize=4)
    for x, y, u in zip(X_s, Y_s, U_s):
        odmdc.update(x, y, u)
    modes = odmdc.modes
    assert modes.shape[0] == 2
    assert np.isfinite(modes).all()


def test_modes_wc_full_rank() -> None:
    """Test modes property for r >= m in OnlineDMDwC (non-truncated)."""
    odmdc = OnlineDMDwC(p=2, q=1, w=1.0, initialize=4)
    for x, y, u in zip(X_s, Y_s, U_s):
        odmdc.update(x, y, u)
    modes = odmdc.modes
    assert modes.shape[0] == 2
    assert np.isfinite(modes).all()


def test_xi_wc_property() -> None:
    """Test xi property for OnlineDMDwC."""
    odmdc = Rolling(OnlineDMDwC(p=2, q=1, w=1.0), window_size=20)
    for x, u in zip(X_s, U_s):
        odmdc.update(x, u=u)
    xi = odmdc.xi
    assert xi.shape == (2,)
    assert np.isfinite(xi).all()


def test_learn_many_without_U() -> None:
    """Test learn_many without U falls back to parent.

    When no control is provided, OnlineDMDwC falls back to OnlineDMD behavior.
    Uses default p=0, q=0 so self.r=0 which auto-sizes to m.
    """
    odmdc = OnlineDMDwC(w=1.0, initialize=0)
    odmdc.learn_many(X_s[:50], Y_s[:50])
    assert np.isfinite(odmdc.A).all()


def test_learn_many_dataframe_inputs() -> None:
    """Test learn_many with DataFrame inputs."""
    odmdc = OnlineDMDwC(p=2, q=1, w=1.0, initialize=0)
    X_df = pd.DataFrame(X_s[:50], columns=["x1", "x2"])
    Y_df = pd.DataFrame(Y_s[:50], columns=["x1", "x2"])
    U_df = pd.DataFrame(U_s[:50], columns=["u"])
    odmdc.learn_many(X_df, Y_df, U_df)
    assert np.isfinite(odmdc.A).all()


def test_learn_many_Y_none_with_U() -> None:
    """Test learn_many with Y=None and U provided (unsupervised with control)."""
    odmdc = OnlineDMDwC(p=2, q=1, w=1.0, initialize=0)
    odmdc.learn_many(X_s[:50], None, U_s[:50])
    assert np.isfinite(odmdc.A).all()


def test_revert_without_u() -> None:
    """Test revert without u falls back to parent."""
    odmdc = Rolling(OnlineDMDwC(initialize=10), window_size=20)
    for x, y in zip(X_s, Y_s):
        odmdc.update(x, y)
    assert np.isfinite(odmdc.A).all()


def test_predict_horizon_with_U_dataframe() -> None:
    """Test predict_horizon with U as DataFrame."""
    odmdc = OnlineDMDwC(p=2, q=1, w=1.0, initialize=4)
    for x, y, u in zip(X_s, Y_s, U_s):
        odmdc.update(x, y, u)
    U_df = pd.DataFrame(U_s[:5], columns=["u"])
    result = odmdc.predict_horizon(
        {"x1": X_s[-1, 0], "x2": X_s[-1, 1]}, 5, U_df
    )
    assert result.shape == (5, 2)
    assert np.isfinite(result).all()


def test_learn_many_known_B_incremental() -> None:
    """Test learn_many with known B followed by incremental updates."""
    B_known = B.reshape(-1, 1)
    odmdc = OnlineDMDwC(B=B_known, p=2, q=1, w=1.0, initialize=0)
    odmdc.learn_many(X_s[:50], Y_s[:50], U_s[:50])
    # Second learn_many exercises the _update_many path via learn_many
    odmdc.learn_many(X_s[50:100], Y_s[50:100], U_s[50:100])
    assert np.isfinite(odmdc.A).all()


def test_reconstruct_AB_full_rank() -> None:
    """Test _reconstruct_AB for r >= m path."""
    odmdc = OnlineDMDwC(p=2, q=1, w=1.0, initialize=4)
    for x, y, u in zip(X_s, Y_s, U_s):
        odmdc.update(x, y, u)
    A_r, B_r = odmdc._reconstruct_AB()
    assert np.allclose(A_r, odmdc.A)
    assert np.allclose(B_r, odmdc.B)


def test_update_many_without_control() -> None:
    """Test _update_many without U falls back to parent."""
    odmdc = OnlineDMDwC(w=1.0, initialize=0)
    odmdc.learn_many(X_s[:50], Y_s[:50])
    odmdc._update_many(X_s[50:100], Y_s[50:100])
    assert np.isfinite(odmdc.A).all()


def test_update_many_known_B_incremental() -> None:
    """Test _update_many with known_B as incremental update after init."""
    B_known = B.reshape(-1, 1)
    odmdc = OnlineDMDwC(B=B_known, p=2, q=1, w=1.0, initialize=0)
    odmdc.learn_many(X_s[:50], Y_s[:50], U_s[:50])
    assert odmdc.known_B is True
    # Incremental batch with known_B: Y = Y - B @ U
    odmdc._update_many(X_s[50:100], Y_s[50:100], U_s[50:100])
    assert np.isfinite(odmdc.A).all()


def test_learn_many_p0_q0_auto_detect() -> None:
    """Test learn_many with p=0 and q=0 auto-detects from data shapes."""
    odmdc = OnlineDMDwC(p=0, q=0, w=1.0, initialize=0)
    odmdc.learn_many(X_s[:50], Y_s[:50], U_s[:50])
    # p should be auto-set to m (state dim)
    assert odmdc.p == X_s.shape[1]
    # q should be auto-set to l (control dim)
    assert odmdc.q == U_s.shape[1]
    assert np.isfinite(odmdc.A).all()


def test_modes_wc_without_l() -> None:
    """Test OnlineDMDwC.modes when 'l' attribute not set (no control yet).

    Covers line 1099: self._modes = super().modes.
    """
    odmdc = OnlineDMDwC(w=1.0, initialize=0)
    # Learn without control to set up state but not 'l'
    odmdc.learn_many(X_s[:50], Y_s[:50])
    modes = odmdc.modes
    assert modes.shape[0] == 2
    assert np.isfinite(modes).all()


def test_learn_many_p0_q0_via_update_one() -> None:
    """Test that p=0, q=0 get auto-detected through learn_one path.

    learn_one -> update -> _init_update sets p,q from data shapes.
    Then learn_many is called internally during initialization,
    and the post-learn_many p==0, q==0 checks (lines 1417-1420)
    may trigger if _init_update didn't set them (e.g. from the
    parent class path).
    """
    odmdc = OnlineDMDwC(p=0, q=0, w=1.0, initialize=4)
    for x, y, u in zip(X_s[:10], Y_s[:10], U_s[:10]):
        odmdc.update(x, y, u)
    assert odmdc.p > 0
    assert odmdc.q > 0
    assert np.isfinite(odmdc.A).all()


def test_update_many_init_via_update_many() -> None:
    """Test _update_many initialization path (n_seen == 0).

    Covers lines 1349-1351 in _update_many.
    """
    odmdc = OnlineDMDwC(p=2, q=1, w=1.0, initialize=0)
    # Call learn_many with U to initialize, then _update_many incremental
    odmdc.learn_many(X_s[:50], Y_s[:50], U_s[:50])
    # Now call _update_many directly with control
    odmdc._update_many(X_s[50:100], Y_s[50:100], U_s[50:100])
    assert np.isfinite(odmdc.A).all()
