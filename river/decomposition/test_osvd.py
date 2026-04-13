"""Tests for osvd.py covering complex branches not easily tested via doctests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from river.decomposition.osvd import OnlineSVDZhang


def _make_svd_with_updates(
    n_components: int = 2,
    n_init: int = 6,
    n_total: int = 12,
    m: int = 4,
    seed: int = 42,
    **kwargs: object,
) -> tuple[OnlineSVDZhang, pd.DataFrame]:
    """Create an OnlineSVDZhang model initialized and updated with QR-based data."""
    np.random.seed(seed)
    X = pd.DataFrame(np.linalg.qr(np.random.rand(n_total, m))[0])
    svd = OnlineSVDZhang(n_components=n_components, **kwargs)
    svd.learn_many(X.iloc[:n_init])
    for _, row in X.iloc[n_init:].iterrows():
        svd.update(row.to_dict())
    return svd, X


class TestOnlineSVDZhangInitialize:
    """Test Zhang initialization via learn_one."""

    def test_zhang_init_via_learn_one(self) -> None:
        """When initialize > 0, learn_one accumulates samples until threshold."""
        np.random.seed(0)
        X = pd.DataFrame(np.linalg.qr(np.random.rand(10, 3))[0])
        svd = OnlineSVDZhang(n_components=2, initialize=3, rank_updates=False)
        for _, row in X.iloc[:3].iterrows():
            svd.learn_one(row.to_dict())
        assert svd.n_seen == 3
        assert hasattr(svd, "_U")
        assert svd._U.shape == (3, 2)


class TestOnlineSVDZhangBufferedUpdate:
    """Test Zhang tolerance-triggered buffering and flush."""

    def test_buffered_then_normal_update(self) -> None:
        """Feed near-zero data to trigger buffering, then normal data to flush."""
        np.random.seed(42)
        m = 4
        X_init = pd.DataFrame(np.linalg.qr(np.random.rand(10, m))[0])
        svd = OnlineSVDZhang(n_components=2, rank_updates=False, tol=1e-6)
        svd.learn_many(X_init.iloc[:6])

        tiny = np.zeros((1, m))
        tiny[0, 0] = 1e-15
        svd.update(tiny)
        assert svd._q_u == 1
        assert svd._V_buff.shape[1] == 1

        normal = np.random.rand(1, m)
        svd.update(normal)
        assert svd._q_u == 0
        assert svd._V_buff.shape[1] == 0

    def test_buffered_v_reconstruction(self) -> None:
        """Multiple near-zero updates accumulate buffer; normal update flushes it."""
        np.random.seed(42)
        m = 4
        X_init = pd.DataFrame(np.linalg.qr(np.random.rand(10, m))[0])
        svd = OnlineSVDZhang(n_components=2, rank_updates=False, tol=1e-6)
        svd.learn_many(X_init.iloc[:6])

        for _ in range(3):
            tiny = np.zeros((1, m))
            tiny[0, 0] = 1e-15
            svd.update(tiny)

        assert svd._q_u == 3
        assert svd._V_buff.shape[1] == 3

        normal = np.random.rand(1, m)
        svd.update(normal)
        assert svd._q_u == 0


class TestOnlineSVDZhangReorthogonalize:
    """Test Zhang reorthogonalization path."""

    def test_reorthogonalization_triggered(self) -> None:
        """Data correlated with U[:,0] triggers reorthogonalization of P."""
        np.random.seed(123)
        m = 4
        X_init = pd.DataFrame(np.random.rand(10, m))
        svd = OnlineSVDZhang(n_components=2, rank_updates=False, tol=1e-15)
        svd.learn_many(X_init.iloc[:6])

        x = svd._U[:, 0] * 5.0 + np.random.rand(m) * 0.01
        svd.update(x.reshape(1, -1))
        assert svd._U.shape == (m, 2)


class TestOnlineSVDZhangPtPCond:
    """Test PtP_cond branch where P.T @ W @ P has negative values."""

    def test_ptp_negative_branch(self) -> None:
        """Negative-eigenvalue weighting matrix W triggers the QR fallback."""
        np.random.seed(42)
        m = 4
        X_init = pd.DataFrame(np.linalg.qr(np.random.rand(10, m))[0])
        svd = OnlineSVDZhang(n_components=2, rank_updates=False)
        svd.learn_many(X_init.iloc[:6])

        svd.W = np.diag([1.0, 1.0, -0.5, 1.0])
        x = np.random.rand(1, m)
        svd.update(x)
        assert svd._U.shape == (m, 2)


class TestOnlineSVDZhangRevertQR:
    """Test Zhang revert with _q_r > 0 for all idx variants."""

    def test_revert_qr_path_idx_neg1(self) -> None:
        """Trigger _q_r buffering with high tol, then flush with idx=-1."""
        svd, X = _make_svd_with_updates(rank_updates=False, tol=1e6)

        svd.revert(X.iloc[-1].to_dict(), idx=-1)
        assert svd._q_r == 1

        svd.tol = 1e-12
        svd.revert(X.iloc[-2].to_dict(), idx=-1)
        assert svd._q_r == 0

    def test_revert_qr_path_idx_positive(self) -> None:
        """Trigger _q_r buffering, then flush with idx=0."""
        svd, X = _make_svd_with_updates(rank_updates=False, tol=1e6)

        svd.revert(X.iloc[-1].to_dict(), idx=-1)
        assert svd._q_r == 1

        svd.tol = 1e-12
        svd.revert(X.iloc[0].to_dict(), idx=0)
        assert svd._q_r == 0

    def test_revert_qr_path_idx_negative_not_neg1(self) -> None:
        """Trigger _q_r buffering, then flush with idx=-2."""
        svd, X = _make_svd_with_updates(rank_updates=False, tol=1e6)

        svd.revert(X.iloc[-1].to_dict(), idx=-1)
        assert svd._q_r == 1

        svd.tol = 1e-12
        svd.revert(X.iloc[-3].to_dict(), idx=-2)
        assert svd._q_r == 0


class TestOnlineSVDZhangRankDecreasingRevert:
    """Test Zhang revert with rank_updates=True."""

    def test_rank_decreasing_revert(self) -> None:
        """Revert on rank-increased SVD exercises the rank_updates revert path."""
        np.random.seed(42)
        m = 4
        X_init = pd.DataFrame(np.random.rand(20, m))
        svd = OnlineSVDZhang(n_components=2, rank_updates=True)
        svd.learn_many(X_init.iloc[:6])

        for _, row in X_init.iloc[6:].iterrows():
            svd.update(row.to_dict())
        n_before = svd.n_components

        svd.revert(X_init.iloc[-1].to_dict(), idx=-1)
        assert svd.n_components <= n_before



class TestOnlineSVDZhangRevertQtQNegative:
    """Test the QtQ_cond branch in Zhang revert.

    Q.T @ Q is PSD by construction, so negative values only appear from
    floating-point rounding. We directly manipulate _Vt to force it.
    """

    def test_qtq_negative_forced(self) -> None:
        np.random.seed(0)
        r, m, n = 2, 4, 30
        X = pd.DataFrame(np.linalg.qr(np.random.rand(n, m))[0])
        svd = OnlineSVDZhang(n_components=r, rank_updates=False)
        svd.learn_many(X.iloc[:10])
        for _, row in X.iloc[10:20].iterrows():
            svd.update(row.to_dict())

        # Force _Vt to have values > 1 so Q.T @ Q = (B - V@N).T @ (B - V@N)
        # can have negative entries due to 1 - N.T@N going negative
        svd._Vt[:, -1] = 2.0  # Make N large so Q.T @ Q < 0
        svd.revert(X.iloc[19].to_dict(), idx=-1)
        assert np.isfinite(svd._S).all()


class TestOnlineSVDZhangRankDecrease:
    """Test the rank-decreasing revert path."""

    def test_rank_decrease_revert(self) -> None:
        np.random.seed(42)
        r, m = 2, 4
        X = pd.DataFrame(np.linalg.qr(np.random.rand(20, m))[0])
        svd = OnlineSVDZhang(n_components=r, rank_updates=True, tol=1e6)

        svd.learn_many(X.iloc[:10])
        for _, row in X.iloc[10:15].iterrows():
            svd.update(row.to_dict())

        n_before = svd.n_components
        # With very high tol, S_[-1] <= tol is always true
        svd.revert(X.iloc[14].to_dict(), idx=-1)
        assert svd.n_components <= n_before


class TestOnlineSVDTransformManyUninitialized:
    """Test transform_many before learning."""

    def test_transform_many_uninitialized_dataframe(self) -> None:
        svd = OnlineSVDZhang(n_components=2)
        X = pd.DataFrame(np.ones((3, 4)))
        result = svd.transform_many(X)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == X.shape
