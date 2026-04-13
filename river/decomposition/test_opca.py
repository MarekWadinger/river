"""Tests for OnlinePCA that cover branches unreachable through normal doctests."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

from river.decomposition.opca import OnlinePCA


def test_s_hat_update_when_thresholds_pass():
    """Force the S_hat update branch by patching np.linalg.svd.

    The ratio check sigma_r[n_components] <= (1+tau)*sigma_r[1] is always
    True for valid tau >= 0 because singular values are sorted descending.
    We patch SVD to return values where the ratio check fails, triggering
    the S_hat update.
    """
    np.random.seed(42)
    pca = OnlinePCA(n_components=2, b=3, sigma=0, tau=0, seed=42)
    data = [
        {"a": float(i), "b": float(i * 2), "c": float(i * 3)}
        for i in range(1, 10)
    ]

    # Learn first block normally (n_seen == b-1 triggers SVD init, not the else branch)
    for x in data[:3]:
        pca.learn_one(x)

    s_hat_before = pca.S_hat.copy()

    # Patch SVD for the second block to return singular values where
    # sigma_r[n_components] > (1+tau)*sigma_r[1], forcing the update.
    # sigma_r[2] > (1+0)*sigma_r[1] means sigma_r[2] > sigma_r[1].
    original_svd = np.linalg.svd

    def patched_svd(matrix, **kwargs):
        U, s, Vt = original_svd(matrix, **kwargs)
        # Reverse the order so sigma_r[2] > sigma_r[1]
        s_patched = np.array([s[0], s[-1], s[0]])
        return U, s_patched, Vt

    with patch("numpy.linalg.svd", side_effect=patched_svd):
        for x in data[3:6]:
            pca.learn_one(x)

    # S_hat should have been updated since both threshold checks fail
    assert not np.allclose(s_hat_before, pca.S_hat), (
        "S_hat should have been updated when thresholds pass"
    )
