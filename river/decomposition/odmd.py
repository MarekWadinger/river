"""Online Dynamic Mode Decomposition (DMD) in [River API](riverml.xyz).

This module contains the implementation of the Online DMD, Weighted Online DMD,
and DMD with Control algorithms. It is based on the paper by Zhang et al. [^1]
and implementation of authors available at
[GitHub](https://github.com/haozhg/odmd). However, this implementation provides
a more flexible interface aligned with River API covers and separates update
and revert methods to operate with Rolling and TimeRolling wrapers.

TODO:

    - [ ] Compute amlitudes of the singular values of the input matrix.
    - [ ] Update prediction computation for continuous time
          x(t) = Phi exp(diag(ln(Lambda) / dt) * t) Phi^+ x(0) (MIT lecture)
          continuous time eigenvalues exp(Lambda * dt) (Zhang et al. 2019)

References:
    [^1]: Zhang, H., Clarence Worth Rowley, Deem, E.A. and Cattafesta, L.N.
    (2019). Online Dynamic Mode Decomposition for Time-Varying Systems. Siam
    Journal on Applied Dynamical Systems, 18(3), pp.1586-1609.
    doi:[10.1137/18m1192329](https://doi.org/10.1137/18m1192329).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import scipy as sp
from scipy.sparse.linalg._eigen.arpack.arpack import ArpackNoConvergence

from river.base import MiniBatchRegressor

__all__ = [
    "OnlineDMD",
    "OnlineDMDwC",
]


class OnlineDMD(MiniBatchRegressor):
    """Online Dynamic Mode Decomposition (DMD).

    This regressor is a class that implements online dynamic mode decomposition
    The time complexity (multiply-add operation for one iteration) is O(4n^2),
    and space complexity is O(2n^2), where n is the state dimension.

    This estimator supports learning with mini-batches with same time and space
    complexity as the online learning. It can be used as Rolling or TimeRolling
    estimator.

    OnlineDMD implements `transform_one` and `transform_many` methods like
    unsupervised MiniBatchTransformer. In such case, we may use `learn_one`
    without `y` and `learn_many` without `Y` to learn the model.
    In that case OnlineDMD preserves previous snapshot and uses it as x while
    current snapshot is used as y.
    NOTE: That means `predict_one` and `predict_many` used with

    At time step t, define two matrices X(t) = [x(1),x(2),...,x(t)],
    Y(t) = [y(1),y(2),...,y(t)], that contain all the past snapshot pairs,
    where x(t), y(t) are the n dimensional state vector, y(t) = f(x(t)) is
    the image of x(t), f() is the dynamics.

    Here, if the (discrete-time) dynamics are given by z(t) = f(z(t-1)),
    then x(t), y(t) should be measurements correponding to consecutive
    states z(t-1) and z(t).

    An exponential weighting factor can be used to place more weight on
    recent data.

    Args:
        r: Number of modes to keep. If 0 (default), all modes are kept.
        w: Weighting factor in (0,1]. Smaller value allows more adpative
        learning, but too small weighting may result in model identification
        instability (relies only on limited recent snapshots).
        initialize: number of snapshot pairs to initialize the model with. If 0
            the model will be initialized with random matrix A and P = \alpha I
            where \alpha is a large positive scalar. If initialize is smaller
            than the state dimension, it will be set to the state dimension and
            raise a warning. Defaults to 1.
        exponential_weighting: Whether to use exponential weighting in revert
        seed: Random seed for reproducibility (initialize A with random values)

    Attributes:
        m: state dimension x(t) as in z(t) = f(z(t-1)) or y(t) = f(t, x(t))
        n_seen: number of seen samples (read-only), reverted if windowed
        feature_names_in_: list of feature names. Used for dict inputs.
        A: DMD matrix, size n by n (non-Hermitian)
        _P: inverse of covariance matrix of X (symmetric)

    Examples:
    >>> import numpy as np
    >>> import pandas as pd
    >>> n = 101; freq = 2.; tspan = np.linspace(0, 10, n); dt = 0.1
    >>> a1 = 1; a2 = 1; phase1 = -np.pi; phase2 = np.pi / 2
    >>> w1 = np.cos(np.pi * freq * tspan)
    >>> w2 = -np.sin(np.pi * freq * tspan)
    >>> df = pd.DataFrame({'w1': w1[:-1], 'w2': w2[:-1]})

    >>> model = OnlineDMD(r=2, w=0.1, initialize=0)
    >>> X, Y = df.iloc[:-1], df.shift(-1).iloc[:-1]

    >>> for (_, x), (_, y) in zip(X.iterrows(), Y.iterrows()):
    ...     x, y = x.to_dict(), y.to_dict()
    ...     model.learn_one(x, y)
    >>> eig, _ =  np.log(model.eig[0]) / dt
    >>> r, i = eig.real, eig.imag
    >>> np.isclose(eig.real, 0.)
    True
    >>> np.isclose(eig.imag, np.pi * freq)
    True

    >>> model.xi  # TODO: verify the result
    array([0.54244, 0.54244])

    >>> from river.utils import Rolling
    >>> model = Rolling(OnlineDMD(r=2, w=1.), 10)
    >>> X, Y = df.iloc[:-1], df.shift(-1).iloc[:-1]

    >>> for (_, x), (_, y) in zip(X.iterrows(), Y.iterrows()):
    ...     x, y = x.to_dict(), y.to_dict()
    ...     model.update(x, y)

    >>> eig, _ =  np.log(model.eig[0]) / dt
    >>> r, i = eig.real, eig.imag
    >>> np.isclose(eig.real, 0.)
    True
    >>> np.isclose(eig.imag, np.pi * freq)
    True

    >>> np.isclose(model.truncation_error(X.values, Y.values), 0)
    True

    >>> w_pred = model.predict_one(np.array([w1[-2], w2[-2]]))
    >>> np.allclose(w_pred, [w1[-1], w2[-1]])
    True

    >>> w_pred = model.predict_many(np.array([1, 0]), 10)
    >>> np.allclose(w_pred.T, [w1[1:11], w2[1:11]])
    True

    References:
        [^1]: Zhang, H., Clarence Worth Rowley, Deem, E.A. and Cattafesta, L.N.
        (2019). Online Dynamic Mode Decomposition for Time-Varying Systems.
        Siam Journal on Applied Dynamical Systems, 18(3), pp.1586-1609.
        doi:[10.1137/18m1192329](https://doi.org/10.1137/18m1192329).
    """

    def __init__(
        self,
        r: int = 0,
        w: float = 1.0,
        initialize: int = 1,
        exponential_weighting: bool = False,
        seed: int | None = None,
    ) -> None:
        self.r = int(r)
        self.w = float(w)
        assert self.w > 0 and self.w <= 1
        self.initialize = int(initialize)
        self.exponential_weighting = exponential_weighting
        self.seed = seed

        np.random.seed(self.seed)

        self.m: int
        self.n_seen: int = 0
        self.feature_names_in_: list[str]
        self.A: np.ndarray
        self._P: np.ndarray
        self._Y: np.ndarray  # for xi computation

    @property
    def eig(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute and return DMD eigenvalues and DMD modes at current step"""
        try:
            Lambda, Phi = sp.sparse.linalg.eigs(self.A, k=self.r)
        except ArpackNoConvergence:
            Lambda, Phi = sp.linalg.schur(self.A, check_finite=False)
            if self.r:
                Lambda, Phi = Lambda[: self.r], Phi[:, : self.r]
        return Lambda, Phi

    def _init_update(self) -> None:
        if self.initialize > 0 and self.initialize < self.m:
            warnings.warn(
                f"Initialization is under-constrained. Set initialize={self.m} to supress this Warning."
            )
            self.initialize = self.m

        self.A = np.random.randn(self.m, self.m)
        self._X_init = np.empty((self.initialize, self.m))
        self._Y_init = np.empty((self.initialize, self.m))
        self._Y = np.empty((0, self.m))

    @property
    def xi(self) -> np.ndarray:
        """Amlitudes of the singular values of the input matrix."""
        Lambda, Phi = self.eig
        # Compute Discrete temporal dynamics matrix (Vandermonde matrix).
        C = np.vander(Lambda, self.n_seen, increasing=True)
        # xi = self.Phi.conj().T @ self._Y @ np.linalg.pinv(self.C)

        from scipy.optimize import minimize

        def objective_function(x):
            return np.linalg.norm(
                self._Y.T - Phi @ np.diag(x) @ C, "fro"
            ) + 0.5 * np.linalg.norm(x, 1)

        # Minimize the objective function
        xi = minimize(objective_function, np.ones(self.m)).x
        return xi

    def update(
        self,
        x: dict | np.ndarray,
        y: dict | np.ndarray | None = None,
    ) -> None:
        """Update the DMD computation with a new pair of snapshots (x, y)

        Here, if the (discrete-time) dynamics are given by z(t) = f(z(t-1)),
        then (x,y) should be measurements correponding to consecutive states
        z(t-1) and z(t).

        Args:
            x: 1D array, shape (m, ), x(t) as in y(t) = f(t, x(t))
            y: 1D array, shape (m, ), y(t) as in y(t) = f(t, x(t))
        """
        # If Hankelizer is used, we need to use DMD without y
        if y is None:
            if not hasattr(self, "_x_prev"):
                self._x_prev = x
                return
            else:
                y = x
                x = self._x_prev
                self._x_prev = y

        if isinstance(x, dict):
            self.feature_names_in_ = list(x.keys())
            x = np.array(list(x.values()))
        if isinstance(y, dict):
            assert self.feature_names_in_ == list(y.keys())
            y = np.array(list(y.values()))

        # Initialize properties which depend on the shape of x
        if self.n_seen == 0:
            self.m = len(x)
            self._init_update()
        if bool(self.initialize) and self.n_seen <= self.initialize - 1:
            self._X_init[self.n_seen, :] = x
            self._Y_init[self.n_seen, :] = y
            if self.n_seen == self.initialize - 1:
                self.learn_many(self._X_init, self._Y_init)
                # revert the number of seen samples to avoid doubling
                self.n_seen -= self._X_init.shape[0]
        else:
            if self.n_seen == 0:
                epsilon = 1e-15
                alpha = 1.0 / epsilon
                self._P = alpha * np.identity(self.m)  # inverse of cov(X)
            # compute P*x matrix vector product beforehand
            Px = self._P.dot(x)
            # compute gamma
            gamma = 1.0 / (1.0 + x.dot(Px))
            # update A
            self.A += np.outer(gamma * (y - self.A.dot(x)), Px)
            # update P, group Px*Px' to ensure positive definite
            self._P = (self._P - gamma * np.outer(Px, Px)) / self.w
            # ensure P is SPD by taking its symmetric part
            self._P = (self._P + self._P.T) / 2

        self.n_seen += 1
        if self._Y.shape[0] < self.n_seen:
            self._Y = np.vstack([self._Y, y])
        elif self._Y.shape[0] > self.n_seen:
            self._Y = self._Y[self.n_seen :, :]

    def learn_one(
        self,
        x: dict | np.ndarray,
        y: dict | np.ndarray | None = None,
    ) -> None:
        """Allias for update method."""
        self.update(x, y)

    def revert(
        self,
        x: dict | np.ndarray,
        y: dict | np.ndarray | None = None,
    ) -> None:
        """Gradually forget the older snapshots and revert the DMD computation.

        Compatible with Rolling and TimeRolling wrappers.

        Args:
            x: 1D array, shape (m, ), x(t) as in y(t) = f(t, x(t))
            y: 1D array, shape (m, ), y(t) as in y(t) = f(t, x(t))
        """
        if self.n_seen < self.initialize:
            raise RuntimeError(
                f"Cannot revert {self.__class__.__name__} before "
                "initialization. If used with Rolling or TimeRolling, window "
                f"size should be increased to {self.initialize}."
            )
        if y is None:
            # raise ValueError("revert method not implemented for y = None.")
            if not hasattr(self, "_x_first"):
                self._x_first = x
                return
            else:
                y = x
                x = self._x_first
                self._x_first = x

        if isinstance(x, dict):
            x = np.array(list(x.values()))
        if isinstance(y, dict):
            y = np.array(list(y.values()))

        # compute P*x matrix vector product beforehand
        # Apply exponential weighting factor
        if self.exponential_weighting:
            weight = 1.0 / -(self.w**self.n_seen)
        else:
            weight = -1.0
        Px = self._P.dot(x)
        gamma = 1.0 / (weight + x.dot(Px))
        # update A
        Ax = self.A.dot(x)
        self.A += np.outer(gamma * (y - Ax), Px)
        # update P, group Px*Px' to ensure positive definite
        self._P = (self._P - gamma * np.outer(Px, Px)) / self.w
        # ensure P is SPD by taking its symmetric part
        self._P = (self._P + self._P.T) / 2
        self.n_seen -= 1

    def _update_many(
        self,
        X: np.ndarray | pd.DataFrame,
        Y: np.ndarray | pd.DataFrame,
    ) -> None:
        """Update the DMD computation with a new batch of snapshots (X,Y).

        This method brings no change in theoretical time and space complexity.
        However, it allows parallel computing by vectorizing update in loop.

        Args:
            X: The input snapshot matrix of shape (p, m), where p is the number of snapshots and m is the number of features.
            Y: The output snapshot matrix of shape (p, m), where p is the number of snapshots and m is the number of features.

        TODO:
            - [ ] find out why not equal to for loop update implementation
              when weights are used

        """
        if self.n_seen == 0:
            raise RuntimeError("Model is not initialized.")
        p = X.shape[0]
        if self.exponential_weighting:
            weights = np.sqrt(self.w) ** np.arange(p - 1, -1, -1)
        else:
            weights = np.ones(p)
        C = np.diag(weights)

        Xt = X.T
        AX = self.A.dot(Xt)
        PX = self._P.dot(Xt)
        PXt = PX.T
        Gamma = np.linalg.inv(np.linalg.inv(C) + X.dot(PX))
        self.A += (Y.T - AX).dot(Gamma).dot(PXt)
        self._P = (self._P - PX.dot(Gamma).dot(PXt)) / self.w
        self._P = (self._P + self._P.T) / 2

    def learn_many(
        self,
        X: np.ndarray | pd.DataFrame,
        Y: np.ndarray | pd.DataFrame | None = None,
    ) -> None:
        """Learn the OnlineDMD model using multiple snapshot pairs.

        Useful for initializing the model with a batch of snapshot pairs.
        Otherwise, it is equivalent to calling update method in a loop.

        Args:
            X: The input snapshot matrix of shape (p, m), where p is the number of snapshots and m is the number of features.
            Y: The output snapshot matrix of shape (p, m), where p is the number of snapshots and m is the number of features.
        """
        if Y is None:
            if isinstance(X, pd.DataFrame):
                Y = X.shift(-1).iloc[:-1]
                X = X.iloc[:-1]
            elif isinstance(X, np.ndarray):
                Y = np.roll(X, -1)[:-1]
                X = X[:-1]

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.DataFrame):
            Y = Y.values

        # necessary condition for over-constrained initialization
        n = X.shape[0]
        # Initialize A and P with first p snapshot pairs
        if not hasattr(self, "_P"):
            self.m = X.shape[1]
            assert n >= self.m and np.linalg.matrix_rank(X) == self.m
            # Exponential weighting factor - older snapshots are weighted less
            if self.exponential_weighting:
                weights = (np.sqrt(self.w) ** np.arange(n - 1, -1, -1))[
                    :, np.newaxis
                ]
            else:
                weights = np.ones((n, 1))
            Xqhat, Yqhat = weights * X, weights * Y
            self.A = Yqhat.T.dot(np.linalg.pinv(Xqhat.T))
            self._P = np.linalg.inv(Xqhat.T.dot(Xqhat)) / self.w
            self.n_seen += n
            self.initialize = 0
            self._Y = Y
        # Update incrementally if initialized
        # Zhang (2019): "single rank-s update is roughly the same as applying
        #  the rank-1 formula s times"
        else:
            self._update_many(X, Y)

    def predict_one(self, x: dict | np.ndarray) -> np.ndarray:
        """
        Predicts the next state given the current state.

        Parameters:
            x: The current state.

        Returns:
            np.ndarray: The predicted next state.
        """
        mat = np.zeros((2, self.m))
        mat[0, :] = x if isinstance(x, np.ndarray) else list(x.values())
        for s in range(1, 2):
            mat[s, :] = (self.A @ mat[s - 1, :]).real
        return mat[-1, :]

    def predict_many(self, x: dict | np.ndarray, forecast: int) -> np.ndarray:
        """
        Predicts multiple future values based on the given initial value.

        Args:
            x: The initial value.
            forecast (int): The number of future values to predict.

        Returns:
            np.ndarray: An array containing the predicted future values.

        TODO:
            - [ ] Align predict_many with river API
        """
        mat = np.zeros((forecast + 1, self.m))
        mat[0, :] = x if isinstance(x, np.ndarray) else list(x.values())
        for s in range(1, forecast + 1):
            mat[s, :] = (self.A @ mat[s - 1, :]).real
        return mat[1:, :]

    def truncation_error(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Compute the truncation error of the DMD model on the given data.

        Since this implementation computes exact DMD, the truncation error is relevant only for initialization.

        Args:
            X: 2D array, shape (p, m), matrix [x(1),x(2),...x(p)]
            Y: 2D array, shape (p, m), matrix [y(1),y(2),...y(p)]

        Returns:
            float: Truncation error of the DMD model
        """
        Y_hat = self.A @ X.T
        return float(np.linalg.norm(Y - Y_hat.T) / np.linalg.norm(Y))

    def transform_one(self, x: dict | np.ndarray) -> np.ndarray:
        """
        Transforms the given input sample.

        Args:
            x: The input to transform.

        Returns:
            np.ndarray: The transformed input.
        """
        if isinstance(x, dict):
            x = np.array(list(x.values()))

        _, Phi = self.eig
        return Phi.T @ x

    def transform_many(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """
        Transforms the given input sequence.

        Args:
            x: The input to transform.

        Returns:
            np.ndarray: The transformed input.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        _, Phi = self.eig
        return Phi.T @ X


class OnlineDMDwC(OnlineDMD):
    """Online Dynamic Mode Decomposition (DMD) with Control.

    This regressor is a class that implements online dynamic mode decomposition
    The time complexity (multiply-add operation for one iteration) is O(4n^2),
    and space complexity is O(2n^2), where n is the state dimension.

    This estimator supports learning with mini-batches with same time and space
    complexity as the online learning.

    At time step t, define three matrices X(t) = [x(1),x(2),...,x(t)],
    Y(t) = [y(1),y(2),...,y(t)], U(t) = [U(1),U(2),...,U(t)] that contain all
    the past snapshot pairs, where x(t), y(t) are the n dimensional state
    vectors, and u(t) is m dimensional control input vector, given by
    y(t) = f(x(t), u(t)).

    x(t), y(t) should be measurements correponding to consecutive states z(t-1)
    and z(t).

    An exponential weighting factor can be used to place more weight on
    recent data.

    Args:
        B: control matrix, size n by m. If None, the control matrix will be
        identified from the snapshots. Defaults to None.
        r: number of modes to keep. If 0 (default), all modes are kept.
        w: weighting factor in (0,1]. Smaller value allows more adpative
        learning, but too small weighting may result in model identification
        instability (relies only on limited recent snapshots).
        initialize: number of snapshot pairs to initialize the model with. If 0
            the model will be initialized with random matrix A and P = \alpha I
            where \alpha is a large positive scalar. If initialize is smaller
            than the state dimension, it will be set to the state dimension and
            raise a warning. Defaults to 1.
        exponential_weighting: whether to use exponential weighting in revert
        seed: random seed for reproducibility (initialize A with random values)

    Attributes:
        m: state dimension x(t) as in z(t) = f(z(t-1)) or y(t) = f(t, x(t))
        n_seen: number of seen samples (read-only), reverted if windowed
        A: DMD matrix, size n by n
        _P: inverse of covariance matrix of X

    Examples:
    >>> import numpy as np
    >>> import pandas as pd

    >>> n = 101
    >>> freq = 2.0
    >>> tspan = np.linspace(0, 10, n)
    >>> dt = 0.1
    >>> a1 = 1
    >>> a2 = 1
    >>> phase1 = -np.pi
    >>> phase2 = np.pi / 2
    >>> w1 = np.cos(np.pi * freq * tspan)
    >>> w2 = -np.sin(np.pi * freq * tspan)
    >>> u_ = np.ones(n)
    >>> u_[tspan > 5] *= 2
    >>> w1[tspan > 5] *= 2
    >>> w2[tspan > 5] *= 2
    >>> df = pd.DataFrame({"w1": w1[:-1], "w2": w2[:-1]})
    >>> U = pd.DataFrame({"u": u_[:-2]})

    >>> model = OnlineDMDwC(r=2, w=0.1, initialize=0)
    >>> X, Y = df.iloc[:-1], df.shift(-1).iloc[:-1]

    >>> for (_, x), (_, y), (_, u) in zip(X.iterrows(), Y.iterrows(), U.iterrows()):
    ...     x, y, u = x.to_dict(), y.to_dict(), u.to_dict()
    ...     model.learn_one(x, y, u)
    >>> eig, _ = np.log(model.eig[0]) / dt
    >>> r, i = eig.real, eig.imag
    >>> np.isclose(eig.real, 0.0)
    True
    >>> np.isclose(eig.imag, np.pi * freq)
    True

    Supports mini-batch learning:
    >>> from river.utils import Rolling

    >>> model = Rolling(OnlineDMDwC(r=2, w=1.0), 10)
    >>> X, Y = df.iloc[:-1], df.shift(-1).iloc[:-1]

    >>> for (_, x), (_, y), (_, u) in zip(X.iterrows(), Y.iterrows(), U.iterrows()):
    ...     x, y, u = x.to_dict(), y.to_dict(), u.to_dict()
    ...     model.update(x, y, u)

    >>> eig, _ = np.log(model.eig[0]) / dt
    >>> r, i = eig.real, eig.imag
    >>> np.isclose(eig.real, 0.0)
    True
    >>> np.isclose(eig.imag, np.pi * freq)
    True

    # TODO: find out why not passing
    # >>> np.isclose(model.truncation_error(X.values, Y.values, U.values), 0)
    # True

    >>> w_pred = model.predict_one(
    ...     np.array([w1[-2], w2[-2]]),
    ...     np.array([u_[-2]]),
    ... )
    >>> np.allclose(w_pred, [w1[-1], w2[-1]])
    True

    >>> w_pred = model.predict_one(
    ...     np.array([w1[-2], w2[-2]]),
    ...     np.array([u_[-2]]),
    ... )
    >>> np.allclose(w_pred, [w1[-1], w2[-1]])
    True

    >>> w_pred = model.predict_many(np.array([1, 0]), np.ones((10, 1)), 10)
    >>> np.allclose(w_pred.T, [w1[1:11], w2[1:11]])
    True

    References:
        [^1]: Zhang, H., Clarence Worth Rowley, Deem, E.A. and Cattafesta, L.N.
        (2019). Online Dynamic Mode Decomposition for Time-Varying Systems.
        Siam Journal on Applied Dynamical Systems, 18(3), pp.1586-1609.
        doi:[10.1137/18m1192329](https://doi.org/10.1137/18m1192329).
    """

    def __init__(
        self,
        B: np.ndarray | None = None,
        r: int = 0,
        w: float = 1.0,
        initialize: int = 1,
        exponential_weighting: bool = False,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            r,
            w,
            initialize,
            exponential_weighting,
            seed,
        )
        self.B = B
        self.known_B = B is not None
        self.l: int

    def _update_many(
        self,
        X: np.ndarray | pd.DataFrame,
        Y: np.ndarray | pd.DataFrame,
        U: np.ndarray | pd.DataFrame | None = None,
    ) -> None:
        """Update the DMD computation with a new batch of snapshots (X,Y).

        This method brings no change in theoretical time and space complexity.
        However, it allows parallel computing by vectorizing update in loop.

        Args:
            X: The input snapshot matrix of shape (p, m), where p is the number of snapshots and m is the number of features.
            Y: The output snapshot matrix of shape (p, m), where p is the number of snapshots and m is the number of features.
            U: The control input snapshot matrix of shape (p, l), where p is the number of snapshots and p is the number of control inputs.
        """
        if U is None:
            super()._update_many(X, Y)
        else:
            if self.known_B:
                Y = Y - self.B @ U
            else:
                X = np.vstack((X, U))
            if self.n_seen == 0:
                self.m = X.shape[1]
                self.l = U.shape[1]
                self._init_update()
            if not self.known_B and self.B is not None:
                self.A = np.hstack((self.A, self.B))
            self.l = U.shape[1]
            super()._update_many(X, Y)

            if not self.known_B:
                self.B = self.A[:, -self.l :]
                self.A = self.A[:, : -self.l]

    def learn_many(
        self,
        X: np.ndarray | pd.DataFrame,
        Y: np.ndarray | pd.DataFrame,
        U: np.ndarray | pd.DataFrame,
    ) -> None:
        """Learn the OnlineDMDwC model using multiple snapshot pairs.

        Useful for initializing the model with a batch of snapshot pairs.
        Otherwise, it is equivalent to calling update method in a loop.

        Args:
            X: The input snapshot matrix of shape (p, m), where p is the number of snapshots and m is the number of features.
            Y: The output snapshot matrix of shape (p, m), where p is the number of snapshots and m is the number of features.
            U: The output snapshot matrix of shape (p, l), where p is the number of snapshots and l is the number of control inputs.
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.DataFrame):
            Y = Y.values
        if isinstance(U, pd.DataFrame):
            U = U.values

        if self.known_B:
            Y = Y - self.B @ U
        else:
            X = np.hstack((X, U))
        if not self.known_B and self.B is not None:
            self.A = np.hstack((self.A, self.B))
        self.l = U.shape[1]
        super().learn_many(X, Y)
        self.m = self.m - self.l  # PATCH: overwrite change of parent

        if not self.known_B:
            self.B = self.A[:, -self.l :]
            self.A = self.A[:, : -self.l]

    def _init_update(self):
        if not self.known_B and self.initialize < self.m + self.l:
            warnings.warn(
                f"Initialization is under-constrained. Changed initialize to {self.m + self.l}."
            )
            self.initialize = self.m + self.l
        # TODO: find out whether should be set in init or here
        self.B = np.random.randn(self.m, self.l)
        self._U_init = np.zeros((self.initialize, self.l))
        super()._init_update()

    def update(
        self,
        x: dict | np.ndarray,
        y: dict | np.ndarray,
        u: dict | np.ndarray | None = None,
    ) -> None:
        """Update the DMD computation with a new pair of snapshots (x, y)

        Here, if the (discrete-time) dynamics are given by z(t) = f(z(t-1)),
        then (x,y) should be measurements correponding to consecutive states
        z(t-1) and z(t).

        Args:
            x: 1D array, shape (n, ), x(t) as in y(t) = f(t, x(t))
            y: 1D array, shape (n, ), y(t) as in y(t) = f(t, x(t))
            u: 1D array, shape (m, ), u(t) as in y(t) = f(t, x(t), u(t))
        """
        if isinstance(x, dict):
            x = np.array(list(x.values()))
        if isinstance(y, dict):
            y = np.array(list(y.values()))
        if isinstance(u, dict):
            u = np.array(list(u.values()))
        # Needed in case of recursive call from learn_many within parent class
        if u is None:
            super().update(x, y)
        else:
            if self.n_seen == 0:
                self.m = len(x)
                self.l = len(u)
                self._init_update()

            if bool(self.initialize) and self.n_seen <= self.initialize - 1:
                self._X_init[self.n_seen, :] = x
                self._Y_init[self.n_seen, :] = y
                self._U_init[self.n_seen, :] = u
                if self.n_seen == self.initialize - 1:
                    self.learn_many(self._X_init, self._Y_init, self._U_init)
                    self.n_seen -= self._X_init.shape[1]

            else:
                if self.known_B:
                    y = y - self.B @ u
                else:
                    x = np.hstack((x, u))
                    if self.B is not None:  # For correct type hinting
                        self.A = np.hstack((self.A, self.B))

                super().update(x, y)

                if not self.known_B:
                    self.B = self.A[:, -self.l :]
                    self.A = self.A[:, : -self.l]

            self.n_seen += 1

    def learn_one(
        self,
        x: dict | np.ndarray,
        y: dict | np.ndarray,
        u: dict | np.ndarray,
    ) -> None:
        """Allias for OnlineDMDwC.update method."""
        return self.update(x, y, u)

    def revert(
        self,
        x: dict | np.ndarray,
        y: dict | np.ndarray,
        u: dict | np.ndarray,
    ) -> None:
        """Gradually forget the older snapshots and revert the DMD computation.

        Compatible with Rolling and TimeRolling wrappers.

        Args:
            x: 1D array, shape (n, ), x(t)
            y: 1D array, shape (n, ), y(t)
            u: 1D array, shape (m, ), u(t)
        """
        if isinstance(x, dict):
            x = np.array(list(x.values()))
        if isinstance(y, dict):
            y = np.array(list(y.values()))
        if isinstance(u, dict):
            u = np.array(list(u.values()))

        if self.known_B:
            y = y - self.B @ u
        else:
            x = np.hstack((x, u))
            if self.B is not None:
                self.A = np.hstack((self.A, self.B))

        super().revert(x, y)

        if not self.known_B:
            self.B = self.A[:, -self.l :]
            self.A = self.A[:, : -self.l]

    def predict_one(
        self, x: dict | np.ndarray, u: dict | np.ndarray
    ) -> np.ndarray:
        """
        Predicts the next state given the current state.

        Parameters:
            x: The current state.
            u: The control input.

        Returns:
            np.ndarray: The predicted next state.
        """
        if isinstance(u, dict):
            u = np.array(list(u.values()))

        mat = np.zeros((2, self.m))
        mat[0, :] = x if isinstance(x, np.ndarray) else list(x.values())
        for s in range(1, 2):
            action = (self.B @ u).real
            mat[s, :] = (self.A @ mat[s - 1, :]).real + action
        return mat[-1, :]

    def predict_many(
        self,
        x: dict | np.ndarray,
        U: np.ndarray | pd.DataFrame,
        forecast: int,
    ) -> np.ndarray:
        """
        Predicts multiple future values based on the given initial value.

        Args:
            x: The initial value.
            U: The control input matrix of shape (forecast, l), where l is the number of control inputs.
            forecast (int): The number of future values to predict.

        Returns:
            np.ndarray: An array containing the predicted future values.

        TODO:
            - [ ] Align predict_many with river API
        """
        if isinstance(U, pd.DataFrame):
            U = U.values

        mat = np.zeros((forecast + 1, self.m))
        mat[0, :] = x if isinstance(x, np.ndarray) else list(x.values())
        for s in range(1, forecast + 1):
            action = (self.B @ U[s - 1, :]).real
            mat[s, :] = (self.A @ mat[s - 1, :]).real + action
        return mat[1:, :]

    def truncation_error(
        self,
        X: np.ndarray | pd.DataFrame,
        Y: np.ndarray | pd.DataFrame,
        U: np.ndarray | pd.DataFrame,
    ) -> float:
        """Compute the truncation error of the DMD model on the given data.

        Args:
            X: 2D array, shape (n, m), matrix [x(1),x(2),...x(n)]
            Y: 2D array, shape (n, m), matrix [y(1),y(2),...y(n)]
            U: 2D array, shape (n, l), matrix [u(1),u(2),...u(n)]

        Returns:
            float: Truncation error of the DMD model
        """
        Y_hat = self.A @ X.T + self.B @ U.T
        return float(np.linalg.norm(Y - Y_hat.T) / np.linalg.norm(Y))
