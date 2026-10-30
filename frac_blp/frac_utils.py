"""Utility helpers for building instruments and projections in FRAC."""

from itertools import combinations_with_replacement
from typing import cast

import numpy as np
from scipy import linalg as spla


def mnl_probabilities(u: np.ndarray, outside: bool = False) -> np.ndarray:
    """Return the MNL probabilities for utilities u.

    Args:
        u (np.ndarray): Utilities of the J alternatives, a 1D array.
        outside (bool): If True, include an outside option with utility 0.

    Returns:
        np.ndarray: The J probabilities ``exp(u) / sum(exp(u))`` or, if
        ``outside`` is True, ``exp(u) / (1 + sum(exp(u)))``.
    """
    u = np.asarray(u)
    u_max = np.max(u)
    exp_u = np.exp(u - u_max)
    if outside:
        return np.asarray(exp_u / (np.exp(-u_max) + np.sum(exp_u)))
    return np.asarray(exp_u / np.sum(exp_u))


def make_X(X_exo: np.ndarray | None, X_endo: np.ndarray | None) -> np.ndarray:
    """
    Combine exogenous and endogenous regressors into a single regressor matrix.

    Args:
        X_exo (np.ndarray | None): Exogenous regressors.
        X_endo (np.ndarray | None): Endogenous regressors.

    Returns:
        np.ndarray: The concatenated regressor matrix.

    Raises:
        ValueError: If both `X_exo` and `X_endo` are ``None``.
    """
    if X_exo is not None and X_endo is not None:
        X = np.column_stack((X_exo, X_endo))
    elif X_exo is not None:
        X = X_exo
    elif X_endo is not None:
        X = X_endo
    else:
        raise ValueError("At least one of X_exo or X_endo must be provided.")
    return X


def make_y(shares: np.ndarray, J: int) -> np.ndarray:
    """
    Create the dependent variable vector for FRAC estimation.

    Args:
        shares (np.ndarray): Market shares, a (T*J,) vector stacked by market.
        J (int): Number of products in each market.

    Returns:
        np.ndarray: The log of each share divided by the outside share
        of its market, with the same shape as ``shares``.
    """
    n_markets = shares.shape[0] // J
    shares_mat = shares.reshape((n_markets, J))
    print(f"{shares_mat.min()=}", f"{shares_mat.max()=}")
    shares_zero = 1.0 - shares_mat.sum(axis=1)
    print(f"{shares_zero.min()=}")
    y = np.log(shares_mat / shares_zero.reshape((-1, 1)))
    return cast(np.ndarray, y.reshape(shares.shape))


def _monomials_of_degree(mat: np.ndarray, d: int) -> list[np.ndarray]:
    n_obs, n_cols = mat.shape
    if d == 0:
        return [np.ones(n_obs)]
    result = []
    for combo in combinations_with_replacement(range(n_cols), d):
        term = np.ones(n_obs)
        for c in combo:
            term = term * mat[:, c]
        result.append(term)
    return result


def make_Z_full(
    Z: np.ndarray,
    X1_exo: np.ndarray | None = None,
    degree_Z: int = 2,
    degree_X1: int = 2,
) -> np.ndarray:
    """
    Build a full set of polynomial instruments for FRAC without demographics.

    Args:
        Z (np.ndarray): Baseline instruments.
        X1_exo (np.ndarray | None): Exogenous regressors without random coefficients.
        degree_Z (int): Maximum degree applied to columns of ``Z`` (must be >= 0).
        degree_X1 (int): Maximum degree applied to columns of ``X1_exo``.

    Returns:
        np.ndarray: Instrument matrix whose columns enumerate every admissible
        combination of polynomial terms, preceded by a column of ones.

    Raises:
        ValueError: If ``degree_Z`` is negative.
    """
    if degree_Z < 0:
        raise ValueError("degree_Z must be non-negative.")

    n_obs = Z.shape[0]
    effective_X1 = X1_exo if X1_exo is not None else np.ones((n_obs, 1))
    max_dx1 = degree_X1 if X1_exo is not None else 0

    columns: list[np.ndarray] = []
    for d_z in range(degree_Z + 1):
        for z_term in _monomials_of_degree(Z, d_z):
            for d_x1 in range(max_dx1 + 1):
                for x1_term in _monomials_of_degree(effective_X1, d_x1):
                    columns.append(z_term * x1_term)

    return np.column_stack(columns)


def proj_on_Z(
    X: np.ndarray, Z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project each column of ``X`` onto ``Z`` and report fitted values.

    Args:
        X (np.ndarray): Variables to project, an (n, K) matrix.
        Z (np.ndarray): Instruments used for the projections, an (n, L) matrix.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``X_proj``, an (n, K) matrix
        of projected columns; ``coef``, an (L, K) matrix of least-squares
        coefficients; and ``r2``, a K-vector of column-wise R-squared values.
    """
    EPS = 1e-12
    # one least-squares call handles all columns of X
    coef = spla.lstsq(Z, X)[0]
    X_proj = Z @ coef
    var_x = np.var(X, axis=0)
    var_x_proj = np.var(X_proj, axis=0)
    # columns with (numerically) zero variance get r2 = 1, as before
    safe_var_x = np.where(var_x < EPS, 1.0, var_x)
    r2 = np.where(var_x < EPS, 1.0, var_x_proj / safe_var_x)
    return X_proj, coef, r2


def make_centered_covariance(X: np.ndarray) -> np.ndarray:
    """
    Compute the centered covariance matrix of a given matrix.

    Args:
        X (np.ndarray): Input matrix with shape (n_samples, n_features).

    Returns:
        np.ndarray: Centered covariance matrix with shape (n_features, n_features).
    """
    n_samples = X.shape[0]
    X_centered = X - np.mean(X, axis=0)
    return cast(np.ndarray, (X_centered.T @ X_centered) / n_samples)
