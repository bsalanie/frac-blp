"""TSLS function that can be iterated."""

import numpy as np

from frac_blp.frac_utils import make_centered_covariance, proj_Z_full


def tsls_iterate(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    prev_residuals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a single iteration of two-stage least squares.

    Args:
        y (np.ndarray): Dependent variable.
        X (np.ndarray): Regressors.
        Z (np.ndarray): Instruments.
        prev_residuals: Previous residuals, a (T,) array.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing the estimated coefficients, moments,  and objective.
    """
    if prev_residuals is None:  # we use the OLS residuals
        ols_est = np.linalg.lstsq(X, y, rcond=None)[0]
        prev_residuals = y - X @ ols_est

    # First stage: project X onto Z
    D_mat, _, _ = -proj_Z_full(X, Z)

    n_markets = X.shape[0]

    # compute an Omega matrix
    Omega = make_centered_covariance(prev_residuals)
    # now the optimal instruments
    Z_opt = D_mat.T @ Omega

    # and the W weighting matrix
    Z_opt_moments = Z_opt @ prev_residuals
    W_mat = make_centered_covariance(Z_opt_moments)
    W_inv = np.linalg.pinv(W_mat)

    ZX = Z_opt @ X / n_markets
    ZXT = ZX.T
    Zy = Z_opt @ y / n_markets
    theta_est = np.linalg.solve(ZXT @ W_inv @ ZX, ZXT @ W_inv @ Zy)

    residuals = y - X @ theta_est

    return theta_est, residuals
