"""TSLS function that can be iterated."""

import numpy as np
from bs_python_utils.bsutils import print_stars

from frac_blp.frac_utils import make_centered_covariance, proj_on_Z


def tsls_estimate(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    J: int,
    prev_residuals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run two-stage least squares on $E(y-X\theta|Z) = 0$ with optimal instruments and weighting matrix.

    Args:
        y (np.ndarray): Dependent variable, a T*J vector
        X (np.ndarray): Regressors, a (T*J, K) matrix
        Z (np.ndarray): Instruments, a (T*J,L) matrix
        J (int): the number of products.
        prev_residuals: Previous residuals, a T*J vector. If None, OLS residuals are used.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple containing
          the estimated coefficients,
          their estimated variance, and the new estimated residuals.
    """
    if prev_residuals is None:  # we use the OLS residuals
        ols_est = np.linalg.lstsq(X, y, rcond=None)[0]
        prev_residuals = y - X @ ols_est

    # First stage: project X onto Z
    D_mat = -proj_on_Z(X, Z)[0]

    n_obs, n_x = X.shape
    T = n_obs // J

    # compute an Omega matrix
    Omega = make_centered_covariance(prev_residuals.reshape((T, J)))
    print(f"Omega matrix:\n{Omega.shape}\n", f"{D_mat.shape=}")
    Omega_inv = np.linalg.inv(Omega)
    # now the optimal instruments
    Z_opt = np.zeros((n_obs, n_x))
    tJ = 0
    for t in range(T):
        market_t = slice(tJ, tJ + J)
        Z_opt[market_t, :] = Omega_inv @ D_mat[market_t, :]
        tJ += J

    # and the W weighting matrix
    Z_opt_moments = Z_opt * prev_residuals.reshape((-1, 1))
    W_mat = make_centered_covariance(Z_opt_moments)
    W_inv = np.linalg.pinv(W_mat)
    print(f"{W_inv=}")

    ZX = Z_opt.T @ X / T
    ZXT = ZX.T
    Zy = Z_opt.T @ y / T
    print(f"{ZX.shape=}, {ZXT.shape=}, {Zy.shape=}")
    theta_est = np.linalg.solve(ZXT @ W_inv @ ZX, ZXT @ W_inv @ Zy)

    residuals = y - X @ theta_est

    theta_est_var = np.linalg.inv(ZXT @ W_inv @ ZX) / T

    return theta_est, theta_est_var, residuals


def print_tsls_results(
    estimated_coeffs: np.ndarray,
    estimated_stderrs: np.ndarray,
    names_coeffs: list[str] | None = None,
    title: str | None = None,
) -> None:
    """
    Print the estimated coefficients and their standard errors.

    Args:
        estimated_coeffs: Estimated coefficients.
        estimated_stderrs: Estimated standard errors.
        names_coeffs: List of coefficient names. Just an index is printed if None.
    """
    n_coeffs = estimated_coeffs.size
    if names_coeffs is None:
        names_coeffs = [f"coeff_{i}" for i in range(n_coeffs)]
    if title is None:
        title = "Two-stage least squares estimates:"
    print_stars(title)
    for i in range(len(names_coeffs)):
        print(
            f"   {names_coeffs[i]}: {estimated_coeffs[i]:.3f} (std err: {estimated_stderrs[i]:.3f})"
        )
