"""TSLS function that can be iterated."""

import numpy as np
from bs_python_utils.bsutils import print_stars

from frac_blp.frac_utils import make_centered_covariance, proj_on_Z


def tsls_estimate(
    y: np.ndarray,
    X: np.ndarray,
    Z: np.ndarray,
    J: int,
    optimal_instruments: bool = True,
    prev_residuals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate $E(y - X\theta | Z) = 0$ by 2SLS with optimal instruments and weighting matrix.

    The previous residuals are used to estimate the within-market covariance
    of the errors, which gives the optimal instruments and the weighting matrix.

    Args:
        y (np.ndarray): Dependent variable, a (T*J,) vector.
        X (np.ndarray): Regressors, a (T*J, K) matrix.
        Z (np.ndarray): Instruments, a (T*J, L) matrix.
        J (int): Number of products in each market.
        optimal_instruments (bool): If True, use optimal instruments. Default is True.
        prev_residuals (np.ndarray | None): Previous residuals, a (T*J,) vector.
            If None, the OLS residuals of ``y`` on ``X`` are used.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: The estimated coefficients
        (a K-vector), their estimated variance (a (K, K) matrix), and the new
        residuals (a (T*J,) vector).
    """
    if prev_residuals is None:  # we use the OLS residuals
        ols_est = np.linalg.lstsq(X, y, rcond=None)[0]
        prev_residuals = y - X @ ols_est

    # First stage: project X onto Z
    D_mat = -proj_on_Z(X, Z)[0]

    n_obs, n_x = X.shape

    if optimal_instruments:
        Z_opt = D_mat
    else:
        Z_opt = Z

    # and the W weighting matrix
    Z_opt_moments = Z_opt * prev_residuals.reshape((-1, 1))
    W_mat = make_centered_covariance(Z_opt_moments)
    print(f"{W_mat=}")
    W_inv = np.linalg.pinv(W_mat)
    print(f"{W_inv=}")

    ZX = Z_opt.T @ X / n_obs
    ZXT = ZX.T
    Zy = Z_opt.T @ y / n_obs
    theta_est = np.linalg.solve(ZXT @ W_inv @ ZX, ZXT @ W_inv @ Zy)

    residuals = y - X @ theta_est

    theta_est_var = np.linalg.inv(ZXT @ W_inv @ ZX) / n_obs

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
        estimated_coeffs (np.ndarray): Estimated coefficients.
        estimated_stderrs (np.ndarray): Estimated standard errors.
        names_coeffs (list[str] | None): Coefficient names. If None, the
            coefficients are labeled ``coeff_0``, ``coeff_1``, ...
        title (str | None): Header printed above the results. If None,
            ``"Two-stage least squares estimates:"`` is used.
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
