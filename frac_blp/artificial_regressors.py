"""Helpers to construct Salanié-Wolak artificial regressors."""

from typing import cast

import numpy as np
from bs_python_utils.bsnputils import TwoArrays


def make_this_K(X2: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """
    Build second-order Salanié-Wolak artificial regressors on one market.

    Args:
        X2 (np.ndarray): Product characteristics of shape ``(n_products, n_x2)``.
        shares (np.ndarray): Product-level market shares of shape ``(n_products,)``.

    Returns:
        np.ndarray: Matrix ``K`` with shape ``(n_products, n_x)``.
    """
    eS_X2 = X2.T @ shares
    djm = eS_X2 - X2 / 2.0
    return cast(np.ndarray, -djm * X2)


def make_K_and_y(X2: np.ndarray, shares: np.ndarray, J: int) -> TwoArrays:
    """
    Construct second-order regressors and the log-share LHS by market.

    Args:
        X2 (np.ndarray): Regressors with random coefficients.
        shares (np.ndarray): Observed market shares.
        J (int): Number of products per market.

    Returns:
        TwoArrays: ``(K, y)`` where ``K`` are artificial regressors and ``y`` is the
        stacked log share ratios.
    """
    n_obs = X2.shape[0]
    n_x2 = X2.shape[1]
    K = np.zeros((n_obs, n_x2))  # the artificial Salanie-Wolak regressors
    y = np.zeros(n_obs)  # and the regression LHS

    for t in range(n_obs // J):
        this_market = slice(t * J, (t + 1) * J)
        these_shares = shares[this_market]
        sum_shares = these_shares.sum()
        this_market_zero_share = 1.0 - sum_shares
        this_X2 = X2[this_market, :]

        # artificial regressors and LHS for Salanie-Wolak
        K[this_market, :] = make_this_K(this_X2, these_shares)
        y[this_market] = np.log(these_shares / this_market_zero_share)
    return K, y


def make_this_T(X2: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """
    Construct third-order regressors for a single market.

    Args:
        X2 (np.ndarray): Regressors with random coefficients for a single market.
        shares (np.ndarray): Observed market shares for the same market.

    Returns:
        np.ndarray: Matrix of third-order regressors for the market.
    """
    eS_X2 = X2.T @ shares
    X2_sq = X2 * X2
    eS_X2_sq = X2_sq.T @ shares
    return cast(
        np.ndarray,
        (X2_sq / 6.0 - X2 * eS_X2 / 2.0 - eS_X2_sq / 2.0 + eS_X2 * eS_X2) * X2,
    )


def make_T(X2: np.ndarray, shares: np.ndarray, J: int) -> np.ndarray:
    """
    Construct third-order regressors and the log-share LHS by market.

    Args:
        X2 (np.ndarray): Regressors with random coefficients.
        shares (np.ndarray): Observed market shares.
        J (int): Number of products per market.

    Returns:
        np.ndarray: Matrix ``T`` with shape ``(n_products, n_x2)```.
    """
    n_obs = X2.shape[0]
    n_x2 = X2.shape[1]
    T = np.zeros((n_obs, n_x2))  # the artificial Salanie-Wolak regressors

    for t in range(n_obs // J):
        this_market = slice(t * J, (t + 1) * J)
        these_shares = shares[this_market]
        this_X2 = X2[this_market, :]

        # third-order artificial regressors for Salanie-Wolak
        T[this_market, :] = make_this_T(this_X2, these_shares)

    return T


def make_this_V(X2: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """
    Construct fourth-order regressors for the excess kurtosis for a single market.

    Args:
        X2 (np.ndarray): Regressors with random coefficients for a single market.
        shares (np.ndarray): Observed market shares for the same market.

    Returns:
        np.ndarray: Matrix of fourth-order regressors for the market.
    """
    eS_X2 = X2.T @ shares
    squared_eS_X2 = eS_X2 * eS_X2
    cubed_eS_X2 = squared_eS_X2 * eS_X2
    X2_sq = X2 * X2
    eS_X2_sq = X2_sq.T @ shares
    X2_cube = X2 * X2_sq
    eS_X2_cube = X2_cube.T @ shares
    return cast(
        np.ndarray,
        (
            X2_cube / 24.0
            - X2_sq * eS_X2 / 6.0
            - X2 * eS_X2_sq / 4.0
            + X2 * squared_eS_X2 / 2.0
            + eS_X2 * eS_X2_sq
            - cubed_eS_X2
            - eS_X2_cube / 6.0
        )
        * X2,
    )


def make_V(X2: np.ndarray, shares: np.ndarray, J: int) -> np.ndarray:
    """
    Construct fourth-order regressors for the excess kurtosis by market.

    Args:
        X2 (np.ndarray): Regressors with random coefficients.
        shares (np.ndarray): Observed market shares.
        J (int): Number of products per market.

    Returns:
        np.ndarray: Matrix ``V`` with shape ``(n_products, n_x2)```.
    """
    n_obs = X2.shape[0]
    n_x2 = X2.shape[1]
    V = np.zeros((n_obs, n_x2))

    for t in range(n_obs // J):
        this_market = slice(t * J, (t + 1) * J)
        these_shares = shares[this_market]
        this_X2 = X2[this_market, :]

        # fourth-order artificial regressors for Salanie-Wolak
        V[this_market, :] = make_this_V(this_X2, these_shares)

    return V


def make_this_W(X2: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """
    Construct fourth-order regressors for the products of variances for a single market.

    Args:
        X2 (np.ndarray): Regressors with random coefficients for a single market.
        shares (np.ndarray): Observed market shares for the same market.

    Returns:
        np.ndarray: Matrix of fourth-order regressors for the market.
    """
    eS_X2 = X2.T @ shares
    squared_eS_X2 = np.outer(eS_X2, eS_X2)
    eS_X2_X2 = np.einsum("j,jm,jn->mn", shares, X2, X2)
    covar_eS = eS_X2_X2 - squared_eS_X2
    X2_sq = np.einsum("jm,jn->jmn", X2, X2)
    Xm_eSXn = np.einsum("jm,n->jmn", X2, eS_X2)
    Xn_eSXm = np.einsum("jmn->jnm", Xm_eSXn)
    return cast(
        np.ndarray, np.einsum("jmn,mn->jmn", Xm_eSXn + Xn_eSXm - X2_sq, covar_eS)
    )


def make_W(X2: np.ndarray, shares: np.ndarray, J: int) -> np.ndarray:
    """
    Construct fourth-order regressors for the product of variances by market.

    Args:
        X2 (np.ndarray): Regressors with random coefficients.
        shares (np.ndarray): Observed market shares.
        J (int): Number of products per market.

    Returns:
        np.ndarray: Matrix ``V`` with shape ``(n_products, n_x2)```.
    """
    n_obs = X2.shape[0]
    n_x2 = X2.shape[1]
    W = np.zeros((n_obs, n_x2, n_x2))

    for t in range(n_obs // J):
        this_market = slice(t * J, (t + 1) * J)
        these_shares = shares[this_market]
        this_X2 = X2[this_market, :]

        # fourth-order artificial regressors for Salanie-Wolak
        W[this_market, :, :] = make_this_W(this_X2, these_shares)

    return W
