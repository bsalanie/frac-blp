"""Helpers to construct Salanié-Wolak artificial regressors."""

from typing import cast

import numpy as np

from bs_python_utils.bsnputils import TwoArrays


def make_K(X: np.ndarray, shares: np.ndarray) -> np.ndarray:
    """
    Build second-order Salanié-Wolak artificial regressors.

    Args:
        X (np.ndarray): Product characteristics of shape ``(n_products, n_x)``.
        shares (np.ndarray): Product-level market shares of shape ``(n_products,)``.

    Returns:
        np.ndarray: Matrix ``K`` with shape ``(n_products, n_x)``.
    """
    eS_X = X.T @ shares
    djm = eS_X - X / 2.0
    return cast(np.ndarray, -djm * X)


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
        K[this_market, :] = make_K(this_X2, these_shares)
        y[this_market] = np.log(these_shares / this_market_zero_share)
    return K, y


# def make_T(X: np.ndarray, shares: np.ndarray) -> np.ndarray:
#     """
#     Build third-order Salanié-Wolak artificial regressors.

#     Args:
#         X (np.ndarray): Product characteristics of shape ``(n_products, n_x)``.
#         shares (np.ndarray): Product-level market shares.

#     Returns:
#         np.ndarray: Matrix ``T`` with shape ``(n_products, n_x)``.
#     """
#     eS_X = X.T @ shares
#     X2 = X * X
#     eS_X2 = X2.T @ shares
#     djm = eS_X - X / 2.0
#     return (X2 / 6.0 + djm * eS_X - eS_X2 / 2.0) * X


# def make_QW(X: np.ndarray, shares: np.ndarray) -> TwoArrays:
#     """
#     Build fourth-order Salanié-Wolak artificial regressors.

#     Args:
#         X (np.ndarray): Product characteristics of shape ``(n_products, n_x)``.
#         shares (np.ndarray): Product-level market shares.

#     Returns:
#         tuple[np.ndarray, np.ndarray]: ``(Q, W)`` where ``Q`` has shape
#         ``(n_products, n_x)`` and ``W`` has shape ``(n_products, n_x, n_x)``.
#     """
#     eS_X = X.T @ shares
#     X2 = X * X
#     X3 = X * X2
#     eS_X2 = X2.T @ shares
#     eS_X3 = X3.T @ shares
#     djm = eS_X - X / 2.0
#     dX = djm * X
#     dX2 = dX * X
#     eS_dX = dX.T @ shares
#     eS_dX2 = dX2.T @ shares
#     Q = (
#         eS_X * eS_X2
#         - djm * eS_X * eS_X
#         + X3 / 24.0
#         - eS_X3 / 6.0
#         - X * eS_X2 / 4.0
#         - X2 * eS_X / 6.0
#     ) * X
#     nproducts, nx = X.shape
#     W = np.zeros((nproducts, nx, nx))
#     dej = djm + eS_X
#     for m in range(nx):
#         Xm = X[:, m]
#         dm = djm[:, m]
#         dXm = dX[:, m]
#         eS_dXm = eS_dX[m]
#         dXXm = dX * Xm.reshape((-1, 1))
#         dXXn = X * dXm.reshape((-1, 1))
#         # eS_XmXndm = dXXm.T @ shares
#         eS_XmXndn = dXXn.T @ shares
#         eS_dX2m = eS_dX2[m]
#         dejm = dej[:, m]
#         W[:, m, :] = (
#             np.outer(Xm * dejm, eS_dX)
#             - np.outer(Xm, eS_XmXndn)
#             - (dX * dXm.reshape((-1, 1))) / 2.0
#         )
#         W[:, m, m] = Xm * (dejm * eS_dXm - eS_dX2m - Xm * dm * dm / 2.0)
#     return (Q, W)
