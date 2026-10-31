"""Small-sigma expansion of the BLP inversion xi_j(S, X; sigma, c, s, kappa).

Model: beta_i = b + sigma * eps_i, with eps_{il} independent across l,
E eps_l = 0, E eps_l^2 = sigma^2_l, E eps_l^3 = sigma_l^3 * s_l, E eps_l^4 = sigma_l^4 * (3 + kappa_l).

Key identity: for the logit share function s(delta),
    s_j(delta + X t) / s_j(delta) = exp(t'X_j - K(t)),
where K is the cumulant generating function of the characteristics X under the
distribution of shares on {0, ..., J} (with X_0 = 0). Hence
    ln(S_j / S_0) = delta_j + L_j - L_0,   L_j = ln E exp(sigma eps'y_j - [K(sigma eps) - sigma eps'e]),
with y_j = X_j - e (e = (e_1, ..., e_k)). A cumulant expansion of L_j, followed by replacing the
logit shares s(delta) by the observed S, gives xi_j to order sigma^4.

The result is written in raw share-weighted moments of X over the inside goods
(the outside good has X_0 = 0, so it contributes nothing), evaluated at the observed S:
    e_l = sum_j S_j X_jl,  e_2l = sum_j S_j X_jl^2,  e_3l = sum_j S_j X_jl^3,
    v_lm = sum_j S_j X_jl X_jm - e_l e_m.
"""

import numpy as np


def xi_terms(
    S: np.ndarray,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the successive terms of the expansion of xi on all markets.

    xi = y - X*b - term2 * sigmas^2 - term3 * s *  sigmas^3
        - (sigmas^2)' * term4_gaussian * sigmas^2 - term4_kappa * sigmas^4 + O(sigmas^5)

    Args:
        S (np.ndarray): Inside shares, a (T, J) matrix with one row per market.
        X (np.ndarray): Characteristics with random coefficients, a (T, J, K) array.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ``term2``, a (T, J, K)
        array of order 2 terms; ``term3``, a (T, J, K) array of order 3 terms;
        ``term4_gaussian``, a (T, J, K, K) array of Gaussian order 4 terms; and
        ``term4_kappa``, a (T, J, K) array of order 4 terms from excess kurtosis.
    """
    X2 = X * X
    X3 = X2 * X
    # market-level moments, each a (T, K) matrix
    e1 = np.einsum("tj,tjk->tk", S, X)
    e2 = np.einsum("tj,tjk->tk", S, X2)
    e3 = np.einsum("tj,tjk->tk", S, X3)
    v = np.einsum("tj,tjl,tjm->tlm", S, X, X) - np.einsum("tl,tm->tlm", e1, e1)
    # broadcast the market-level moments over the J products
    e1, e2, e3 = e1[:, None, :], e2[:, None, :], e3[:, None, :]

    # order 2:  sum_l (sigma^2_l * term2_l)
    term2 = X * (X / 2.0 - e1)

    # order 3:  sum_l (sigma_l^3 * s_l * term3_l)
    term3 = X * (X2 / 6.0 - X * e1 / 2 + e1**2 - e2 / 2.0)

    # order 4: sum_l sigma_l^4 * kappa_l * term4_kappa_l
    #                   + sum_{l,m} sigma_l^2 sigma_m^2 term4_gaussian_lm
    term4_kappa = X * (
        X3 / 24.0
        - e1 * X2 / 6.0
        + (e1**2 / 2.0 - e2 / 4.0) * X
        + e1 * e2
        - e1**3
        - e3 / 6
    )
    term4_gaussian = -np.einsum("tlm,tjl,tjm->tjlm", v, X, X / 2.0 - e1)

    return term2, term3, term4_gaussian, term4_kappa
