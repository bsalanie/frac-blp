"""Numerical check of the small-sigma expansion of xi.

Simulates eps following a mixture of two Gaussians with mean zero.
The mixture is parameterized as:
    eps ~ weight * N(mu1, var1) + (1 - weight) * N(mu2, var2)
where mu1 + mu2 are set to make the overall mean zero.
"""

from typing import Any

import numpy as np

from frac_blp.expansions import xi_terms
from frac_blp.frac_utils import mnl_probabilities


def shares_mixture_gaussian_eps(
    J, k, weight1, mu1, var1, mu2, var2, c, n_draws, rng, X, b, xi, sigma
):
    """Simulated shares from `k` mixtures of two Gaussians. For `j=0,...,k-1`,
    eps_j ~ (weight1[j] * N(mu1[j], var1[j]) + (1 - weight1[j]) * N(mu2[j], var2[j])) * sqrt(c[j])

    Parameters
    ----------
    J : int
        Number of products
    k : int
        Number of characteristics (features)
    weight1 : array-like of shape (k,) in (0,1)
        Weights on first Gaussian
    mu1 : array-like of shape (k,)
        Means of first Gaussian
    var1 : array-like of shape (k,)
        Variances of first Gaussian
    mu2 : array-like of shape (k,)
        Means of second Gaussian
    var2 : array-like of shape (k,)
        Variances of second Gaussian
    c : array-like of shape (k,)
        Variances
    n_draws : int
        Number of draws for Monte Carlo approximation
    rng : np.random.Generator
        Random number generator
    X : array of shape (J, k)
        Characteristics matrix
    b : array of shape (k,)
        Coefficients
    xi : array of shape (J,)
        Unobserved utilities
    sigma : float
        Scaling parameter

    Returns
    -------
    S : array of shape (J,)
        Simulated market shares
    """
    eps_draws = np.zeros((n_draws, k))
    sig1 = np.sqrt(var1)
    sig2 = np.sqrt(var2)
    c_root = np.sqrt(c)
    for j in range(k):
        sig1c = sig1[j] * c_root[j]
        sig2c = sig2[j] * c_root[j]
        mixture2 = rng.uniform(size=n_draws) > weight1[j]
        eps_draws[:, j] = rng.normal(mu1[j], sig1c, size=n_draws)
        eps_draws[mixture2, j] = rng.normal(mu2[j], sig2c, size=mixture2.sum())

    u = (sigma * eps_draws * c + b) @ X.T + xi
    probs = np.asarray([mnl_probabilities(u[i], outside=True) for i in range(n_draws)])
    print(f"{probs.shape=}")
    shares = np.mean(probs, axis=0)
    return shares


def run(
    J,
    k,
    weight1,
    mu1,
    var1,
    X,
    b,
    xi,
    c,
    n_draws=10_000,
    seed=None,
) -> None:
    """Run expansion check with mixture of Gaussians for eps.

    Parameters
    ----------
    J : int
        Number of products
    k : int
        Number of characteristics
    weight1 : array-like of shape (k,) in (0,1)
        Weights on first Gaussian
    mu1, var1 : array-like of shape (k,)
        Parameters of the first Gaussian
    X : array of shape (J, k)
        Characteristics matrix
    b : array of shape (k,)
        Coefficients
    xi : array of shape (J,)
        Unobserved utilities
    c: array-like of shape (k,)
        variances
    n_draws : int, optional
        Number of Monte Carlo draws (default 10,000)
    seed : int, optional
        Random seed for reproducibility
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    mu1 = np.asarray(mu1)
    var1 = np.asarray(var1)
    mu2 = -mu1 * weight1 / (1.0 - weight1)
    dmu = mu2 - mu1
    var2 = (1.0 - weight1 * (1.0 - weight1) * dmu * dmu - weight1 * var1) / (
        1.0 - weight1
    )
    if not np.all(var2 > 0):
        raise ValueError(f"Mixture variance not positive: {var2}. Try reducing mu1.")

    # Verify mean zero and variance one
    overall_means = weight1 * mu1 + (1 - weight1) * mu2
    if not np.allclose(overall_means, 0.0, atol=1e-10):
        raise ValueError(
            f"Mixture mean not zero: {overall_means}. "
            f"Set weight * mu1 + (1 - weight) * mu2 = 0."
        )

    overall_vars = (
        weight1 * var1 + (1 - weight1) * var2 + weight1 * (1.0 - weight1) * dmu * dmu
    )
    if not np.allclose(overall_vars, 1.0, atol=1e-10):
        raise ValueError(f"Mixture variance not one: {overall_vars}. ")

    sigma_scales = np.array([1.0, 0.5, 0.2, 0.1, 0.05, 0.01])
    labels = ["order 0", "order 2", "order 3", "order 4"]
    errs = np.zeros((len(sigma_scales), 4))

    for i, sig in enumerate(sigma_scales):
        # Sample shares
        S = shares_mixture_gaussian_eps(
            J, k, weight1, mu1, var1, mu2, var2, c, n_draws, rng, X, b, xi, sig
        )
        sig2 = sig * sig
        varc = sig2 * c

        # Compute moments of the mixture
        m2 = weight1 * (mu1 * mu1 + var1) + (1.0 - weight1) * (mu2 * mu2 + var2)
        m2 *= varc
        m3 = weight1 * (mu1**3 + 3 * mu1 * var1) + (1 - weight1) * (
            mu2**3 + 3 * mu2 * var2
        )
        m3 *= varc * np.sqrt(varc)
        m4 = weight1 * (3 * var1**2 + 6 * mu1**2 * var1 + mu1**4) + (1 - weight1) * (
            3 * var2**2 + 6 * mu2**2 * var2 + mu2**4
        )
        kappas = m4 - 3.0

        print(f"\nsigma = {sig}")

        print(f"  varc = {varc}")
        print(f"m2^2 = {m2 * m2}")
        print(f"  m3 = {m3}")
        print(f"m4={m4}")
        print(f"  kappas = {kappas}")

        term0, term2, term3, term4_gaussian, term4_kappa = xi_terms(S, X, b)

        print(f"  \nxi = {xi}")
        print(f"  term0 = {term0}")
        print(f"  term2 @ m2 = {term2 @ m2}")
        print(f"  term3 @ m3 = {term3 @ m3}")
        print(f"  term4_gaussian = {np.einsum('l, jlm, m->j', m2, term4_gaussian, m2)}")
        print(f"  term4_kappa = {term4_kappa @ (m2 * m2 * kappas)}")

        approx_0 = term0.copy()
        approx_2 = term0 - term2 @ m2
        approx_3 = approx_2 - term3 @ m3
        approx_4 = (
            approx_3
            - np.einsum("l, jlm, m->j", m2, term4_gaussian, m2)
            - term4_kappa @ (m2 * m2 * kappas)
        )
        print("\n Approximation errors:")
        print(f"{np.column_stack([approx_0, approx_2, approx_3, approx_4]).T - xi}")

        errs[i, 0] = np.abs(approx_0 - xi).max()
        errs[i, 1] = np.abs(approx_2 - xi).max()
        errs[i, 2] = np.abs(approx_3 - xi).max()
        errs[i, 3] = np.abs(approx_4 - xi).max()

    print(f"J={J}, k={k}")
    print(f"Mixture: weights={weight1}, c={c}")
    print(f"  Gaussian 1: mu={mu1}, var={var1}")
    print(f"  Gaussian 2: mu={mu2}, var={var2}")
    print("max_j |xi_approx - xi_true|")
    print("sigma    " + "".join(f"{label:>12}" for label in labels))
    for sig, e in zip(sigma_scales, errs):
        print(f"{sig:<8.3f} " + "".join(f"{v:12.3e}" for v in e))
    slopes = np.diff(np.log(errs), axis=0) / np.diff(np.log(sigma_scales))[:, None]
    print("log-log slopes (expected 2, 3, 4, 5):")
    for row in slopes:
        print("         " + "".join(f"{v:12.2f}" for v in row))


if __name__ == "__main__":
    J = 5
    k = 3
    seed = 89675
    rng = np.random.default_rng(seed)

    X = np.column_stack([np.ones(J), rng.normal(size=(J, k - 1))])
    b = rng.uniform(low=-1.0, high=1.0, size=k)
    xi = 0.5 * rng.normal(size=J)

    # Mixture parameters: weight on first Gaussian, means and variances
    # Constraint: weight * mu1 + (1 - weight) * mu2 = 0 (mean zero)
    # Constraint: weight * var1 + (1 - weight) * var2 + weight * mu1^2 + (1 - weight) * mu2^2 = 1 (variance one)
    weight1 = rng.uniform(low=0.1, high=0.9, size=k)
    mu1 = rng.uniform(low=-0.5, high=0.5, size=k)
    var1 = rng.uniform(low=0.0, high=0.5, size=k)
    c: np.ndarray[tuple[Any, ...], np.dtype[np.float64]] = rng.uniform(
        low=0.0, high=2.0, size=k
    )
    c[0] = 1.0  # normalization
    c = np.ones(k)
    run(J, k, weight1, mu1, var1, X, b, xi, c, n_draws=100_000, seed=seed)
