"""Tensor Gauss-Hermite quadrature for integrals against a Gaussian density."""

from functools import lru_cache

import numpy as np
from scipy.special import roots_hermite


@lru_cache(maxsize=None)
def gauss_hermite_grid(dim, points_per_dim):
    """Return tensor Gauss-Hermite nodes and normalized weights for N(0, I).

    The result depends only on ``dim`` and ``points_per_dim``, so it is cached.
    The returned arrays are shared between calls and must not be modified.

    Args:
        dim (int): Number of dimensions.
        points_per_dim (int): Number of quadrature points in each dimension.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``standard_nodes``, a
        ``(points_per_dim**dim, dim)`` array of nodes, and ``standard_weights``,
        a ``(points_per_dim**dim,)`` array of weights that sum to 1.
    """
    # 1. Get 1D physicists' Gauss-Hermite nodes and strictly positive weights
    # They integrate against the weight function w(x) = exp(-x^2)
    x_1d, w_1d = roots_hermite(points_per_dim)

    # 2. Build the multi-dimensional grid using meshgrid
    mesh_args = [x_1d] * dim
    grids = np.meshgrid(*mesh_args, indexing="ij")
    standard_nodes = np.stack([g.ravel() for g in grids], axis=-1)

    # 3. Build the multi-dimensional weights (outer product of 1D weights)
    w_grids = np.meshgrid(*[w_1d] * dim, indexing="ij")
    standard_weights = np.prod(np.stack([w.ravel() for w in w_grids], axis=-1), axis=1)

    # Normalize weights so they sum up to 1 (converting physicists' weights to a probability distribution)
    # The weight function for physicists' Hermite is exp(-x^2), but normal distributions use exp(-x^2/2) / sqrt(2*pi)
    standard_weights = standard_weights / (np.pi ** (dim / 2))
    return standard_nodes, standard_weights


def integrate_gaussian(f, dim_f, mu, sigma, points_per_dim=5):
    """Integrate f(X) for X ~ N(mu, sigma) by tensor Gauss-Hermite quadrature.

    Supports up to 5 dimensions. The quadrature weights are positive.

    Args:
        f (Callable[[np.ndarray], np.ndarray | float]): Function of one
            node, a ``(dim,)`` array. Returns a scalar if ``dim_f`` is 1, and
            a ``(dim_f,)`` array otherwise.
        dim_f (int): Dimension of the output of ``f``. Use 1 for a scalar.
        mu (np.ndarray): Mean of the Gaussian, a ``(dim,)`` array.
        sigma (np.ndarray): Covariance matrix of the Gaussian, a ``(dim, dim)``
            positive definite array.
        points_per_dim (int): Number of quadrature points in each dimension.
            Default is 5.

    Returns:
        np.ndarray | float: The integral of ``f``: a scalar if ``dim_f`` is 1,
        and a ``(dim_f,)`` array otherwise.

    Raises:
        AssertionError: If ``mu`` has more than 5 elements.
    """
    dim = len(mu)
    assert dim <= 5, "This specific grid generation script is optimized for d <= 5"

    standard_nodes, standard_weights = gauss_hermite_grid(dim, points_per_dim)

    # 4. Transform nodes from Standard Normal space to the target Gaussian N(mu, sigma)
    # Handle covariance via Cholesky decomposition: X = mu + L * (sqrt(2) * standard_nodes)
    L = np.linalg.cholesky(sigma)
    transformed_nodes = mu + (np.sqrt(2) * standard_nodes) @ L.T

    # 5. Evaluate function and calculate the dot product
    # Every single weight in 'standard_weights' is strictly >
    if dim_f == 1:
        f_values = np.array([f(node) for node in transformed_nodes])
        integral = np.dot(f_values, standard_weights)
    else:
        f_values = np.zeros((dim_f, transformed_nodes.shape[0]))
        for i_node, node in enumerate(transformed_nodes):
            f_values[:, i_node] = f(node)
        integral = np.dot(f_values, standard_weights)

    return integral


# --- Example Usage ---
if __name__ == "__main__":
    # Define a 3-dimensional Gaussian environment
    mu = np.array([0.0, 1.0, -0.5])
    sigma = np.array([[1.0, 0.2, 0.1], [0.2, 1.5, 0.3], [0.1, 0.3, 2.0]])

    # Define a sample test function f(X) = sum(X^2)
    def my_function(x):
        return np.sum(x**2)

    # Define a sample test function f(X) = [X[0]^2, X[1]^4]
    def my_function2(x):
        return np.array([x[0] ** 2, x[1] ** 4])

    print("Testing 3D case with f(X) = sum(X^2)")
    # Run the integration using 6 points per dimension (6^3 = 216 total points)
    result = integrate_gaussian(my_function, 1, mu, sigma, points_per_dim=6)

    # Analytical solution for E[sum(X^2)] is sum(mu^2) + trace(sigma)
    analytical = np.sum(mu**2) + np.trace(sigma)

    print(f"Numerical Integral: {result:.6f}")
    print(f"Analytical Expected: {analytical:.6f}")

    # Run the integration using 6 points per dimension (6^2= 36 total points)
    result2 = np.array(
        integrate_gaussian(my_function2, 2, mu[:2], sigma[:2, :2], points_per_dim=6)
    )

    # Analytical solution for E[X^2] is mu^2 + np.diag(sigma); for X^4 is mu^4 + 6*mu^2*np.diag(sigma) + 3*np.diag(sigma)**2
    print("Testing 2D case with f(X) = [X[0]^2, X[1]^4]")
    analytical2 = mu[0] ** 2 + sigma[0, 0]
    analytical2_x4 = mu[1] ** 4 + 6 * mu[1] ** 2 * sigma[1, 1] + 3 * sigma[1, 1] ** 2
    print("Analytical Expected:")
    print([analytical2, analytical2_x4])
    print("Numerical Integral:")
    print(result2)
