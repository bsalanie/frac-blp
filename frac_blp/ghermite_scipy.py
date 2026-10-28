import numpy as np
from scipy.special import roots_hermite


def integrate_gaussian(f, dim_f, mu, sigma, points_per_dim=5):
    """
    Integrates f(X) where X ~ N(mu, sigma) up to 5D using
    Tensor Gauss-Hermite Quadrature. Weights are guaranteed to be positive.
    """
    dim = len(mu)
    assert dim <= 5, "This specific grid generation script is optimized for d <= 5"

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
    analytical2 = mu[0] ** 2 + sigma[0, 0]
    analytical2_x4 = mu[1] ** 4 + 6 * mu[1] ** 2 * sigma[1, 1] + 3 * sigma[1, 1] ** 2
    print("Analytical Expected:")
    print([analytical2, analytical2_x4])
    print("Numerical Integral:")
    print(result2)
