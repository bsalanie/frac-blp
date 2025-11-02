"""
FRAC estimation on macro-BLP, without demographics
"""

import numpy as np
import scipy.linalg as spla

from bs_python_utils.bsutils import print_stars, bs_error_abort
from bs_python_utils.bsnputils import TwoArrays

from frac_blp.frac_classes import FracNoDemogData, FracNoDemogSimulatedData
from frac_blp.frac_utils import proj_Z_full, make_Z_full
from frac_blp.artificial_regressors import make_K_and_y


def frac_nodemog_estimate(
    frac_data: FracNoDemogData,
    degree_Z: int = 2,
    degree_X1: int = 2,
) -> TwoArrays:
    """
    Estimate FRAC parameters without demographics using two-stage least squares.

    Args:
        frac_data (FracNoDemogData): Data container with regressors, instruments, and
            simulated or empirical shares.
        degree_Z (int): Degree of polynomial expansion for instruments. Default is 2.
        degree_X1 (int): Degree of polynomial expansion for exogenous regressors in X1.
            Default is 2.

    Returns:
        TwoArrays: Tuple ``(betas_est, sigmas_est)`` with fixed and random coefficient
        estimates, respectively.
    """
    X1_exo = frac_data.X1_exo
    X1, X2 = frac_data.X1, frac_data.X2
    J = frac_data.J
    Z = frac_data.Z
    names_vars_beta = frac_data.names_vars_beta
    names_vars_sigma = frac_data.names_vars_sigma
    shares = frac_data.shares
    K, y = make_K_and_y(X2, shares, J)
    n_x1 = X1.shape[1]
    n_x2 = X2.shape[1]

    # combine exogenous regressors and instruments
    Z_full = make_Z_full(Z, X1_exo, degree_Z=degree_Z, degree_X1=degree_X1)

    # project on the full set of instruments
    y_hat, _, r2_y = proj_Z_full(y.reshape((-1, 1)), Z_full)
    K_hat, _, r2_K = proj_Z_full(K, Z_full)
    X1_hat, _, r2_X1 = proj_Z_full(X1, Z_full)

    print_stars(f"The first stage R2s are (using {Z_full.shape[1]} instruments):")
    print(f"     y: {r2_y[0]:.3f}")
    for ix in range(n_x1):
        print(f"     {names_vars_beta[ix]}: {r2_X1[ix]:.3f}")
    for ix in range(n_x2):
        print(f"     K_{names_vars_sigma[ix]}: {r2_K[ix]:.3f}")
    print("\n")

    # run the second stage
    RHS_proj = np.column_stack((X1_hat, K_hat))
    betas_sigmas_est = spla.lstsq(RHS_proj, y_hat[:, 0])[0]
    betas_est = betas_sigmas_est[:n_x1]
    sigmas_squared_est = betas_sigmas_est[n_x1:]
    if np.min(sigmas_squared_est) < 0.0:
        print_stars("\n The variance estimates are")
        print(sigmas_squared_est)
        bs_error_abort("Negative variance estimate!")
    sigmas_est = np.sqrt(sigmas_squared_est)

    print_stars("Final estimates")
    for i in range(len(names_vars_beta)):
        print(f"   beta1_{names_vars_beta[i]}: {betas_est[i]:.3f}")
    for i in range(len(names_vars_sigma)):
        print(f"   sigma_{names_vars_sigma[i]}: {sigmas_est[i]:.3f}")
    return betas_est, sigmas_est


if __name__ == "__main__":
    T = 50
    J = 20
    n_obs = T * J
    rng = np.random.default_rng(seed=None)
    xi_var = rng.normal(0.0, 1.0, size=(n_obs, 1))
    Z = rng.normal(0.0, 1.0, size=(n_obs, 1))
    rho_xz = np.sqrt(0.5)
    rho_x_xi = np.sqrt(0.5)
    x_var = rho_xz * Z + np.sqrt(1.0 - rho_xz**2) * (
        rho_x_xi * xi_var
        + np.sqrt(1.0 - rho_x_xi**2) * rng.normal(0.0, 1.0, size=(n_obs, 1))
    )
    X1 = np.column_stack((np.ones(n_obs), x_var))
    X2 = x_var.reshape((n_obs, 1))

    betas = np.array([-4.3, 1.0])
    sigma = 1.0
    names_vars_beta = ["constant", "x"]
    names_vars_sigma = ["x"]

    frac_data = FracNoDemogSimulatedData(
        T=T,
        J=J,
        X1_exo=X1,
        X1_endo=None,
        X2_exo=X2,
        X2_endo=None,
        xi_var=xi_var,
        Z=Z,
        names_vars_beta=names_vars_beta,
        names_vars_sigma=names_vars_sigma,
        betas=betas,
        sigmas=np.array([sigma]),
    )

    betas_est, sigmas_est = frac_nodemog_estimate(frac_data)
