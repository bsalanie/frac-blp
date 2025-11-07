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
    X1, X2 = frac_data.X1.astype(np.float64), frac_data.X2.astype(np.float64)
    J = frac_data.J
    Z = frac_data.Z
    names_X1 = frac_data.names_X1
    names_X2 = frac_data.names_X2
    shares = frac_data.shares
    K, y = make_K_and_y(X2, shares, J)
    K = K.astype(np.float64)
    y = y.astype(np.float64)
    n_X1 = X1.shape[1]
    n_X2 = X2.shape[1]

    # combine exogenous regressors and instruments
    Z_full = make_Z_full(Z, X1_exo, degree_Z=degree_Z, degree_X1=degree_X1).astype(
        np.float64
    )

    # project on the full set of instruments
    y_hat, _, r2_y = proj_Z_full(y.reshape((frac_data.n_obs, 1)), Z_full)
    K_hat, _, r2_K = proj_Z_full(K, Z_full)
    X1_hat, _, r2_X1 = proj_Z_full(X1, Z_full)

    # breakpoint()
    print_stars(
        f"The first stage R2s of projecting on the full set of {Z_full.shape[1]} instruments are:"
    )
    print(f"    for y: {r2_y[0]:.3f}")
    for ix in range(n_X1):
        print(f"     for {names_X1[ix]}: {r2_X1[ix]:.3f}")
    for ix in range(n_X2):
        print(f"     for K_{names_X2[ix]}: {r2_K[ix]:.3f}")
    print("\n")

    # run the second stage
    RHS_proj = np.column_stack((X1_hat, K_hat))
    betas_sigmas_est = spla.lstsq(RHS_proj, y_hat[:, 0])[0]
    betas_est = betas_sigmas_est[:n_X1]
    sigmas_squared_est = betas_sigmas_est[n_X1:]
    if np.min(sigmas_squared_est) < 0.0:
        print_stars("\n The variance estimates are")
        print(sigmas_squared_est)
        bs_error_abort("We have a negative variance estimate!")
    sigmas_est = np.sqrt(sigmas_squared_est)

    print_stars("The final estimates are:")
    for i in range(len(names_X1)):
        print(f"   beta1_{names_X1[i]}: {betas_est[i]:.3f}")
    for i in range(len(names_X2)):
        print(f"   sigma_{names_X2[i]}: {sigmas_est[i]:.3f}")
    return betas_est, sigmas_est


if __name__ == "__main__":
    T = 50
    J = 20
    n_obs = T * J
    betas = np.array([-4.3, 1.0])
    sigma = 1.0
    names_X1 = ["constant", "x"]
    names_X2 = ["x"]

    frac_data = FracNoDemogSimulatedData(
        T=T,
        J=J,
        names_X1_exo=["constant"],
        names_X1_endo=["x"],
        names_X2_exo=[],
        names_X2_endo=["x"],
        betas=betas,
        sigmas=np.array([sigma]),
    )

    betas_est, sigmas_est = frac_nodemog_estimate(frac_data)
