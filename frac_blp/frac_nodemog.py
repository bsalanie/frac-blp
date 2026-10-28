"""FRAC estimation on macro-BLP, without demographics."""

import numpy as np
from bs_python_utils.bsnputils import TwoArrays
from bs_python_utils.bsutils import bs_error_abort, print_stars

from frac_blp.frac_classes import FracNoDemogData, FracNoDemogSimulatedData
from frac_blp.frac_expansions import xi_terms
from frac_blp.frac_utils import make_y, make_Z_full, proj_on_Z
from frac_blp.twosls import print_tsls_results, tsls_estimate


def frac_nodemog_estimate(
    frac_data: FracNoDemogData,
    degree_Z: int = 2,
    degree_X1: int = 2,
    optimal_instruments: bool = True,
    run_projections: bool = True,
) -> TwoArrays:
    """
    Estimate FRAC parameters without demographics using two-stage least squares.

    Args:
        frac_data (FracNoDemogData): Data container with regressors, instruments, and
            simulated or empirical shares.
        degree_Z (int): Degree of polynomial expansion for instruments. Default is 2.
        degree_X1 (int): Degree of polynomial expansion for exogenous regressors in X1.
            Default is 2.
        optimal_instruments (bool): If True, use optimal instruments. Default is True.
        run_projections (bool): If True, print the R-squared values of the first-stage
            projections on the instruments. Default is True.

    Returns:
        TwoArrays: Tuple ``(betas_sigmas_est, stderrs_est)``: the estimated fixed
        coefficients followed by the estimated variances of the random coefficients,
        and their standard errors.

    Raises:
        SystemExit: If one of the estimated variances is negative.
    """
    X1_exo = frac_data.X1_exo
    X1, X2 = frac_data.X1.astype(np.float64), frac_data.X2.astype(np.float64)
    J, T = frac_data.J, frac_data.T
    Z = frac_data.Z
    names_X1 = frac_data.names_X1
    names_X2 = frac_data.names_X2
    shares = frac_data.shares

    print(f"{shares.min()=}, {shares.max()=}")

    y = make_y(shares, J)
    K = np.zeros_like(X2)

    tJ = 0
    for t in range(T):
        shares_t = shares[tJ : (tJ + J)]
        X2_t = X2[t * J : (t + 1) * J, :]
        term2, term3, term4_gaussian, term4_kappa = xi_terms(shares_t, X2_t)
        K[tJ : (tJ + J), :] = term2
        tJ += J

    n_X1 = X1.shape[1]
    n_X2 = X2.shape[1]

    # combine exogenous regressors and instruments
    Z_full = make_Z_full(Z, X1_exo, degree_Z=degree_Z, degree_X1=degree_X1).astype(
        np.float64
    )

    if run_projections:
        # project on the full set of instruments
        _, _, r2_y = proj_on_Z(y.reshape((frac_data.n_obs, 1)), Z_full)
        _, _, r2_K = proj_on_Z(K, Z_full)
        _, _, r2_X1 = proj_on_Z(X1, Z_full)

        print_stars(
            f"The first stage R2s of projecting on the full set of {Z_full.shape[1]} instruments are:"
        )
        print(f"    for y: {r2_y[0]:.3f}")
        for ix in range(n_X1):
            print(f"     for {names_X1[ix]}: {r2_X1[ix]:.3f}")
        for ix in range(n_X2):
            print(f"     for K_{names_X2[ix]}: {r2_K[ix]:.3f}")
        print("\n")

    # combine covariates and artificial regressors
    X = np.column_stack((X1, K))

    # run TSLS
    print_stars("Running TSLS...")
    betas_sigmas_est, beta_sigmas_varest, xi_est = tsls_estimate(
        y,
        X,
        Z_full,
        J,
        optimal_instruments,
        prev_residuals=None,
    )
    sigmas_squared_est = betas_sigmas_est[n_X1:]
    if np.min(sigmas_squared_est) < 0.0:
        print_stars("\n The variance estimates are")
        print(sigmas_squared_est)
        bs_error_abort("We have a negative variance estimate!")

    stderrs_est = np.sqrt(np.diag(beta_sigmas_varest))

    print_tsls_results(
        betas_sigmas_est,
        stderrs_est,
        names_coeffs=names_X1 + ["sigma2_" + name for name in names_X2],
        title="The FRAC second-order estimates are:",
    )

    return betas_sigmas_est, stderrs_est


if __name__ == "__main__":
    T = 100
    J = 10
    n_obs = T * J
    n_X1_exo = 1
    n_X1_endo = 4
    n_X1 = n_X1_exo + n_X1_endo
    n_X2_exo = 0
    n_X2_endo = n_X1_endo
    n_X2 = n_X2_exo + n_X2_endo
    betas = np.array([-4.3] + [2.0] * (n_X1_exo + n_X1_endo - 1))
    sigmas = np.array([0.3] * (n_X1_exo + n_X1_endo - 1))
    names_X1_endo = [f"x_{i}" for i in range(1, n_X1_endo + 1)]
    names_X1 = ["constant"] + names_X1_endo
    names_X2 = names_X1_endo

    frac_data = FracNoDemogSimulatedData(
        T=T,
        J=J,
        names_X1_exo=["constant"],
        names_X1_endo=names_X1_endo,
        names_X2_exo=[],
        names_X2_endo=names_X1_endo,
        betas=betas,
        sigmas=sigmas,
    )
    print(frac_data)

    betas_sigmas_est, stderrs_est = frac_nodemog_estimate(
        frac_data, optimal_instruments=True, run_projections=True
    )

    print_stars("The true coefficients were")
    print(f"    betas: {betas}")
    print(f"    sigmas: {sigmas}")
