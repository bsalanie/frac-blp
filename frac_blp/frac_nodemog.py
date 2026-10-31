"""FRAC estimation on macro-BLP, without demographics."""

import argparse
from pathlib import Path

import numpy as np
from bs_python_utils.bsnputils import ThreeArrays
from bs_python_utils.bsutils import print_stars

from frac_blp.config import build_frac_data, load_config
from frac_blp.frac_classes import FracNoDemogData
from frac_blp.frac_utils import make_y, make_Z_full, proj_on_Z
from frac_blp.twosls import print_tsls_results, tsls_estimate


def frac_nodemog_estimate(
    frac_data: FracNoDemogData,
    degree_Z: int = 2,
    degree_X1: int = 2,
    optimal_instruments: bool = True,
    run_projections: bool = True,
) -> ThreeArrays:
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

    # print(f"{shares.min()=}, {shares.max()=}")

    y = make_y(shares, J)

    # term2 = X * (X / 2 - e1), with e1 = sum_j S_j X_j, computed for all markets at once
    S_all = shares.reshape((T, J))
    X2_all = X2.reshape((T, J, -1))
    e1 = np.einsum("tj,tjk->tk", S_all, X2_all)
    K = (X2_all * (X2_all / 2.0 - e1[:, None, :])).reshape(X2.shape)

    n_X1 = X1.shape[1]
    n_X2 = X2.shape[1]

    # combine exogenous regressors and instruments
    Z_full = make_Z_full(Z, X1_exo, degree_Z=degree_Z, degree_X1=degree_X1).astype(
        np.float64
    )

    # combine covariates and artificial regressors
    X = np.column_stack((X1, K))

    # project X on the full set of instruments, once: used for the R2s and by TSLS
    X_proj, _, r2_X = proj_on_Z(X, Z_full)

    if run_projections:
        _, _, r2_y = proj_on_Z(y.reshape((frac_data.n_obs, 1)), Z_full)
        r2_X1, r2_K = r2_X[:n_X1], r2_X[n_X1:]

        print_stars(
            f"The first stage R2s of projecting on the full set of {Z_full.shape[1]} instruments are:"
        )
        print(f"    for y: {r2_y[0]:.3f}")
        for ix in range(n_X1):
            print(f"     for {names_X1[ix]}: {r2_X1[ix]:.3f}")
        for ix in range(n_X2):
            print(f"     for K_{names_X2[ix]}: {r2_K[ix]:.3f}")
        print("\n")

    # run TSLS
    print_stars("Running TSLS...")
    betas_sigmas2_est, betas_sigmas2_varest, xi_est = tsls_estimate(
        y,
        X,
        Z_full,
        J,
        optimal_instruments,
        prev_residuals=None,
        X_proj=X_proj,
    )
    sigmas_squared_est = betas_sigmas2_est[n_X1:]
    if np.min(sigmas_squared_est) < 0.0:
        print_stars("\n The variance estimates are")
        print(sigmas_squared_est)
        print_stars("We have a negative variance estimate!")

    stderrs_est = np.sqrt(np.diag(betas_sigmas2_varest))

    print_tsls_results(
        betas_sigmas2_est,
        stderrs_est,
        names_coeffs=names_X1 + ["sigma2_" + name for name in names_X2],
        title="The FRAC second-order estimates are:",
    )

    return betas_sigmas2_est, stderrs_est, xi_est


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Estimate FRAC without demographics using a TOML configuration file."
    )
    parser.add_argument("config", type=Path, help="path to the TOML configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    frac_data = build_frac_data(config)
    print(frac_data)

    est = config.estimation
    betas_sigmas2_est, stderrs_est, xi_est = frac_nodemog_estimate(
        frac_data,
        degree_Z=est.degree_Z,
        degree_X1=est.degree_X1,
        optimal_instruments=est.optimal_instruments,
        run_projections=est.run_projections,
    )

    with open(file=est.results_file, mode="w") as f:
        f.write("The FRAC second-order estimates are:\n")
        for name, est, stderr in zip(
            frac_data.names_X1 + ["sigma2_" + name for name in frac_data.names_X2],
            betas_sigmas2_est,
            stderrs_est,
        ):
            f.write(f"{name}: {est:.6f} ({stderr:.6f})\n")
        f.write("\n")
        f.write("The estimated product effects are:\n")
        for xi_val in xi_est:
            f.write(f"{xi_val:.6f}\n")

    if config.data.source == "simulated":
        print_stars("The true coefficients were")
        print(f"    betas: {config.simulation.betas}")
        print(f"    sigmas2: {np.array(config.simulation.sigmas) ** 2}")
