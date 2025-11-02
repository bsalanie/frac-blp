"""Helper script to simulate FRAC datasets used in tutorials."""

from typing import cast
import numpy as np

from frac_blp.frac_classes import (
    FracNoDemogSimulatedData,
    FracNoDemogSimulationParameters,
)


def simulate_frac_nodemog_data(
    params: FracNoDemogSimulationParameters,
) -> FracNoDemogSimulatedData:
    """
    Simulate FRAC data with endogenous random-coefficient regressors.

    Endogenous regressors `X1_k` are generated as follows:
    $$
    X1_k &= \\sigma_x(\\rho_{xz} z_k \\
        &+ \\sqrt{1 -\\rho_z ^ 2} \\
        &(\\rho_{x\\xi} \\xi / \\sigma_{\\xi}  +   N(0, 1-\\rho_{x\\xi}^2)))
    $$

    Args:
        FracNoDemogSimulationParameters: parameters for the simulation.

    Returns:
        FracNoDemogSimulatedData: Simulated dataset ready for FRAC estimation.
    """
    T, J = (
        params.T,
        params.J,
    )
    n_obs = T * J
    rng = np.random.default_rng(seed=None)
    sigma_x, sigma_xi = (
        params.sigma_x,
        params.sigma_xi,
    )
    xi_var = rng.normal(0.0, sigma_xi, size=n_obs)
    n_Z = params.n_Z
    Z = rng.normal(0.0, 1.0, size=(n_obs, n_Z))
    rho_xz = params.rho_xz
    rho_x_xi = params.rho_x_xi
    root1 = np.sqrt(1 - rho_xz**2)
    root2 = np.sqrt(1 - rho_x_xi**2)
    n_X1_exo, n_X1_endo, n_X2_exo, n_X2_endo = (
        params.n_X1_exo,
        params.n_X1_endo,
        params.n_X2_exo,
        params.n_X2_endo,
    )
    ones_vec = np.ones((n_obs, 1))
    X1_exo = (
        ones_vec
        if n_X1_exo == 0
        else np.column_stack((ones_vec, rng.normal(0.0, 1.0, size=(n_obs, n_X1_exo))))
    )
    X2_exo = None if n_X2_exo == 0 else X1_exo[:, :n_X2_exo]
    X1_endo = None
    if n_X1_endo > 0:
        X1_endo = np.zeros((n_obs, n_X1_endo))
        for i in range(n_X1_endo):
            X1_endo[:, i] = sigma_x * (
                rho_xz * Z[:, i]
                + root1
                * (
                    rho_x_xi * xi_var / sigma_xi
                    + root2 * rng.normal(0.0, 1.0, size=n_obs)
                )
            )
    X2_endo = None
    if n_X2_endo > 0:
        X1_endo = cast(np.ndarray, X1_endo)
        X2_endo = X1_endo[:, :n_X2_endo]
    names_vars_beta = ["constant"] + [f"x_{i + 1}" for i in range(n_X1_exo + n_X1_endo)]
    names_vars_sigma = [f"x_{i + 1}" for i in range(n_X2_exo + n_X2_endo)]

    return FracNoDemogSimulatedData(
        T=T,
        J=J,
        X1_exo=X1_exo,
        X1_endo=X1_endo,
        X2_exo=X2_exo,
        X2_endo=X2_endo,
        Z=Z,
        xi_var=xi_var,
        names_vars_beta=names_vars_beta,
        names_vars_sigma=names_vars_sigma,
        betas=params.betas,
        sigmas=params.sigmas,
    )


if __name__ == "__main__":
    frac_params = FracNoDemogSimulationParameters(T=50, J=20)
    frac_data = simulate_frac_nodemog_data(frac_params)
    print(frac_data)
