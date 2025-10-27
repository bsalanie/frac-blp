"""Helper script to simulate FRAC datasets used in tutorials."""

import numpy as np

from frac_blp.frac_classes import FracNoDemogSimulatedData


def simulate_frac_nodemog_data(T: int, J: int) -> FracNoDemogSimulatedData:
    """
    Simulate FRAC data with endogenous random-coefficient regressors.

    Args:
        T (int): Number of markets.
        J (int): Number of products per market.

    Returns:
        FracNoDemogSimulatedData: Simulated dataset ready for FRAC estimation.
    """
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
    X1_exo = np.ones((n_obs, 1))
    X1_endo = x_var.reshape((n_obs, 1))
    X2_exo = None
    X2_endo = x_var.reshape((n_obs, 1))

    betas = np.array([-4.3, 1.0])
    sigmas = np.array([1.0])
    names_vars_beta1 = ["constant", "x"]
    names_vars_sigma = ["x"]

    return FracNoDemogSimulatedData(
        T=T,
        J=J,
        X1_exo=X1_exo,
        X1_endo=X1_endo,
        X2_exo=X2_exo,
        X2_endo=X2_endo,
        Z=Z,
        xi_var=xi_var,
        names_vars_beta=names_vars_beta1,
        names_vars_sigma=names_vars_sigma,
        betas=betas,
        sigmas=sigmas,
    )


if __name__ == "__main__":
    frac_data = simulate_frac_nodemog_data(T=50, J=20)
    print(frac_data)
