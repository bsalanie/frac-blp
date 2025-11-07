"""Helper script to simulate FRAC datasets used in tutorials."""

import numpy as np

from frac_blp.frac_classes import FracNoDemogSimulatedData


def simulate_frac_nodemog_data(
    T: int = 50,
    J: int = 20,
    n_X1_exo: int = 1,
    n_X1_endo: int = 1,
    n_X2_exo: int = 0,
    n_X2_endo: int = 1,
    n_Z: int = 1,
    sigma_x: float = 1.0,
    sigma_xi: float = 1.0,
    rho_x_xi: float = np.sqrt(0.5),
    rho_x_z: float = np.sqrt(0.5),
    betas: np.ndarray = np.array([-4.3, 1.0]),
    sigmas: np.ndarray = np.array([1.0]),
) -> FracNoDemogSimulatedData:
    """
    Simulate FRAC data with endogenous random-coefficient regressors.

    Returns:
        FracNoDemogSimulatedData: Simulated dataset ready for FRAC estimation.
    """
    n_X1 = n_X1_exo + n_X1_endo
    n_X2 = n_X2_exo + n_X2_endo
    names_X1_exo = ["constant"]
    if n_X1_exo > 1:
        names_X1_exo += [f"x_{i + 1}" for i in range(n_X1_exo - 1)]
    names_X1_endo = []
    names_X1_endo = [f"x_{i}" for i in range(n_X1_exo, n_X1)]
    names_X2_exo = [f"x_{i + 1}" for i in range(n_X2_exo)]
    names_X2_endo = [f"x_{i + 1}" for i in range(n_X2_exo, n_X2)]

    return FracNoDemogSimulatedData(
        T=T,
        J=J,
        names_X1_exo=names_X1_exo,
        names_X1_endo=names_X1_endo,
        names_X2_exo=names_X2_exo,
        names_X2_endo=names_X2_endo,
        n_Z=n_Z,
        sigma_x=sigma_x,
        sigma_xi=sigma_xi,
        rho_x_z=rho_x_z,
        rho_x_xi=rho_x_xi,
        betas=betas,
        sigmas=sigmas,
    )


if __name__ == "__main__":
    frac_data = simulate_frac_nodemog_data(T=50, J=20)
    print(frac_data)
