"""Executable example demonstrating FRAC estimation without demographics.
The example uses simulated data with all `X1` variables generated as N(0,1) iid;
the user can modify the dimensions and other parameters of the simulation as desired.
"""

import numpy as np
import pandas as pd

from bs_python_utils.bsutils import print_stars

from frac_blp.frac_classes import FracNoDemogRealData
from frac_blp.simulate_frac_nodemog_data import (
    simulate_frac_nodemog_data,
)
from frac_blp.frac_nodemog import frac_nodemog_estimate


def run_example(
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
):
    """Simulates data and estimates it with FRAC without demographics;
    then repeats the estimation using the real data interface.
    """
    print_stars("Hello from frac_blp!")
    simulated_frac_data = simulate_frac_nodemog_data(
        T=T,
        J=J,
        n_X1_exo=n_X1_exo,
        n_X1_endo=n_X1_endo,
        n_X2_exo=n_X2_exo,
        n_X2_endo=n_X2_endo,
        n_Z=n_Z,
        sigma_x=sigma_x,
        sigma_xi=sigma_xi,
        rho_x_xi=rho_x_xi,
        rho_x_z=rho_x_z,
        betas=betas,
        sigmas=sigmas,
    )
    print_stars("Simulated Data:")
    print(simulated_frac_data)

    print_stars("Estimating with FRAC")
    _, _ = frac_nodemog_estimate(simulated_frac_data, degree_Z=3, degree_X1=3)

    print_stars("Example with the real data interface:")
    df_X1 = pd.DataFrame(
        np.column_stack(
            (
                simulated_frac_data.X1_exo,
                simulated_frac_data.X1_endo,
            )
        ),
        columns=simulated_frac_data.names_X1,
    )
    real_frac_data = FracNoDemogRealData(
        T=simulated_frac_data.T,
        J=simulated_frac_data.J,
        df_X1=df_X1,
        Z=simulated_frac_data.Z,
        names_X1_exo=simulated_frac_data.names_X1_exo,
        names_X1_endo=simulated_frac_data.names_X1_endo,
        names_X2_exo=simulated_frac_data.names_X2_exo,
        names_X2_endo=simulated_frac_data.names_X2_endo,
        shares=simulated_frac_data.shares,
    )
    print_stars("Estimating with FRAC")
    _, _ = frac_nodemog_estimate(real_frac_data, degree_Z=3, degree_X1=3)


if __name__ == "__main__":
    run_example()
