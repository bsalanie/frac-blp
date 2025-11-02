"""Executable example demonstrating FRAC estimation without demographics."""

import numpy as np
from bs_python_utils.bsutils import print_stars

from frac_blp.frac_classes import FracNoDemogSimulationParameters, FracNoDemogRealData
from frac_blp.simulate_frac_nodemog_data import simulate_frac_nodemog_data
from frac_blp.frac_nodemog import frac_nodemog_estimate


def run_example():
    """Run the FRAC estimation example on simulated data."""
    print_stars("Hello from fracblp!")
    params = FracNoDemogSimulationParameters(
        T=100,
        J=10,
        n_X1_endo=2,
        n_X2_endo=2,
        n_Z=3,
        betas=np.array([-4.0, 0.5, -0.5]),
        rho_x_z=0.5,
        sigmas=np.array([0.7, 0.6]),
    )
    simulated_frac_data = simulate_frac_nodemog_data(params)
    print_stars("Simulated Data:")
    print(simulated_frac_data)

    print_stars("Estimating with FRAC")
    _, _ = frac_nodemog_estimate(simulated_frac_data, degree_Z=3, degree_X1=3)

    print_stars("Example with the real data interface:")
    real_frac_data = FracNoDemogRealData(
        T=simulated_frac_data.T,
        J=simulated_frac_data.J,
        X1_exo=simulated_frac_data.X1_exo,
        X1_endo=simulated_frac_data.X1_endo,
        X2_exo=simulated_frac_data.X2_exo,
        X2_endo=simulated_frac_data.X2_endo,
        Z=simulated_frac_data.Z,
        shares=simulated_frac_data.shares,
        names_vars_beta=simulated_frac_data.names_vars_beta,
        names_vars_sigma=simulated_frac_data.names_vars_sigma,
    )
    print_stars("Estimating with FRAC")
    _, _ = frac_nodemog_estimate(real_frac_data, degree_Z=3, degree_X1=3)


if __name__ == "__main__":
    run_example()
