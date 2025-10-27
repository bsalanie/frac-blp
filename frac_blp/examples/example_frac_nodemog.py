"""Executable example demonstrating FRAC estimation without demographics."""

from bs_python_utils.bsutils import print_stars

from frac_blp.simulate_frac_nodemog_data import simulate_frac_nodemog_data
from frac_blp.frac_nodemog import frac_nodemog_estimate


def run_example():
    """Run the FRAC estimation example on simulated data."""
    print("Hello from fracblp!")
    frac_data = simulate_frac_nodemog_data(
        T=50,
        J=20,
    )
    print(frac_data)

    print_stars("Estimating with FRAC")
    _, _ = frac_nodemog_estimate(frac_data)


if __name__ == "__main__":
    run_example()
