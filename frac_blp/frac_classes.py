"""Data containers used by FRAC estimators."""

import numpy as np
from dataclasses import dataclass, field

from bs_python_utils.bsutils import print_stars
from bs_python_utils.bs_sparse_gaussian import setup_sparse_gaussian

from frac_blp.frac_utils import make_X


@dataclass
class FracNoDemogRealData:
    """Container for real FRAC data without demographics."""

    T: int
    J: int
    X1_exo: np.ndarray | None
    X1_endo: np.ndarray | None
    X2_exo: np.ndarray | None
    X2_endo: np.ndarray | None
    Z: np.ndarray
    names_vars_beta: list[str]
    names_vars_sigma: list[str]
    betas: np.ndarray
    sigmas: np.ndarray
    shares: np.ndarray

    X1: np.ndarray = field(init=False)
    X2: np.ndarray = field(init=False)
    n_obs: int = field(init=False)

    def __post_init__(self):
        """Compute derived attributes once inputs are provided."""
        self.n_obs = self.T * self.J
        self.X1 = make_X(self.X1_exo, self.X1_endo)
        self.X2 = make_X(self.X2_exo, self.X2_endo)
    

    def __str__(self) -> str:
        """
        Return a text summary of the simulated dataset.

        Returns:
            str: Multi-line description with key parameters.
        """
        desc = "Simulated Data for FRAC w/o demographics:\n"
        desc += f"  Number of markets (T): {self.T}\n"
        desc += f"  Products per market (J): {self.J}\n"
        desc += (
            f"  Names of variables with fixed coefficients: {self.names_vars_beta}\n"
        )
        desc += (
            f"  Names of variables with random coefficients: {self.names_vars_sigma}\n"
        )
        desc += f"  Betas: {self.betas}\n"
        desc += f"  Sigmas: {self.sigmas}\n"
        return desc


@dataclass
class FracNoDemogSimulatedData:
    """Container for simulated FRAC data without demographics."""

    T: int
    J: int
    X1_exo: np.ndarray | None
    X1_endo: np.ndarray | None
    X2_exo: np.ndarray | None
    X2_endo: np.ndarray | None
    xi_var: np.ndarray
    Z: np.ndarray
    names_vars_beta: list[str]
    names_vars_sigma: list[str]
    betas: np.ndarray
    sigmas: np.ndarray

    shares: np.ndarray = field(init=False)
    X1: np.ndarray = field(init=False)
    X2: np.ndarray = field(init=False)
    n_obs: int = field(init=False)

    def __post_init__(self):
        """Populate derived arrays and simulate shares."""
        self.n_obs = self.T * self.J
        self.X1 = make_X(self.X1_exo, self.X1_endo)
        self.X2 = make_X(self.X2_exo, self.X2_endo)
        self.shares = self.compute_shares()

    def compute_shares(self) -> np.ndarray:
        """
        Simulate market shares via sparse Gaussian quadrature.

        Returns:
            np.ndarray: Simulated shares stacked across all markets.
        """
        T, J = self.T, self.J
        sigmas = self.sigmas
        sigma = sigmas[0]
        n_obs = T * J
        X2 = self.X2
        n_x2 = X2.shape[1]
        mean_utils = self.X1 @ self.betas + self.xi_var.reshape(n_obs)
        shares = np.zeros(n_obs)
        nodes, weights = setup_sparse_gaussian(n_x2, 17)
        nodes_T = nodes.T
        zero_share = np.zeros(self.T)
        for t in range(T):
            this_market = slice(t * J, (t + 1) * J)
            these_mean_utils = mean_utils[this_market]
            this_X2 = X2[this_market, :]

            def shares_random(eps_vals):
                randoms = sigma * np.outer(this_X2, eps_vals)
                randoms = this_X2 @ (nodes_T * sigmas.reshape((-1, 1)))
                random_utils = randoms + these_mean_utils.reshape((-1, 1))
                max_util = np.max(random_utils, axis=0)
                shifted_utils = random_utils - max_util
                exp_utils = np.exp(shifted_utils)
                denom = np.exp(-max_util) + np.sum(exp_utils, axis=0)
                return exp_utils / denom

            shares[this_market] = shares_random(nodes) @ weights
            zero_share[t] = 1.0 - shares[this_market].sum()

        print_stars(
            f"""
                    Data generation completed; the average zero share is {zero_share.mean():.4f}
                    """
        )
        return shares

    def __str__(self) -> str:
        """
        Return a text summary of the simulated dataset.

        Returns:
            str: Multi-line description with key parameters.
        """
        desc = "Simulated Data for FRAC w/o demographics:\n"
        desc += f"  Number of markets (T): {self.T}\n"
        desc += f"  Products per market (J): {self.J}\n"
        desc += (
            f"  Names of variables with fixed coefficients: {self.names_vars_beta}\n"
        )
        desc += (
            f"  Names of variables with random coefficients: {self.names_vars_sigma}\n"
        )
        desc += f"  Betas: {self.betas}\n"
        desc += f"  Sigmas: {self.sigmas}\n"
        return desc


FracNoDemogData = FracNoDemogRealData | FracNoDemogSimulatedData
