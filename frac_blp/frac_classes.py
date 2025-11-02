"""Data containers used by FRAC estimators."""

from __future__ import annotations

from textwrap import dedent

import numpy as np

from pydantic import (
    BaseModel,
    ConfigDict,
    PrivateAttr,
    computed_field,
    field_validator,
    model_validator,
)

from bs_python_utils.bsutils import print_stars
from bs_python_utils.bs_sparse_gaussian import setup_sparse_gaussian

from frac_blp.frac_utils import make_X


class _FracNoDemogBase(BaseModel):
    """Shared configuration for FRAC datasets without demographics."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
    )

    T: int
    J: int
    X1_exo: np.ndarray | None
    X1_endo: np.ndarray | None
    X2_exo: np.ndarray | None
    X2_endo: np.ndarray | None
    Z: np.ndarray
    names_vars_beta: list[str]
    names_vars_sigma: list[str]

    _X1_cache: np.ndarray | None = PrivateAttr(default=None)
    _X2_cache: np.ndarray | None = PrivateAttr(default=None)

    @field_validator("T", "J", mode="before")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("T and J must be strictly positive integers.")
        return int(value)

    @field_validator(
        "X1_exo",
        "X1_endo",
        "X2_exo",
        "X2_endo",
        mode="before",
    )
    @classmethod
    def _coerce_optional_matrix(cls, value: np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return None
        return np.asarray(value)

    @field_validator("Z", mode="before")
    @classmethod
    def _coerce_matrix(cls, value: np.ndarray) -> np.ndarray:
        return np.asarray(value)

    @computed_field(return_type=int)  # type: ignore[misc]
    @property
    def n_obs(self) -> int:
        """Total number of product-market observations."""
        return self.T * self.J

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def X1(self) -> np.ndarray:
        """Regressors with fixed coefficients."""
        if self._X1_cache is None:
            self._X1_cache = make_X(self.X1_exo, self.X1_endo)
        return self._X1_cache

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def X2(self) -> np.ndarray:
        """Regressors with random coefficients."""
        if self._X2_cache is None:
            self._X2_cache = make_X(self.X2_exo, self.X2_endo)
        return self._X2_cache

    @model_validator(mode="after")
    def _validate_base_shapes(self) -> "_FracNoDemogBase":
        n_obs = self.n_obs
        for name in ("X1_exo", "X1_endo", "X2_exo", "X2_endo", "Z"):
            arr = getattr(self, name)
            if arr is None:
                continue
            if arr.ndim != 2:
                raise ValueError(f"{name} must be a 2D array.")
            if arr.shape[0] != n_obs:
                raise ValueError(f"{name} must have {n_obs} rows (got {arr.shape[0]}).")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} must contain only finite values.")

        n_x1 = self.X1.shape[1]
        if len(self.names_vars_beta) != n_x1:
            raise ValueError(
                "names_vars_beta length must match the number of columns in X1."
            )

        n_x2 = self.X2.shape[1]
        if len(self.names_vars_sigma) != n_x2:
            raise ValueError(
                "names_vars_sigma length must match the number of columns in X2."
            )

        return self


class FracNoDemogRealData(_FracNoDemogBase):
    """Container for real FRAC data without demographics."""

    shares: np.ndarray

    @field_validator("shares", mode="before")
    @classmethod
    def _coerce_shares(cls, value: np.ndarray) -> np.ndarray:
        shares = np.asarray(value).squeeze()
        return shares

    @model_validator(mode="after")
    def _validate_shares(self) -> "FracNoDemogRealData":
        shares = self.shares
        if shares.ndim != 1:
            raise ValueError("shares must be a 1D array.")
        if shares.shape[0] != self.n_obs:
            raise ValueError(
                f"shares must have length {self.n_obs} (got {shares.shape[0]})."
            )
        if not np.all(np.isfinite(shares)):
            raise ValueError("shares must contain only finite values.")
        if np.any((shares < 0.0) | (shares > 1.0)):
            raise ValueError("shares must lie between 0 and 1.")
        T, J = self.T, self.J
        for t in range(T):
            market_shares = shares[t * J : (t + 1) * J]
            if market_shares.sum() > 1.0:
                raise ValueError(
                    f"Shares in market {t} sum to more than 1 (got {market_shares.sum():.4f})."
                )
        return self

    def __str__(self) -> str:
        """
        Return a text summary of the observed dataset.

        Returns:
            str: Multi-line description with key parameters.
        """
        desc = "Observed Data for FRAC w/o demographics:\n"
        desc += f"  Number of markets (T): {self.T}\n"
        desc += f"  Products per market (J): {self.J}\n"
        desc += (
            f"  Names of variables with fixed coefficients: {self.names_vars_beta}\n"
        )
        desc += (
            f"  Names of variables with random coefficients: {self.names_vars_sigma}\n"
        )
        return desc


class FracNoDemogSimulationParameters(BaseModel):
    """Parameters used in FRAC data simulation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    T: int
    J: int
    n_X1_exo: int = 0
    n_X1_endo: int = 1
    n_X2_exo: int = 0
    n_X2_endo: int = 1
    n_Z: int = 1
    sigma_x: float = 1.0
    sigma_xi: float = 1.0
    rho_xz: float = float(np.sqrt(0.5))
    rho_x_xi: float = float(np.sqrt(0.5))
    betas: np.ndarray = np.array([-4.3, 1.0])
    sigmas: np.ndarray = np.array([1.0])
    names_vars_beta: list[str] = ["constant", "x"]
    names_vars_sigma: list[str] = ["x"]

    @field_validator(
        "T", "J", "n_X1_exo", "n_X1_endo", "n_X2_exo", "n_X2_endo", mode="before"
    )
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("T and J must be strictly positive integers.")
        return int(value)

    @field_validator("betas", "sigmas", mode="before")
    @classmethod
    def _coerce_vector(cls, value: np.ndarray) -> np.ndarray:
        # Coerce to a 1-D array; avoid squeeze creating 0-D for 1x1 inputs.
        return np.asarray(value).ravel()

    @field_validator("sigma_xi")
    @classmethod
    def _validate_sigma_xi(cls, v: float):
        if not np.isfinite(v):
            raise ValueError("sigma_xi must be finite.")
        if v < 0.0:
            raise ValueError("sigma_xi must be non-negative.")
        return float(v)

    @model_validator(mode="after")
    def _validate_betas_length(self) -> "FracNoDemogSimulationParameters":
        expected_n1 = 1 + int(self.n_X1_exo) + int(self.n_X1_endo)
        if self.betas.ndim != 1:
            raise ValueError("betas must be a 1D array.")
        if self.betas.shape[0] != expected_n1:
            raise ValueError(
                f"betas must have length 1 + n_X1_exo + n_X1_endo = {expected_n1} (got {self.betas.shape[0]})."
            )
        return self

    @model_validator(mode="after")
    def _validate_sigmas_length(self) -> "FracNoDemogSimulationParameters":
        expected_n2 = int(self.n_X2_exo) + int(self.n_X2_endo)
        if self.sigmas.ndim != 1:
            raise ValueError("sigmas must be a 1D array.")
        if self.sigmas.shape[0] != expected_n2:
            raise ValueError(
                f"sigmas must have length n_X2_exo + n_X2_endo = {expected_n2} (got {self.sigmas.shape[0]})."
            )
        return self

    @model_validator(mode="after")
    def _validate_dimensions_relationships(self) -> "FracNoDemogSimulationParameters":
        if int(self.n_X2_exo) > int(self.n_X1_exo):
            raise ValueError("n_X2_exo must be less than or equal to n_X1_exo.")
        if int(self.n_X2_endo) > int(self.n_X1_endo):
            raise ValueError("n_X2_endo must be less than or equal to n_X1_endo.")
        if int(self.n_Z) < int(self.n_X1_endo):
            raise ValueError("n_Z must be greater than or equal to n_X1_endo.")
        return self

    @field_validator("sigmas")
    @classmethod
    def _validate_sigmas_nonneg(cls, v: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(v)):
            raise ValueError("sigmas must contain only finite values.")
        if np.any(v < 0.0):
            raise ValueError("all components of sigmas must be non-negative.")
        return v

    @field_validator("rho_xz", "rho_x_xi")
    @classmethod
    def _validate_rho(cls, v: float, info):
        if not np.isfinite(v):
            raise ValueError(f"{info.field_name} must be finite.")
        if v < -1.0 or v > 1.0:
            raise ValueError(f"{info.field_name} must be between -1 and 1.")
        return float(v)


class FracNoDemogSimulatedData(_FracNoDemogBase):
    """Container for simulated FRAC data without demographics."""

    xi_var: np.ndarray
    betas: np.ndarray
    sigmas: np.ndarray

    _shares_cache: np.ndarray | None = PrivateAttr(default=None)

    @field_validator("xi_var", "betas", "sigmas", mode="before")
    @classmethod
    def _coerce_vector(cls, value: np.ndarray) -> np.ndarray:
        # Coerce to a 1-D array; avoid squeeze creating 0-D for 1x1 inputs.
        return np.asarray(value).ravel()

    @model_validator(mode="after")
    def _validate_simulated(self) -> "FracNoDemogSimulatedData":
        n_obs = self.n_obs
        # print(f"{n_obs=}, {self.betas=}, {self.sigmas=}")

        if self.xi_var.ndim != 1 or self.xi_var.shape[0] != n_obs:
            raise ValueError(f"xi_var must be a 1D array of length {n_obs}.")

        if not np.all(np.isfinite(self.xi_var)):
            raise ValueError("xi_var must contain only finite values.")

        n_x1 = self.X1.shape[1]
        if self.betas.ndim != 1 or self.betas.shape[0] != n_x1:
            raise ValueError(
                "betas must be a 1D array whose length matches X1's columns."
            )

        n_x2 = self.X2.shape[1]
        if self.sigmas.ndim != 1 or self.sigmas.shape[0] != n_x2:
            # print(f"{self.sigmas.shape=}, {n_x2=}")
            raise ValueError(
                "sigmas must be a 1D array whose length matches X2's columns."
            )

        if np.any(self.sigmas < 0.0):
            raise ValueError("sigmas must be non-negative.")

        if not np.all(np.isfinite(self.betas)):
            raise ValueError("betas must contain only finite values.")

        if not np.all(np.isfinite(self.sigmas)):
            raise ValueError("sigmas must contain only finite values.")

        return self

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def shares(self) -> np.ndarray:
        """Simulated market shares using sparse Gaussian quadrature."""
        if self._shares_cache is None:
            self._shares_cache = self.compute_shares()
        return self._shares_cache

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
            dedent(
                f"""
                    Data generation completed; the average zero share is {zero_share.mean():.4f}
                    """
            )
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
