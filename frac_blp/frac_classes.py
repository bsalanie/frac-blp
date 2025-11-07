"""FRAC data containers and simulation utilities (no demographics).

This module defines Pydantic models used to hold inputs and outputs for FRAC
estimators without demographics, along with a parameter container for data
simulation. Arrays are expected in stacked long format by market and product.

Conventions:
    - ``T``: number of markets
    - ``J``: number of products per market
    - ``n_obs = T * J``: total product–market observations
    - ``X``: DataFrame with all regressors, shape ``(n_obs, n_X)``\
    -  we use ``names_X1_exo, names_X1_endo, names_X2_exo, names_X2_endo`` to distinguish the variables of each type in ``X``
    - ``X1``: regressors with fixed coefficients, shape ``(n_obs, n_X1)``
    - ``X2``: regressors with random coefficients, shape ``(n_obs, n_X2)``; must be a subset of ``X1``
    - ``Z``: instrument matrix, shape ``(n_obs, L)``
"""

from __future__ import annotations
from typing import cast

import numpy as np
import pandas as pd

from pydantic import (
    BaseModel,
    ConfigDict,
    computed_field,
    field_validator,
    model_validator,
    PrivateAttr,
    ValidationError,
)
from textwrap import dedent

from bs_python_utils.bsutils import print_stars
from bs_python_utils.bs_sparse_gaussian import setup_sparse_gaussian

from frac_blp.frac_utils import make_X


class _FracNoDemogBase(BaseModel):
    """Shared configuration for FRAC datasets without demographics.

    This internal base model validates array shapes and provides cached
    properties for ``X1`` and ``X2`` built from their exogenous/endogenous
    components via ``make_X``.

    Attributes:
        T (int): Number of markets (must be > 0).
        J (int): Number of products per market (must be > 0).
        names_X1_exo (list[str]): names of exogenous regressors with fixed
            coefficients
        names_X1_endo (list[str]): names of endogenous regressors with fixed
            coefficients
        names_X2_exo (list[str]): names of exogenous regressors with random
            coefficients
        names_X2_endo (list[str]): names of endogenous regressors with random
            coefficients

    Properties:
        n_obs (int): Total observations, equal to ``T * J``.
        names_X1 (list[str]): Concatenated list of names of fixed-coefficient regressors
        names_X2 (list[str]): Concatenated list of names of random-coefficient regressors
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
        validate_assignment=True,
    )

    T: int
    J: int
    names_X1_exo: list[str]
    names_X1_endo: list[str]
    names_X2_exo: list[str]
    names_X2_endo: list[str]

    @field_validator(
        "names_X1_exo",
        "names_X1_endo",
        "names_X2_exo",
        "names_X2_endo",
        mode="before",
    )
    @classmethod
    def _coerce_names_lists(cls, value):
        # Accept None/tuples/ndarrays; ensure concrete list[str]
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list):
            raise TypeError("names_* fields must be lists of strings.")
        for i, item in enumerate(value):
            if not isinstance(item, str):
                value[i] = str(item)
        return value

    @field_validator("T", "J", mode="before")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("T and J must be strictly positive integers.")
        return int(value)

    @computed_field(return_type=int)  # type: ignore[misc]
    @property
    def n_obs(self) -> int:
        """Total number of product-market observations.

        Returns:
            int: ``T * J``.
        """
        return self.T * self.J

    @model_validator(mode="after")
    def _validate_names_subsets(self) -> "_FracNoDemogBase":
        # names_X2_exo must be a sublist of names_X1_exo
        missing_exo = [n for n in self.names_X2_exo if n not in self.names_X1_exo]
        if missing_exo:
            raise ValueError(
                f"names_X2_exo must be a sublist of names_X1_exo; missing {missing_exo}."
            )
        # names_X2_endo must be a sublist of names_X1_endo
        missing_endo = [n for n in self.names_X2_endo if n not in self.names_X1_endo]
        if missing_endo:
            raise ValueError(
                f"names_X2_endo must be a sublist of names_X1_endo; missing {missing_endo}."
            )
        # At least one list of names must be non-empty
        if (
            not self.names_X1_exo
            and not self.names_X1_endo
            and not self.names_X2_exo
            and not self.names_X2_endo
        ):
            raise ValueError(
                "At least one of names_X1_exo, names_X1_endo, names_X2_exo, names_X2_endo must be non-empty."
            )
        return self

    @computed_field(return_type=list[str])  # type: ignore[misc]
    @property
    def names_X1(self) -> list[str]:
        """Names of all regressors with fixed coefficients.

        Returns:
            list[str]: list formed by
            concatenating ``names_X1_exo`` and ``names_X1_endo`` (in that order).
        """
        return self.names_X1_exo + self.names_X1_endo

    @computed_field(return_type=list[str])  # type: ignore[misc]
    @property
    def names_X2(self) -> list[str]:
        """Names of all regressors with random coefficients.

        Returns:
            list[str]: list formed by
            concatenating ``names_X2_exo`` and ``names_X2_endo`` (in that order).
        """
        return self.names_X2_exo + self.names_X2_endo

    def names_len(self, name_str: str) -> int:
        names = getattr(self, name_str, None)
        return 0 if not names else len(names)

    # Counts of names lists (robust to None/empty)
    @computed_field(return_type=int)  # type: ignore[misc]
    @property
    def n_X1_exo(self) -> int:
        return self.names_len("names_X1_exo")

    @computed_field(return_type=int)  # type: ignore[misc]
    @property
    def n_X1_endo(self) -> int:
        return self.names_len("names_X1_endo")

    @computed_field(return_type=int)  # type: ignore[misc]
    @property
    def n_X2_exo(self) -> int:
        return self.names_len("names_X2_exo")

    @computed_field(return_type=int)  # type: ignore[misc]
    @property
    def n_X2_endo(self) -> int:
        return self.names_len("names_X2_endo")

    @computed_field(return_type=int)  # type: ignore[misc]
    @property
    def n_X1(self) -> int:
        return cast(int, self.n_X1_exo + self.n_X1_endo)

    @computed_field(return_type=int)  # type: ignore[misc]
    @property
    def n_X2(self) -> int:
        return cast(int, self.n_X2_exo + self.n_X2_endo)


class FracNoDemogRealData(_FracNoDemogBase):
    """Container for observed FRAC data without demographics.

    Inherits validation and properties from :class:`_FracNoDemogBase` and adds
    the observed market shares vector.

    Attributes:
        df_X1 (pd.DataFrame): DataFrame with regressors ``X1`` only
        Z (np.ndarray): Instrument matrix, shape ``(n_obs, n_Z)``; we add the constant
        shares (np.ndarray): Observed product shares stacked by market, shape
            ``(n_obs,)`` with values in ``[0, 1]`` and per-market sums
            not exceeding ``1``.
    """

    df_X1: pd.DataFrame
    Z: np.ndarray
    shares: np.ndarray

    # Computed design matrices picked from X by name groups

    def get_X_by_names(self, names_str) -> np.ndarray:
        cols = getattr(self, names_str, []) or []
        if len(cols) == 0:
            return np.empty((self.n_obs, 0))
        return cast(np.ndarray, self.df_X1.loc[:, cols].to_numpy().astype(np.float64))

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def X1_exo(self) -> np.ndarray:
        return cast(np.ndarray, self.get_X_by_names("names_X1_exo"))

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def X1_endo(self) -> np.ndarray:
        return cast(np.ndarray, self.get_X_by_names("names_X1_endo"))

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def X2_exo(self) -> np.ndarray:
        return cast(np.ndarray, self.get_X_by_names("names_X2_exo"))

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def X2_endo(self) -> np.ndarray:
        return cast(np.ndarray, self.get_X_by_names("names_X2_endo"))

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def X1(self) -> np.ndarray:
        return make_X(self.X1_exo, self.X1_endo)

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def X2(self) -> np.ndarray:
        return make_X(self.X2_exo, self.X2_endo)

    @field_validator("df_X1", mode="before")
    @classmethod
    def _check_df1(cls, value):
        if isinstance(value, pd.DataFrame):
            return value
        else:
            raise ValidationError("df_X1 must be a pandas DataFrame or array-like.")

    @field_validator("Z", mode="before")
    @classmethod
    def _coerce_Z(cls, value):
        return np.asarray(value)

    @field_validator("shares", mode="before")
    @classmethod
    def _coerce_shares(cls, value: np.ndarray) -> np.ndarray:
        shares = np.asarray(value).ravel()
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
        """Summarize the observed dataset.

        Returns:
            str: Multi-line description with key parameters.
        """
        desc = "Observed Data for FRAC w/o demographics:\n"
        desc += f"  Number of markets (T): {self.T}\n"
        desc += f"  Products per market (J): {self.J}\n"
        desc += f"  Names of exogeneous variables with fixed coefficients: {self.names_X1_exo}\n"
        desc += f"  Names of endogeneous variables with fixed coefficients: {self.names_X1_endo}\n"
        desc += f"  Names of exogeneous variables with random coefficients: {self.names_X2_exo}\n"
        desc += f"  Names of endogeneous variables with random coefficients: {self.names_X2_endo}\n"
        return desc


class FracNoDemogSimulatedData(_FracNoDemogBase):
    """Container for simulated FRAC data without demographics.

    
    The regressors are generated as follows:

    * for `X1_exo`: a constant, then `n_X1_exo`-1 variables N(0, 1) iid
    * for `X1_endo`:
    $$
    X_endo &= \\sigma_x(\\rho_{xz} Z \\
        &+ \\sqrt{1 -\\rho_z ^ 2} \\
        &(\\rho_{x\\xi} \\xi / \\sigma_{\\xi}  +   N(0, 1-\\rho_{x\\xi}^2)))
    $$
    where the $Z$ are iid standard normal and $\\xi$ is $N(0, \\sigma_{\\xi}^2)$.

    Attributes:
        n_Z (int): Number of instruments apart from the constant
        sigma_x (float): Std. dev. used for regressor generation.
        sigma_xi (float): Std. dev. of unobserved product quality ``xi``.
        rho_x_z (float): Correlation between ``X`` and ``Z`` (in ``[-1, 1]``).
        rho_x_xi (float): Correlation between ``X`` and ``xi`` (in ``[-1, 1]``).
        betas (np.ndarray): Fixed coefficients used for simulation, length ``n_X1``.
        sigmas (np.ndarray): Std. devs of random coefficients, length ``n_X2`` and
            element-wise nonnegative..
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Caches to ensure one-time generation per instance
    _X_cache: pd.DataFrame | None = PrivateAttr(default=None)
    _Z_cache: np.ndarray | None = PrivateAttr(default=None)
    _xi_var_cache: np.ndarray | None = PrivateAttr(default=None)
    _shares_cache: np.ndarray | None = PrivateAttr(default=None)

    n_Z: int = 1
    sigma_x: float = 1.0
    sigma_xi: float = 1.0
    rho_x_z: float = float(np.sqrt(0.5))
    rho_x_xi: float = float(np.sqrt(0.5))
    betas: np.ndarray = np.array([-4.3, 1.0])
    sigmas: np.ndarray = np.array([1.0])

    @field_validator("T", "J", mode="before")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("T and J must be strictly positive integers.")
        return int(value)

    @field_validator("betas", "sigmas", mode="before")
    @classmethod
    def _coerce_vectors(cls, value: np.ndarray) -> np.ndarray:
        # Coerce to a 1-D array; avoid squeeze creating 0-D for 1x1 inputs.
        return np.asarray(value).ravel()

    @field_validator("sigma_xi", "sigma_x")
    @classmethod
    def _validate_sigma_xi(cls, v: float):
        if not np.isfinite(v):
            raise ValueError("sigma_x and sigma_xi must be finite.")
        if v < 0.0:
            raise ValueError("sigma_x and sigma_xi must be non-negative.")
        return float(v)

    @model_validator(mode="after")
    def _validate_betas_length(self) -> "FracNoDemogSimulatedData":
        if self.betas.ndim != 1:
            raise ValueError("betas must be a 1D array.")
        if self.betas.shape[0] != self.n_X1:
            raise ValueError(
                f"betas must have length  n_X1 = {self.n_X1} (got {self.betas.shape[0]})."
            )
        return self

    @model_validator(mode="after")
    def _validate_sigmas_length(self) -> "FracNoDemogSimulatedData":
        expected_n2 = int(self.n_X2_exo) + int(self.n_X2_endo)
        if self.sigmas.ndim != 1:
            raise ValueError("sigmas must be a 1D array.")
        if self.sigmas.shape[0] != expected_n2:
            raise ValueError(
                f"sigmas must have length n_X2_exo + n_X2_endo = {expected_n2} (got {self.sigmas.shape[0]})."
            )
        return self

    @field_validator("sigmas")
    @classmethod
    def _validate_vector_nonneg(cls, v: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(v)):
            raise ValueError("sigmas must contain only finite values.")
        if np.any(v < 0.0):
            raise ValueError("all components of sigmas must be non-negative.")
        return v

    @field_validator("rho_x_z", "rho_x_xi")
    @classmethod
    def _validate_rho(cls, v: float, info):
        if not np.isfinite(v):
            raise ValueError(f"{info.field_name} must be finite.")
        if v < -1.0 or v > 1.0:
            raise ValueError(f"{info.field_name} must be between -1 and 1.")
        return float(v)

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def xi_var(self) -> np.ndarray:
        """Realizations of the product effects ``xi``, shape ``(n_obs,)``.

        Returns:
            np.ndarray: Realizations of ``xi``.
        """
        if self._xi_var_cache is not None:
            return self._xi_var_cache
        rng = np.random.default_rng(seed=None)
        xi_vals = cast(np.ndarray, rng.normal(0.0, self.sigma_xi, size=self.n_obs))
        self._xi_var_cache = xi_vals
        return self._xi_var_cache

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def Z(self) -> np.ndarray:
        """Realizations of the instruments ``Z``, shape ``(n_obs, n_Z)``.
        They are a constant plus ``n_Z`` iid N(0, 1) variables.

        Returns:
            np.ndarray: Realizations of ``Z``.
        """
        if self._Z_cache is not None:
            return self._Z_cache
        rng = np.random.default_rng(seed=None)
        Z_vals = rng.normal(0.0, 1.0, size=(self.n_obs, self.n_Z))
        Z_full = np.column_stack((np.ones(self.n_obs), Z_vals))
        self._Z_cache = Z_full
        return self._Z_cache

    @computed_field(return_type=pd.DataFrame)  # type: ignore[misc]
    @property
    def X(self) -> pd.DataFrame:
        """Create the DataFrame `X` with all regressors.

        Returns:
            pd.DataFrame: DataFrame with all regressors.
        """
        if self._X_cache is not None:
            return self._X_cache

        rng = np.random.default_rng(seed=None)
        n_obs, sigma_x, sigma_xi = self.n_obs, self.sigma_x, self.sigma_xi
        xi_var, rho_x_z, rho_x_xi = self.xi_var, self.rho_x_z, self.rho_x_xi
        n_X1_exo, n_X1_endo = self.n_X1_exo, self.n_X1_endo
        X1_exo = np.column_stack(
            (np.ones(n_obs), rng.normal(0, sigma_x, size=(n_obs, n_X1_exo - 1)))
        )
        root1 = float(np.sqrt(1.0 - rho_x_z**2))
        root2 = float(np.sqrt(1.0 - rho_x_xi**2))
        X1_endo = np.empty((n_obs, n_X1_endo))
        for i in range(n_X1_endo):
            X1_endo[:, i] = sigma_x * (
                rho_x_z * self.Z[:, i + 1]  # skip the constant instrument
                + root1
                * (
                    rho_x_xi * xi_var / sigma_xi
                    + root2 * rng.normal(0.0, 1.0, size=n_obs)
                )
            )
        X_mat = make_X(X1_exo, X1_endo)
        df_X = pd.DataFrame(X_mat, columns=self.names_X1)
        print(df_X.head())
        self._X_cache = df_X
        return self._X_cache

    # Computed design matrices generated according to simulation parameters
    def get_X1_piece(self, start: int, n: int) -> np.ndarray:
        n_obs = self.n_obs
        if n == 0:
            return np.empty((n_obs, 0)).astype(np.float64)
        else:
            return cast(
                np.ndarray, self.X.to_numpy()[:, start : (start + n)].astype(np.float64)
            )

    def get_X2_piece(self, names: list[str]) -> np.ndarray:
        if len(names) == 0:
            return np.empty((self.n_obs, 0))
        cols = self.X[names].values.astype(np.float64)
        return cast(np.ndarray, cols)

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def X1_exo(self) -> np.ndarray:
        return self.get_X1_piece(0, self.n_X1_exo)

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def X1_endo(self) -> np.ndarray:
        return self.get_X1_piece(self.n_X1_exo, self.n_X1_endo)

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def X2_exo(self) -> np.ndarray:
        return self.get_X2_piece(self.names_X2_exo)

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def X2_endo(self) -> np.ndarray:
        return self.get_X2_piece(self.names_X2_endo)

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def X1(self) -> np.ndarray:
        return make_X(self.X1_exo, self.X1_endo)

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def X2(self) -> np.ndarray:
        return make_X(self.X2_exo, self.X2_endo)

    @computed_field(return_type=np.ndarray)  # type: ignore[misc]
    @property
    def shares(self) -> np.ndarray:
        """Simulated market shares using sparse Gaussian quadrature.

        Returns:
            np.ndarray: Simulated shares stacked across markets, shape ``(n_obs,)``.
        """
        if self._shares_cache is not None:
            return self._shares_cache
        shares_vals = self.compute_shares()
        self._shares_cache = shares_vals
        return self._shares_cache

    def compute_shares(self) -> np.ndarray:
        """Compute simulated market shares via sparse Gaussian quadrature.

        The routine integrates over the distribution of random coefficients
        using a sparse Gaussian grid. For each market, it computes choice
        probabilities conditional on nodes and averages them with quadrature
        weights.

        Returns:
            np.ndarray: Simulated shares stacked across all markets with shape
            ``(n_obs,)``.
        """
        T, J = self.T, self.J
        sigmas = self.sigmas
        X1 = self.X1
        X2 = self.X2
        n_X2 = X2.shape[1]
        n_obs = T * J
        mean_utils = X1 @ self.betas + self.xi_var.reshape(n_obs)
        shares = np.zeros(n_obs)
        # Handle the degenerate case n_X2 == 0 (no random coefficients):
        if n_X2 == 0:
            zero_share = np.zeros(self.T)
            for t in range(T):
                this_market = slice(t * J, (t + 1) * J)
                these_mean_utils = mean_utils[this_market]
                max_util = np.max(these_mean_utils)
                shifted = these_mean_utils - max_util
                exp_utils = np.exp(shifted)
                denom = np.exp(-max_util) + np.sum(exp_utils)
                shares[this_market] = exp_utils / denom
                zero_share[t] = 1.0 - shares[this_market].sum()
        else:
            nodes, weights = setup_sparse_gaussian(n_X2, 17)
            nodes_T = nodes.T  # shape (n_X2, n_nodes)
            weighted_nodes = nodes_T * sigmas.reshape((-1, 1))  # shape (n_X2, n_nodes)
            zero_share = np.zeros(self.T)
            for t in range(T):
                this_market = slice(t * J, (t + 1) * J)
                these_mean_utils = mean_utils[this_market]
                this_X2 = X2[this_market, :]  # (J, n_X2)

                randoms = this_X2 @ weighted_nodes  # (J, n_nodes)
                random_utils = randoms + these_mean_utils.reshape((-1, 1))
                max_util = np.max(random_utils, axis=0).astype(np.float64)
                shifted_utils = (random_utils - max_util).astype(np.float64)
                exp_utils = np.exp(shifted_utils)
                denom = np.exp(-max_util) + np.sum(exp_utils, axis=0)
                # breakpoint()
                shares[this_market] = exp_utils @ (weights / denom)
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
        """Summarize the simulated dataset.

        Returns:
            str: Multi-line description with key Data.
        """
        desc = "Simulated Data for FRAC w/o demographics:\n"
        desc += f"  Number of markets (T): {self.T}\n"
        desc += f"  Products per market (J): {self.J}\n"
        desc += f"  Names of exogeneous variables with fixed coefficients: {self.names_X1_exo}\n"
        desc += f"  Names of endogeneous variables with fixed coefficients: {self.names_X1_endo}\n"
        desc += f"  Names of exogeneous variables with random coefficients: {self.names_X2_exo}\n"
        desc += f"  Names of endogeneous variables with random coefficients: {self.names_X2_endo}\n"
        desc += f"  Betas: {self.betas}\n"
        desc += f"  Sigmas: {self.sigmas}\n"
        return desc


FracNoDemogData = FracNoDemogRealData | FracNoDemogSimulatedData
