"""Read FRAC estimation settings and data from a TOML configuration file.

The file has three sections:

- ``[data]``: where the data come from. With ``source = "real"``, the data are
  read from a CSV or Parquet file; with ``source = "simulated"``, they are simulated.
- ``[simulation]``: parameters of the simulation, used when ``source = "simulated"``.
- ``[estimation]``: options for ``frac_nodemog_estimate``.

Relative paths for ``file`` are taken relative to the directory of the config file.
"""

import tomllib
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from frac_blp.frac_classes import FracNoDemogData, FracNoDemogRealData
from frac_blp.simulate_frac_nodemog_data import simulate_frac_nodemog_data


class _StrictModel(BaseModel):
    """Base model that rejects keys it does not define."""

    model_config = ConfigDict(extra="forbid")


class DataConfig(_StrictModel):
    """The ``[data]`` section.

    Attributes:
        source (str): ``"real"`` to read a file, ``"simulated"`` to simulate.
        file (Path | None): Path to a CSV or Parquet file with one row per product
            and market, stacked by market. Needed for ``source = "real"``.
        shares (str | None): Column with the observed shares.
        instruments (list[str]): Columns of the instruments. A constant is added
            automatically, so do not include one.
        X1_exo (list[str]): Columns of the exogenous regressors with fixed coefficients.
        X1_endo (list[str]): Columns of the endogenous regressors with fixed coefficients.
        X2_exo (list[str]): Columns of the exogenous regressors with random
            coefficients. Must be a subset of ``X1_exo``.
        X2_endo (list[str]): Columns of the endogenous regressors with random
            coefficients. Must be a subset of ``X1_endo``.
        T (int | None): Number of markets. Needed for ``source = "real"``.
        J (int | None): Number of products per market. Needed for ``source = "real"``.
    """

    model_config = ConfigDict(extra="forbid")

    source: Literal["real", "simulated"]
    file: Path | None = None
    shares: str | None = None
    instruments: list[str] = []
    X1_exo: list[str] = []
    X1_endo: list[str] = []
    X2_exo: list[str] = []
    X2_endo: list[str] = []
    T: int | None = None
    J: int | None = None

    @model_validator(mode="after")
    def _check_real_fields(self) -> "DataConfig":
        """Check that the fields needed for real data are given and do not clash.

        Returns:
            DataConfig: The validated section, unchanged.

        Raises:
            ValueError: If a field needed for real data is missing, or if a
                regressor is listed in both its exogenous and endogenous lists,
                or an ``X2`` name is not in the matching ``X1`` list.
        """
        if self.source == "real":
            missing = [
                name
                for name in ("file", "shares", "T", "J")
                if getattr(self, name) is None
            ]
            if not self.instruments:
                missing.append("instruments")
            if missing:
                raise ValueError(f"[data] with source = 'real' needs: {missing}")
            both = sorted(set(self.X1_exo) & set(self.X1_endo))
            if both:
                raise ValueError(
                    f"[data] lists {both} in both X1_exo and X1_endo; "
                    "each regressor must be in exactly one of them."
                )
            both_X2 = sorted(set(self.X2_exo) & set(self.X2_endo))
            if both_X2:
                raise ValueError(
                    f"[data] lists {both_X2} in both X2_exo and X2_endo; "
                    "each regressor must be in exactly one of them."
                )
            not_in_exo = [name for name in self.X2_exo if name not in self.X1_exo]
            if not_in_exo:
                raise ValueError(
                    f"[data] X2_exo {not_in_exo} must be in X1_exo {self.X1_exo}."
                )
            not_in_endo = [name for name in self.X2_endo if name not in self.X1_endo]
            if not_in_endo:
                raise ValueError(
                    f"[data] X2_endo {not_in_endo} must be in X1_endo {self.X1_endo}."
                )
        return self


class SimulationConfig(_StrictModel):
    """The ``[simulation]`` section; see :func:`simulate_frac_nodemog_data`."""

    T: int = 20_000
    J: int = 10
    n_X1_exo: int = 1
    n_X1_endo: int = 1
    n_X2_exo: int = 0
    n_X2_endo: int = 1
    sigma_x: float = 1.0
    sigma_xi: float = 1.0
    rho_x_xi: float = float(np.sqrt(0.5))
    rho_x_z: float = float(np.sqrt(0.5))
    betas: list[float] = [-4.3, 1.0]
    sigmas: list[float] = [1.0]
    seed: int | None = None  # None: fresh randomness on each run


class EstimationConfig(_StrictModel):
    """The ``[estimation]`` section; see :func:`frac_nodemog_estimate`."""

    degree_Z: int = 2
    degree_X1: int = 2
    results_file: str = "estimation_results.txt"
    optimal_instruments: bool = True
    run_projections: bool = True


class Config(_StrictModel):
    """The whole configuration file."""

    data: DataConfig
    simulation: SimulationConfig = SimulationConfig()
    estimation: EstimationConfig = EstimationConfig()


def load_config(path: str | Path) -> Config:
    """Read and validate a TOML configuration file.

    Args:
        path (str | Path): Path to the TOML file.

    Returns:
        Config: The validated settings. A relative ``data.file`` is resolved
        against the directory of the config file.

    Raises:
        pydantic.ValidationError: If the file has unknown keys or invalid values.
    """
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    config = Config.model_validate(raw)
    if config.data.file is not None and not config.data.file.is_absolute():
        config.data.file = path.parent / config.data.file
    return config


def _read_table(file: Path) -> pd.DataFrame:
    """Read a CSV or Parquet file into a DataFrame.

    Args:
        file (Path): Path to a ``.csv``, ``.parquet`` or ``.pq`` file.

    Returns:
        pd.DataFrame: The table in the file.

    Raises:
        ValueError: If the file extension is not supported.
    """
    if file.suffix == ".csv":
        return pd.read_csv(file)
    if file.suffix in (".parquet", ".pq"):
        return pd.read_parquet(file)
    raise ValueError(f"Unsupported data file {file}; use a .csv or .parquet file.")


def build_frac_data(config: Config) -> FracNoDemogData:
    """Build the FRAC data object described by a configuration.

    Args:
        config (Config): Settings from :func:`load_config`.

    Returns:
        FracNoDemogData: Simulated data, or real data read from ``data.file``.

    Raises:
        ValueError: If the file lacks some of the named columns or does not have
            ``T * J`` rows.
    """
    if config.data.source == "simulated":
        sim = config.simulation
        return simulate_frac_nodemog_data(
            T=sim.T,
            J=sim.J,
            n_X1_exo=sim.n_X1_exo,
            n_X1_endo=sim.n_X1_endo,
            n_X2_exo=sim.n_X2_exo,
            n_X2_endo=sim.n_X2_endo,
            sigma_x=sim.sigma_x,
            sigma_xi=sim.sigma_xi,
            rho_x_xi=sim.rho_x_xi,
            rho_x_z=sim.rho_x_z,
            betas=np.array(sim.betas),
            sigmas=np.array(sim.sigmas),
            seed=sim.seed,
        )

    data = config.data
    assert data.file is not None and data.shares is not None
    assert data.T is not None and data.J is not None
    df = _read_table(data.file)

    names_X1 = data.X1_exo + data.X1_endo
    needed = [data.shares, *data.instruments, *names_X1]
    missing = [name for name in needed if name not in df.columns]
    if missing:
        raise ValueError(f"Columns missing from {data.file}: {missing}")
    if len(df) != data.T * data.J:
        raise ValueError(
            f"{data.file} has {len(df)} rows, but T * J = {data.T * data.J}."
        )

    frac_real_data = FracNoDemogRealData(
        T=data.T,
        J=data.J,
        df_X1=df[names_X1].copy(),
        Z=df[data.instruments].to_numpy(dtype=np.float64),
        shares=df[data.shares].to_numpy(dtype=np.float64),
        names_X1_exo=data.X1_exo,
        names_X1_endo=data.X1_endo,
        names_X2_exo=data.X2_exo,
        names_X2_endo=data.X2_endo,
    )
    return frac_real_data
