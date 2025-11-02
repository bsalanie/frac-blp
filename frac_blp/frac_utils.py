"""Utility helpers for building instruments and projections in FRAC."""

import numpy as np
from scipy import linalg as spla


def make_X(X_exo: np.ndarray | None, X_endo: np.ndarray | None) -> np.ndarray:
    """
    Combine exogenous and endogenous regressors into a single regressor matrix.

    Args:
        X_exo (np.ndarray | None): Exogenous regressors.
        X_endo (np.ndarray | None): Endogenous regressors.

    Returns:
        np.ndarray: The concatenated regressor matrix.

    Raises:
        ValueError: If both `X_exo` and `X_endo` are ``None``.
    """
    if X_exo is not None and X_endo is not None:
        X = np.column_stack((X_exo, X_endo))
    elif X_exo is not None:
        X = X_exo
    elif X_endo is not None:
        X = X_endo
    else:
        raise ValueError("At least one of X_exo or X_endo must be provided.")
    return X


def make_Z_full(
    Z: np.ndarray,
    X1_exo: np.ndarray | None = None,
    degree_Z: int = 2,
    degree_X1: int = 2,
) -> np.ndarray:
    """
    Build a full set of polynomial instruments for FRAC without demographics.

    Args:
        Z (np.ndarray): Baseline instruments.
        X1_exo (np.ndarray | None): Exogenous regressors without random coefficients.
        degree_Z (int): Maximum degree applied to columns of ``Z`` (must be >= 0).
        degree_X1 (int): Maximum degree applied to columns of ``X1_exo``.

    Returns:
        np.ndarray: Instrument matrix whose columns enumerate every admissible
        combination of polynomial terms, preceded by a column of ones.

    Raises:
        ValueError: If ``degree_Z`` is negative.
    """
    if degree_Z < 0:
        raise ValueError("degree_Z must be non-negative.")

    n_obs, n_z = Z.shape
    columns: list[np.ndarray] = [np.ones(n_obs)]

    n_x1 = 0 if X1_exo is None else X1_exo.shape[1]
    max_dx1 = degree_X1 if X1_exo is not None else 0

    for d_z in range(degree_Z + 1):
        z_indices = [None] if d_z == 0 else range(n_z)
        for iz in z_indices:
            base_z = np.ones(n_obs) if iz is None else Z[:, iz] ** d_z
            # str_base_z = " " if iz is None else f"Z[:, {iz}] ** {d_z} "

            for d_x1 in range(max_dx1 + 1):
                x1_indices = [None] if d_x1 == 0 else range(n_x1)
                for ix1 in x1_indices:
                    if ix1 is None:
                        term_x1 = np.ones(n_obs)
                        # str_X1 = "1 "
                    else:
                        assert X1_exo is not None
                        term_x1 = X1_exo[:, ix1] ** d_x1
                        # str_X1 = f"X1_exo[:, {ix1}] ** {d_x1} "
                    columns.append(base_z * term_x1)
                    # print(f"{iz=}, {d_z=}, {ix1=}, {d_x1=}")
                    # print("    " + str_base_z + " * " + str_X1)

    return np.column_stack(columns)


def proj_Z_full(
    X: np.ndarray, Z_full: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project each column of ``X`` onto ``Z_full`` and report fitted values.

    Args:
        X (np.ndarray): Variables to project.
        Z_full (np.ndarray): Instrument matrix used for the projections.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: ``X_proj`` with projected columns,
        ``coef`` with least-squares coefficients, and ``r2`` with column-wise
        R-squared values.
    """
    EPS = 1e-12
    n_x = X.shape[1]
    X_proj = np.empty(X.shape)
    coef = np.empty((Z_full.shape[1], n_x))
    r2 = np.empty(n_x)
    for ix in range(n_x):
        x = X[:, ix]
        coef_x = spla.lstsq(Z_full, x)[0]
        x_proj = Z_full @ coef_x
        X_proj[:, ix] = x_proj
        coef[:, ix] = coef_x
        var_x = np.var(x)
        r2[ix] = 1.0 if var_x < EPS else np.var(x_proj) / var_x
    return X_proj, coef, r2
