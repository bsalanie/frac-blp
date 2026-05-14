import numpy as np
import pytest

from frac_blp.frac_utils import make_Z_full


def test_make_Z_full_rejects_negative_degree():
    Z = np.ones((2, 1))
    with pytest.raises(ValueError):
        make_Z_full(Z, degree_Z=-1)


def test_make_Z_full_generates_polynomials_from_Z_only():
    Z = np.array([[2.0], [3.0], [4.0]])
    result = make_Z_full(Z, degree_Z=2)
    expected = np.column_stack((np.ones(3), Z[:, 0], Z[:, 0] ** 2))
    np.testing.assert_allclose(result, expected)


def test_make_Z_full_with_degree_Z_and_degree_X1_of_2():
    Z = np.array([[2.0], [3.0]])
    X1 = np.array([[4.0], [5.0]])
    result = make_Z_full(
        Z,
        X1_exo=X1,
        degree_Z=2,
        degree_X1=2,
    )

    columns = [
        np.ones(2),
        X1[:, 0],
        X1[:, 0] ** 2,
        Z[:, 0],
        Z[:, 0] * X1[:, 0],
        Z[:, 0] * X1[:, 0] ** 2,
        Z[:, 0] ** 2,
        Z[:, 0] ** 2 * X1[:, 0],
        Z[:, 0] ** 2 * X1[:, 0] ** 2,
    ]
    expected = np.column_stack(columns)

    np.testing.assert_allclose(result, expected)


def test_make_Z_full_includes_cross_terms_for_multi_column_Z():
    Z = np.array([[2.0, 5.0], [3.0, 6.0], [4.0, 7.0]])
    result = make_Z_full(Z, degree_Z=2)
    expected = np.column_stack(
        [
            np.ones(3),
            Z[:, 0],
            Z[:, 1],
            Z[:, 0] ** 2,
            Z[:, 0] * Z[:, 1],
            Z[:, 1] ** 2,
        ]
    )
    np.testing.assert_allclose(result, expected)


def test_make_Z_full_includes_cross_terms_for_multi_column_X1():
    Z = np.array([[2.0], [3.0]])
    X1 = np.array([[4.0, 6.0], [5.0, 7.0]])
    result = make_Z_full(Z, X1_exo=X1, degree_Z=1, degree_X1=1)
    expected = np.column_stack(
        [
            np.ones(2),
            X1[:, 0],
            X1[:, 1],
            Z[:, 0],
            Z[:, 0] * X1[:, 0],
            Z[:, 0] * X1[:, 1],
        ]
    )
    np.testing.assert_allclose(result, expected)


def test_make_Z_full_includes_cross_terms_for_multi_column_Z_and_X1():
    Z = np.array([[2.0, 3.0], [4.0, 5.0]])
    X1 = np.array([[6.0, 7.0], [8.0, 9.0]])
    result = make_Z_full(Z, X1_exo=X1, degree_Z=2, degree_X1=2)
    z0, z1 = Z[:, 0], Z[:, 1]
    x0, x1 = X1[:, 0], X1[:, 1]
    expected = np.column_stack(
        [
            # d_z=0
            np.ones(2),
            x0,
            x1,
            x0**2,
            x0 * x1,
            x1**2,
            # d_z=1: z0
            z0,
            z0 * x0,
            z0 * x1,
            z0 * x0**2,
            z0 * x0 * x1,
            z0 * x1**2,
            # d_z=1: z1
            z1,
            z1 * x0,
            z1 * x1,
            z1 * x0**2,
            z1 * x0 * x1,
            z1 * x1**2,
            # d_z=2: z0^2
            z0**2,
            z0**2 * x0,
            z0**2 * x1,
            z0**2 * x0**2,
            z0**2 * x0 * x1,
            z0**2 * x1**2,
            # d_z=2: z0*z1
            z0 * z1,
            z0 * z1 * x0,
            z0 * z1 * x1,
            z0 * z1 * x0**2,
            z0 * z1 * x0 * x1,
            z0 * z1 * x1**2,
            # d_z=2: z1^2
            z1**2,
            z1**2 * x0,
            z1**2 * x1,
            z1**2 * x0**2,
            z1**2 * x0 * x1,
            z1**2 * x1**2,
        ]
    )
    np.testing.assert_allclose(result, expected)


# `make_Z_full()` does not yet implement `X2_exo` and `degree_X2` arguments
@pytest.mark.skip
def test_make_Z_full_builds_cross_terms_with_x1_and_x2():
    Z = np.array([[2.0], [3.0]])
    X1 = np.array([[4.0], [5.0]])
    X2 = np.array([[6.0], [7.0]])

    result = make_Z_full(
        Z,
        X1_exo=X1,
        X2_exo=X2,
        degree_Z=1,
        degree_X1=1,
        degree_X2=2,
    )

    columns = [
        np.ones(2),
        X2[:, 0],
        X2[:, 0] ** 2,
        X1[:, 0],
        X1[:, 0] * X2[:, 0],
        X1[:, 0] * (X2[:, 0] ** 2),
        Z[:, 0],
        Z[:, 0] * X2[:, 0],
        Z[:, 0] * (X2[:, 0] ** 2),
        Z[:, 0] * X1[:, 0],
        Z[:, 0] * X1[:, 0] * X2[:, 0],
        Z[:, 0] * X1[:, 0] * (X2[:, 0] ** 2),
    ]
    expected = np.column_stack(columns)

    np.testing.assert_allclose(result, expected)
