import numpy as np

from frac_blp.artificial_regressors import make_T, make_this_V, make_this_W


def test_make_T_computes_third_order_regressors_by_market():
    X2 = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [2.0, 1.0],
            [4.0, 3.0],
        ]
    )
    shares = np.array([0.2, 0.3, 0.1, 0.4])
    J = 2

    result = make_T(X2, shares, J)

    expected = np.zeros_like(X2)
    for t in range(X2.shape[0] // J):
        this_market = slice(t * J, (t + 1) * J)
        x = X2[this_market, :]
        s = shares[this_market]

        eS_x = x.T @ s
        x_sq = x * x
        eS_x_sq = x_sq.T @ s
        expected[this_market, :] = (
            x_sq / 6.0 - x * eS_x / 2.0 - eS_x_sq / 2.0 + eS_x * eS_x
        ) * x

    np.testing.assert_allclose(result, expected)


def test_make_this_V_computes_fourth_order_excess_kurtosis_regressors():
    X2 = np.array(
        [
            [1.0, 2.0],
            [3.0, 1.0],
            [2.0, 4.0],
        ]
    )
    shares = np.array([0.2, 0.3, 0.1])

    result = make_this_V(X2, shares)

    eS_X2 = X2.T @ shares
    squared_eS_X2 = eS_X2 * eS_X2
    cubed_eS_X2 = squared_eS_X2 * eS_X2
    X2_sq = X2 * X2
    eS_X2_sq = X2_sq.T @ shares
    X2_cube = X2 * X2_sq
    eS_X2_cube = X2_cube.T @ shares
    expected = (
        X2_cube / 24.0
        - X2_sq * eS_X2 / 6.0
        - X2 * eS_X2_sq / 4.0
        + X2 * squared_eS_X2 / 2.0
        + eS_X2 * eS_X2_sq
        - cubed_eS_X2
        - eS_X2_cube / 6.0
    ) * X2

    np.testing.assert_allclose(result, expected)


def test_make_this_W_computes_fourth_order_product_of_variance_regressors():
    X2 = np.array(
        [
            [1.0, 2.0],
            [3.0, 1.0],
            [2.0, 4.0],
        ]
    )
    shares = np.array([0.2, 0.3, 0.1])

    result = make_this_W(X2, shares)

    assert result.shape == (X2.shape[0], X2.shape[1], X2.shape[1])

    n_products, n_x2 = X2.shape

    eS_X2 = np.zeros(n_x2)
    for m in range(n_x2):
        for j in range(n_products):
            eS_X2[m] += shares[j] * X2[j, m]

    eS_X2_X2 = np.zeros((n_x2, n_x2))
    for m in range(n_x2):
        for n in range(n_x2):
            for j in range(n_products):
                eS_X2_X2[m, n] += shares[j] * X2[j, m] * X2[j, n]

    covar_eS = eS_X2_X2 - np.outer(eS_X2, eS_X2)

    expected = np.zeros((n_products, n_x2, n_x2))
    for j in range(n_products):
        for m in range(n_x2):
            for n in range(n_x2):
                expected[j, m, n] = (
                    X2[j, m] * eS_X2[n] + X2[j, n] * eS_X2[m] - X2[j, m] * X2[j, n]
                ) * covar_eS[m, n]

    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(result, np.swapaxes(result, 1, 2))
