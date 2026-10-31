# frac_blp

[![PyPI](https://img.shields.io/pypi/v/frac-blp)](https://pypi.org/project/frac-blp/)
[![Python versions](https://img.shields.io/pypi/pyversions/frac-blp)](https://pypi.org/project/frac-blp/)
[![CI](https://img.shields.io/github/actions/workflow/status/bsalanie/frac-blp/main.yml?branch=main&label=CI)](https://github.com/bsalanie/frac-blp/actions/workflows/main.yml)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://bsalanie.github.io/frac-blp/)
[![License](https://img.shields.io/github/license/bsalanie/frac-blp)](https://github.com/bsalanie/frac-blp)
[![Last commit](https://img.shields.io/github/last-commit/bsalanie/frac-blp)](https://github.com/bsalanie/frac-blp/commits/main)

**FRAC for macro-BLP (Salanie-Wolak)**.

- **Github repository**: <https://github.com/bsalanie/frac-blp/>
- **Documentation** <https://bsalanie.github.io/frac-blp/>

### Overview
The package estimates a second-order approximation to the macro BLP model with random coefficients using the FRAC method of Salanie and Wolak. 

**At this early stage, the package only implements the basic version of the model without demographics. The random coefficients are assumed to be independently distributed.**

The user should be familiar with the macro BLP model (Berry, Levinsohn, and Pakes, 1995). We use very similar notation to that of Conlon and Gortmaker in their `pyblp` package.

#### inputs
The user must provide the inputs in a TOML configuration file. Here is an example:
```
[data]
source = "real"                   # indicates that it is data from observations
file = "data/simulated_data.csv"   # must be .csv or .parquet
shares = "shares"                 # the name of the column that contains the market shares $S_{jt}$
# the names of the columns that have the instrumments. A constant is added automatically; do not list one
instruments = ["z_1", "z_2", "z_3"]  
# 
X1_exo = ["constant"]
X1_endo = ["x_1", "x_2", "x_3"] 
X2_exo = []                           # must be a subset of X1_exo
X2_endo = ["x_1", "x_2"]              # must be a subset of X1_endo
T = 500                               # number of markets
J = 10                                # number of products on each market

[estimation]
degree_Z = 2                          
# we create instruments from interactions of the `instruments` up to `degree_Z` 
degree_X1 = 2                         #   interacted with powers of `X1_exo` up to `degree_X1`
optimal_instruments = true            #   if `true`, we use the optimal instruments
run_projections = true                #   if `true`, some information about the first stage is printed
results_file = "estimation_results.txt"         # this is the default value.
```
The `file` must have one row per product and market, stacked by market (`N=T * J` rows, where  `T` is the number of markets and `J` is the number of products per market) **The observations should be ordered by market**, i.e., the first `J` rows correspond to market 1, the next `J` rows to market 2, etc. The path of `file` is relative to that of the configuration file.

The names of the variables must correspond to column names in the header of `file`.

The program will construct the following Numpy matrices from the data, using the column names indicated in the 
configuration file:
  * `X1_exo`, the matrix of exogenous variables with fixed coefficients
  * `X1_endo`, the matrix of endogenous variables with fixed coefficients
  * `X2_exo`, the matrix of exogenous variables with fixed coefficients
  * `X1_endo`, the matrix of endogenous variables with fixed coefficients.

**`X2_exo` must be a subset of `X1_exo`, and `X2_endo` must be a subset of `X1_endo` (i.e., all variables with random coefficients must also have fixed coefficients). Any of these four matrices can be `None` if there are no variables of that type.**

**A constant term will not be added automatically**. If one is desired, the user must include a column of ones in `X1_exo`,  and in `X2_exo` if the coefficient of the constant is random.

On the other hand, a constant will be added automatically to the `instruments`. If  `degree_Z=d` is provided, then all interactions of the $n_Z$ columns of `Z` up to degree `d`, plus the constant,  will be used as instruments.  This gives ${n_Z+d\choose d}$ instruments. If `degree_X1=d'` is provided, then all interactions of the $n_{1x}$ columns of `X1_exo` up to degree `d'` will also be used, on their own and multiplied by the interactions of `Z` up to degree `d`. That gives an additional $({n_Z+d\choose d}-1)\times ({n_{1x}+d^\prime\choose d^\prime}-1)+1$ instruments.

The default has `degree_Z=2` and `degree_X1=2`; the total number of instruments then is:
$$
\frac{(n_{1x}+1)(n_{1x}+2)}{2} \times \left(\frac{(n_{Z}+1)(n_{Z}+2)}{2}-1\right).
$$



#### outputs
The outputs are:

* `betas_sigmas2_est`: the estimates of the mean coefficients on the variables `X1_exo` and `X1_endo`, then the estimated variances of the coefficients on `X2_exo` and `X2_endo`, in that order.
* `stderrs_est`: the estimated standard errors of these estimates.
* `xi_est`: the estimated product effects, a vector of size `N`, ordered by market

They are saved in the text file indicated as `results_file` in the configuration file. If no `results_file` is indicated, the results are written in `estimation_results.txt`.  Its path is relative to where the program was run.


### Release notes

#### 0.5 (October 31, 2026)
Improved and accelerated. Uses a TOML configuration file for the parameters.

#### 0.4 (June 1, 2026)
Added higher-order regressors (not used for estimation yet).

#### 0.3 (November 7, 2025)
Improved interface for data input. Still no demographics.

#### 0.2 (October 27, 2025)
Improved README.

#### 0.1 (October 26, 2025)
First working version, no demographics.


### TODO October 31, 2026
1. do up to 4th order with $E(X\vert Z)$ and $E(K_l+\sum_m W_{lm} \hat{\sigma}^2_m$ and $E(T_3\vert Z)$ and $E(T_4^\kappa\vert Z)$ and $\Omega_2$. This should give estimates of all 4 moments.
2. update $\Omega$ and the optimal instruments and run again (and perhaps again).