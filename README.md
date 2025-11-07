# frac_blp

<!-- [GitHub last commit](https://img.shields.io/github/last-commit/bsalanie/frac-blp)

[![Release](https://img.shields.io/github/v/release/bsalanie/frac-blp)](https://img.shields.io/github/v/release/bsalanie/frac-blp)

[![Build status](https://img.shields.io/github/actions/workflow/status/bsalanie/frac-blp/main.yml?branch=main)](https://github.com/bsalanie/frac-blp/actions/workflows/main.yml?query=branch%3Amain) <!-- [![codecov](https://codecov.io/gh/bsalanie/frac-blp/branch/main/graph/badge.svg)](https://codecov.io/gh/bsalanie/frac-blp) 

 [![Commit activity](https://img.shields.io/github/commit-activity/m/bsalanie/frac-blp)](https://img.shields.io/github/commit-activity/m/bsalanie/frac-blp) [![License](https://img.shields.io/github/license/bsalanie/frac-blp)](https://img.shields.io/github/license/bsalanie/frac-blp)
-->

**FRAC for macro-BLP (Salanie-Wolak)**.

- **Github repository**: <https://github.com/bsalanie/frac-blp/>
- **Documentation** <https://bsalanie.github.io/frac-blp/>

### Overview
The package estimates a second-order approximation to the macro BLP model with random coefficients using the FRAC method of Salanie and Wolak. 

At this early stage, the package only implements the basic version of the model without demographics. 

The user should be familiar with the macro BLP model (Berry, Levinsohn, and Pakes, 1995). We use very similar notation to that of Conlon and Gortmaker in their `pyblp` package.

The inputs are:

* `T`: the number of markets 
* `J`: the number of products per market

The data must contain `N=T*J` observations, corresponding to all products in all markets.
The observations should be ordered by market, i.e., the first `J` rows correspond to market 1, the next `J` rows to market 2, etc.

The user must provide the following data matrices:

* `Z`: a matrix of instruments, to which the program will add a constant
* `shares`: a `N`-vector of market shares.
* the regressors as a Pandas data frame with `N` rows. The program will construct the following Numpy matrices from the data frame:
  
  * `X1_exo`, the matrix of exogenous variables with fixed coefficients, using the column names in `names_X1_exo`
  * `X1_endo`, the matrix of endogenous variables with fixed coefficients, using the column names in `names_X1_endo`
  * `X2_exo`, the matrix of exogenous variables with fixed coefficients, using the column names in `names_X2_exo`
  * `X1_endo`, the matrix of endogenous variables with fixed coefficients, using the column names in `names_X2_endo`.

`names_X1_exo, names_X1_endo, names_X2_exo, names_X2_endo` are lists of strings provided by the user with the names of the columns in the data frame corresponding to each of these four types of variables.

A constant term will not be added automatically, so if desired the user must include a column of ones in `X1_exo` and/or `X2_exo`.



**`X2_exo` must be a subset of `X1_exo`, and `X2_endo` must be a subset of `X1_endo` (i.e., all variables with random coefficients must also have fixed coefficients). Any of these four matrices can be `None` if there are no variables of that type.**



The outputs are:

* `betas`: the estimates on the variables with fixed coefficients `X1_exo`, `X1_endo` and the mean coefficients on the variables with random coefficients `X2_exo`, `X2_endo`, in that order.
* `sigmas`: the standard deviations of the coefficients on the variables with random coefficients `X2_exo` and `X2_endo`, in that order.



#### entering the data
The data must first be entered into a `FracNodemogRealData` object:
```python
from frac blp.frac classes import FracNodemogRealData

frac data = FracNodemogRealData(T, J,
                            names_X1_exo,
                            names_X1_endo,
                            names_X2_exo,
                            names_X2_endo,
                            df_X1, 
                            Z, 
                            shares,
                            )
```
Then the model can be estimated with:
```python
from frac_blp.frac_nodemog import estimate

betabar, sigmas = frac nodemog estimate(frac data)
```
`frac_demog_estimate` also has optional arguments to define the combinations of instruments and exogeneous variables to be used in the first stage:

* if `degree_Z=d` is provided, then all interactions of the columns of `Z` up to degree `d` will be used as instruments. 
* if `degree_X1=d'` is provided, then all interactions of the columns of `X1_exo` up to degree `d'` will also be used, on their own and multiplied by the interactions of `Z`.

The default has `degree_Z=2` and `degree_X1=2`.

### Release notes

#### 0.3 (November 67, 2025)
Improved interface for data input. Still no demographics.

#### 0.2 (October 27, 2025)
Improved README.

#### 0.1 (October 26, 2025)
First working version, no demographics.
