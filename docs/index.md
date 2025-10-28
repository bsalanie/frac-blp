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

The user should be familiar with the macro BLP model (Berry, Levinsohn, and Pakes, 1995). We use very similar notation to that of Conlon and Gortmaker in their `pyblp` package:

The inputs are:
* `T`: the number of markets 
* `J`: the number of products per market
* `X1`: variables with fixed coefficients, an `(N=T*J, n1)` matrix
* `X2`: variables with random coefficients, an `(N, n2)` matrix
* `Z`: instruments, an `(N, nz)` matrix.

The outputs are:
* `betas`: the coefficients on the variables with fixed coefficients and the mean coefficients on the variables with random coefficients, an `(n1 + n2)` vector
* `sigmas`: the standard deviations of the coefficients on the variables with random coefficients, an `n2` vector.

#### entering the data
The user must provide the data as numpy arrays with `T*J` rows:

* `X1_exo, X1_endo, X2_exo, X2_endo`: matrices of exogenous and endogenous variables with fixed and random coefficients

* `Z`: matrix of instruments

* `shares`: vector of market shares.

The observations should be ordered by market, i.e., the first `J` rows correspond to market 1, the next `J` rows to market 2, etc.

These are entered in the model as follows:
```python
rom frac blp.frac classes import FracNodemogRealData

frac data = FracNodemogRealData(T, J,
                            X1_exo, X1_endo,
                            X2_exo, X2_endo,
                            Z, shares,
                            names_vars_beta,
                            names_vars_sigma)
```
Then the model can be estimated with:
```python
from frac_blp.frac_nodemog import estimate

betabar, sigmas = frac nodemog estimate(frac data)
```


### Release notes

#### 0.1 (October 26, 2025)
First working version, no demographics.
