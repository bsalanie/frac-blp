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
The inputs are:

* `T`: the number of markets 
* `J`: the number of products per market

The data must contain $N=T \times J$ observations, corresponding to all products in all markets. **The observations should be ordered by market**, i.e., the first `J` rows correspond to market 1, the next `J` rows to market 2, etc.

The user must provide the following data matrices:

* `Z`: an $(N,n_Z)$  matrix of instruments, to which the program will add a constant.
* `shares`: a `N`-vector of market shares.
* the regressors as a Pandas data frame with `N` rows, along with four lists of strings `names_X1_exo, names_X1_endo, names_X2_exo`, and `names_X2_endo`. The program will construct the following Numpy matrices from the data frame:
  
  * `X1_exo`, the matrix of exogenous variables with fixed coefficients, using the column names in `names_X1_exo`
  * `X1_endo`, the matrix of endogenous variables with fixed coefficients, using the column names in `names_X1_endo`
  * `X2_exo`, the matrix of exogenous variables with fixed coefficients, using the column names in `names_X2_exo`
  * `X1_endo`, the matrix of endogenous variables with fixed coefficients, using the column names in `names_X2_endo`.

**`X2_exo` must be a subset of `X1_exo`, and `X2_endo` must be a subset of `X1_endo` (i.e., all variables with random coefficients must also have fixed coefficients). Any of these four matrices can be `None` if there are no variables of that type.**

**A constant term will not be added automatically**. If one is desired, the user must include a column of ones in `X1_exo`,  and in `X2_exo` if the coefficient of the constant is random.



#### outputs
The outputs are:

* `betas`: the estimates on the variables with fixed coefficients `X1_exo`, `X1_endo` and the mean coefficients on the variables with random coefficients `X2_exo`, `X2_endo`, in that order.
* `sigmas`: the standard deviations of the random coefficients on the variables in `X2_exo` and `X2_endo`, in that order.


#### entering the data
The data must first be entered into a `FracNodemogRealData` object:
```python
from frac blp.frac classes import FracNodemogRealData

frac data = FracNodemogRealData(T, J,
                            names_X1_exo,
                            names_X1_endo,
                            names_X2_exo,
                            names_X2_endo,
                            df, 
                            Z, 
                            shares,
                            )
```
Then the model can be estimated with:
```python
from frac_blp.frac_nodemog import estimate

betabar, sigmas = frac nodemog estimate(frac data)
```
#### options
`frac_demog_estimate` also has optional arguments to define the combinations of instruments and exogeneous variables to be used in the first stage:

* if `degree_Z=d` is provided, then all interactions of the $n_Z$ columns of `Z` up to degree `d`, plus the constant,  will be used as instruments.  This gives ${n_Z+d\choose d}$ instruments.
* if `degree_X1=d'` is provided, then all interactions of the $n_{1x}$ columns of `X1_exo` up to degree `d'` will also be used, on their own and multiplied by the interactions of `Z` up to degree `d`. That gives an additional $({n_Z+d\choose d}-1)\times ({n_{1x}+d^\prime\choose d^\prime}-1)+1$ instruments.

The default has `degree_Z=2` and `degree_X1=2`; the total number of instruments then is:
$$
\frac{(n_{1x}+1)(n_{1x}+2)}{2} \times \left(\frac{(n_{Z}+1)(n_{Z}+2)}{2}-1\right).
$$

### Release notes

#### 0.4 (June 1, 2026)
Added higher-order regressors (not used for estimation yet).

#### 0.3 (November 7, 2025)
Improved interface for data input. Still no demographics.

#### 0.2 (October 27, 2025)
Improved README.

#### 0.1 (October 26, 2025)
First working version, no demographics.


### TODO October 18, 2026
#### optimal instruments
For $E(\rho(\theta_0)\vert z)=0$ we use $\hat{E} D^\prime(z) \Omega(z)^{-1}\rho(\theta)=0$ with $D(z)=E(\nabla_\theta\rho(\theta_0)\vert z)$ and $\Omega(z)=E(\rho(\theta_0)\rho(\theta_0)^\prime\vert z)$.

The variance of $\hat\theta$ is $\left(D^\prime\Omega^{-1}D\right)^{-1}$.

#### possible steps
1. regress $y$ on $E(X\vert Z)$ and $E(K\vert Z)$ with $\Omega_1=\text{Id}$, get the residual $\hat{\xi}$
2. then do it again with $\Omega_2=\hat{E}(Z^\prime\hat{\xi}\hat{xi}^\prime Z)$
3. do up to 4th order with $E(X\vert Z)$ and $E(K_l+\sum_m W_{lm} \hat{\sigma}^2_m$ and $E(T_3\vert Z)$ and $E(T_4^\kappa\vert Z)$ and $\Omega_2$. This should give estimates of all 4 moments.
4. update $\Omega$ and the optimal instruments and run again (and perhaps again).