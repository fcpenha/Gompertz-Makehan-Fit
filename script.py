# For running this code, you may use:
# ipython -i script.py

# %%%%%%%%%%%%%%%%
# %%% PREAMBLE %%%
# %%%%%%%%%%%%%%%%

# Common Python libraries
import numpy as np
from matplotlib import pyplot as plt

# Data related Python libraries
import pyexcel_xlsx as pe
import json
import ast

# Non-Linear Least-Square Minimization and Curve-Fitting
#  Documentation: https://lmfit.github.io/lmfit-py/index.html
import lmfit

# Pyplot configuration
fs = 20
plt.rc('text', usetex=True, fontsize=14)
plt.rc('ytick', labelsize=fs)
plt.rc('xtick', labelsize=fs)
plt.rc('axes', labelsize=fs)
plt.rc('legend', fontsize=fs - 6)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

# %%%%%%%%%%%%%
# %%% BEGIN %%%
# %%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%
# %%% READ DATA %%%
# %%%%%%%%%%%%%%%%%

# Read data from ELT16.xlsx
age = np.asarray([
                     n[0] for n in list(
        ast.literal_eval(
            json.dumps(
                pe.get_data('./data/ELT16.xlsx', start_column=0, column_limit=1)
            )[:-2][101:]
        )
    )][:-5],
                 dtype=float)

pop = np.asarray([
                     n[0] for n in list(
        ast.literal_eval(
            json.dumps(
                pe.get_data('./data/ELT16.xlsx', start_column=2, column_limit=3)
            )[:-2][33:]
        )
    )],
                 dtype=float)

# %%%%%%%%%%%%%%%%%%%%
# %%% PROCESS DATA %%%
# %%%%%%%%%%%%%%%%%%%%

rate_death = np.asarray([
                            (pop[j] - pop[j + 1]) / pop[j] for j in range(0, len(pop) - 1)
                            ])

rate_survival = 1. - rate_death


# %%%%%%%%%%%
# %%% FIT %%%
# %%%%%%%%%%%

# Gompertz-Makehan (t=1)


def k_gm(x, b):
    return np.exp(b * x) * (np.exp(b) - 1.) / b


def survival_gm(x, a, b, l):
    return np.exp(-l - a * k_gm(x, b))


par = lmfit.Parameters()

par.add('a', value=1.e-5, min=1.e-6, max=1.)
par.add('b', value=0.085, min=0., max=1.)
par.add('l', value=1.e-4, min=0., max=1.)


# Neyman chi-squared (statistical error)
def residual(p):
    return np.asarray([
                          (np.log(survival_gm(age[j], p['a'], p['b'], p['l'])) -
                           np.log(rate_survival[j])) ** 2 / np.log(rate_survival[j])
                          for j in range(0, len(age[:-1]))])


# create Minimizer
mini = lmfit.Minimizer(residual, par)

# first solve with Nelder-Mead
out1 = mini.minimize(method='nelder')

# then solve with Levenberg-Marquardt using the
# Nelder-Mead solution as a starting point
out2 = mini.minimize(method='leastsq', params=out1.params)

lmfit.report_fit(out2.params, min_correl=1.e-12)

# 1-sigma limits in "lmfit" may be obtained with
fit_result = out2.params

# Covariance matrix
cov = out2.covar


# %%%%%%%%%%%%%%%%%%%%%%
# %%% COMPUTE ERRORS %%%
# %%%%%%%%%%%%%%%%%%%%%%

# Do the error propagation with the full covariance matrix

# %%% Partial Derivatives %%%


def death_gm_partial(x, a, b, l):
    kk = k_gm(x, b)
    pp = survival_gm(x, a, b, l)

    return (
        - kk * pp,
        - a * pp * (2. * np.exp(b * (x + 1.)) - kk / b),
        - pp
    )


# %%% Error propagation %%%


def error(x):
    return np.sqrt(
        np.sum(
            np.asarray([[
                            cov[i, j] * (death_gm_partial(x, **fit_result)[i] ** i) *
                            (death_gm_partial(x, **fit_result)[j] ** j)
                            for i in range(0, cov.shape[0])]
                        for j in range(0, cov.shape[0])])))


# Vectorize function
vec_error = np.vectorize(error)

# %%%%%%%%%%%%
# %%% PLOT %%%
# %%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%
# %%% Best fit curve %%%

# Best-fit
plt.plot(age[1:-1],
         1. - survival_gm(x=age[1:-1], **fit_result),
         color='red', linewidth=3, label=r'Best-fit', zorder=1)

upper_band = 1. - survival_gm(x=age[1:-1], **fit_result) + vec_error(x=age[1:-1])
lower_band = 1. - survival_gm(x=age[1:-1], **fit_result) - vec_error(x=age[1:-1])

# 68.3 confidence band
plt.fill_between(age[1:-1],
                 upper_band,
                 np.maximum(
                     np.asarray([1.e-5 for j in range(0, len(age[1:-1]))]),
                     np.asarray(lower_band)
                 ),
                 facecolor='orange', alpha=0.5, linewidth=0.0, label=r'68.3\% confidence')

# Because we are using log scale, we cannot have a negative lower band
# The trick is to take max("a small number", *)

plt.scatter(age[1:-1],
            rate_death[1:],
            marker='.', zorder=2, s=30, label='ELT16 life table')

plt.xlim([1., 109.])
plt.ylim([1.e-5, 1.])

plt.xlabel(r'Age (years)')
plt.ylabel(r'Death rate')

plt.title(r'$\lambda =$' + str(fit_result['l'].value) + '\n' +
          r'$\alpha =$' + str(fit_result['a'].value) + '\n' +
          r'$\beta =$' + str(fit_result['b'].value)
          )

plt.legend(loc='upper left', frameon=True, numpoints=1, ncol=1)

plt.loglog()
plt.tight_layout()
plt.savefig('./plots/plot_pop.png')
plt.close()

# %%%%%%%%%%%%%%%%%
# %%% Contours %%%%

ci, trace = lmfit.conf_interval(mini, out2, sigmas=[0.68, 0.95],
                                trace=True, verbose=False)

lmfit.printfuncs.report_ci(ci)

# %%% b versus a %%%

cx, cy, grid = lmfit.conf_interval2d(mini,
                                     out2,
                                     'a', 'b',
                                     nx=60, ny=60,
                                     limits=((5.2e-5, 6.8e-5), (0.0908, 0.0882))
                                     )

plt.contourf(1.e5 * cx, 1.e2 * cy, grid, [0., 0.68, 0.95, 0.99], cmap='inferno')

plt.xlabel(r'$\alpha \times 10^5 \,\, (\mathrm{year}^{-1})$')
plt.colorbar()
plt.ylabel(r'$\beta \times 10^2 \,\, (\mathrm{year}^{-1})$')

plt.tight_layout()
plt.savefig('./plots/plot_contours_alpha_beta.png')
plt.close()

# %%% b versus l %%%

cx, cy, grid = lmfit.conf_interval2d(mini,
                                     out2,
                                     'l', 'b',
                                     nx=60, ny=60,
                                     limits=((6.e-6, 8.e-6), (0.0910, 0.0880))
                                     )

plt.contourf(1.e6 * cx, 1.e2 * cy, grid, [0., 0.68, 0.95, 0.99], cmap='inferno')

plt.xlabel(r'$\lambda \times 10^6 \,\, (\mathrm{year}^{-1})$')
plt.colorbar()
plt.ylabel(r'$\beta \times 10^2 \,\, (\mathrm{year}^{-1})$')

plt.tight_layout()
plt.savefig('./plots/plot_contours_lambda_beta.png')
plt.close()
