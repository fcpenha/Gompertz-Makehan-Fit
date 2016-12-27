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
import lmfit

# Pyplot configuration
fs = 24
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

prob_death = np.asarray([
                            (pop[j] - pop[j + 1]) / pop[j] for j in range(0, len(pop) - 1)
                            ])

prob_survival = 1. - prob_death


# %%%%%%%%%%%
# %%% FIT %%%
# %%%%%%%%%%%

# Gompertz-Makehan (t=1)


def k_gm(x, b):
    return np.exp(b * x) * (np.exp(b) - 1.) / b


def log_survival_gm(x, a, b, l):
    return -l - a * k_gm(x, b)


model = lmfit.Model(log_survival_gm, independent_vars=['x'])

amin = 0.
amax = 1.

bmin = 0.
bmax = 0.1

lmin = 0.
lmax = 1.

fit_result = model.fit(
    np.log(prob_survival), x=age[:-1],
    a=lmfit.Parameter(value=1.e-6, min=amin, max=amax),
    b=lmfit.Parameter(value=0.085, min=bmin, max=bmax),
    l=lmfit.Parameter(value=1.e-6, min=lmin, max=lmax)
)

# Full report in "lmfit" may be obtained with
print fit_result.fit_report(min_correl=1.e-6)

# Covariance matrix
cov = fit_result.covar


# %%%%%%%%%%%%%%%%%%%%%%
# %%% COMPUTE ERRORS %%%
# %%%%%%%%%%%%%%%%%%%%%%

# Do the error propagation with the full covariance matrix

# %%% Partial Derivatives %%%


def death_gm_partial(x, a, b, l):
    kk = k_gm(x, b)
    pp = np.exp(log_survival_gm(x, a, b, l))

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
                            cov[i, j] * (death_gm_partial(x, **fit_result.values)[i] ** i) *
                            (death_gm_partial(x, **fit_result.values)[j] ** j)
                            for i in range(0, cov.shape[0])]
                        for j in range(0, cov.shape[0])])))


# Vectorize function
vec_error = np.vectorize(error)

# %%%%%%%%%%%%
# %%% PLOT %%%
# %%%%%%%%%%%%

plt.scatter(age[:-1],
            prob_death,
            marker='.', s=20, label='ELT16 life table')

# Best-fit
plt.plot(age[:-1],
         1. - np.exp(log_survival_gm(x=age[:-1], **fit_result.values)),
         color='red', linewidth=3, label=r'Best-fit')

upper_band = 1. - np.exp(log_survival_gm(x=age[:-1], **fit_result.values)) + vec_error(x=age[:-1])
lower_band = 1. - np.exp(log_survival_gm(x=age[:-1], **fit_result.values)) - vec_error(x=age[:-1])

# 68.3 confidence band
plt.fill_between(age[:-1],
                 upper_band,
                 np.maximum(
                     np.asarray([1.e-5 for j in range(0, len(age[:-1]))]),
                     np.asarray(lower_band)
                 ),
                 facecolor='orange', alpha=0.5, linewidth=0.0, label=r'68.3\% confidence')

plt.xlim([1., 109.])
plt.ylim([1.e-5, 1.])

plt.xlabel(r'Age (years)')
plt.ylabel(r'Death rate')

plt.title(r'$\lambda =$' + str(fit_result.values['l']) + '\n' +
          r'$\alpha =$' + str(fit_result.values['a']) + '\n' +
          r'$\beta =$' + str(fit_result.values['b'])
          )

plt.loglog()
plt.legend(loc='upper left', frameon=True, numpoints=1, ncol=1)

plt.tight_layout()
plt.savefig('./plots/plot_pop.pdf')
plt.close()
