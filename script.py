# For running this code, you may use:
# ipython -i script.py

# %%%%%%%%%%%%%%%%
# %%% PREAMBLE %%%
# %%%%%%%%%%%%%%%%

# Common Python libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

# Data related Python libraries
import pyexcel_xlsx as pe
import json
import ast

# Non-Linear Least-Square Minimization and Curve-Fitting
#  Documentation: https://lmfit.github.io/lmfit-py/index.html
import lmfit

# Pyplot configuration
fs = 14
# plt.rc('text', usetex=True, fontsize=14)
plt.rc('ytick', labelsize=fs)
plt.rc('xtick', labelsize=fs)
plt.rc('axes', labelsize=fs)
plt.rc('legend', fontsize=fs)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

# %%%%%%%%%%%%%
# %%% BEGIN %%%
# %%%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%
# %%% READ DATA %%%
# %%%%%%%%%%%%%%%%%

# Read data from ELT16.xlsx
data = np.genfromtxt('./data/ELT16.csv', delimiter=',').T

age = data[0]

pop = data[1]

# %%%%%%%%%%%%%%%%%%%%
# %%% PROCESS DATA %%%
# %%%%%%%%%%%%%%%%%%%%

hazard = np.asarray([
                        (pop[j] - pop[j + 1]) / pop[j] for j in range(0, len(pop) - 1)
                    ])

# %%%%%%%%%%%
# %%% FIT %%%
# %%%%%%%%%%%

def f(x, p):
    return np.exp(p * x)

def hazard_gm(x, a, b, l):
        return l + a * f(x, b)

par = lmfit.Parameters()

par.add('a', value=1.e-6, min=0., max=1.)
par.add('b', value=1.e-1, min=0., max=1.)
par.add('l', value=1.e-6, min=0., max=1.)

# Neyman chi-squared (statistical error)
#  We take the log of the quantities in the chi-squared,
#  because the data vary over many order of magnitudes
def residual(p):
    return np.asarray([
                        (
                            (
                                hazard_gm(x=age[j], a=p['a'], b=p['b'], l=p['l'])
                                - hazard[j]
                            ) / hazard[j]
                        ) ** 2
                for j in range(0, len(age) - 1)
            ])

age = age[1:]

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

# Min-chi-squared
min_chi2 = sum(residual(fit_result))

# Number of degrees of freedom for 3 parameters
n_degrees = len(residual(fit_result)) - 3


def func_p_value(c, n):
    """

    c : chi-squared value
    n : number of degree of freedom (d.o.f.),
        i.e. number of points subtracted by number of parameters

    Notice: p=0 is considered the worst possible fit and p=1 is
    considered to be the perfect fit. For example,

    In[235]: print(func_p_value(0, 45))
    Out[235]: 1.0

    In[236]: print(func_p_value(100, 45))
    Out[236]: 4.67686463534e-06

    """
    return 1. - stats.chi2.cdf(c, n)


# p-value
p_value = func_p_value(min_chi2, n_degrees)

print('----------------------------')
print(f'p-value ={p_value}')
print('----------------------------')


# %%%%%%%%%%%%%%%%%%%%%%
# %%% COMPUTE ERRORS %%%
# %%%%%%%%%%%%%%%%%%%%%%

# Do the error propagation with the full covariance matrix

# %%% Partial Derivatives %%%


def partial_l(x, a, b, l):

    return 1.


def partial_a(x, a, b, l):

    return f(x, b)


def partial_b(x, a, b, l):

    return a * x * np.exp(b * x)

# %%% Error propagation %%%

def error(x):
    return np.sqrt(
            cov[0, 0] * partial_a(x, **fit_result) * partial_a(x, **fit_result)
            + cov[1, 1] * partial_b(x, **fit_result) * partial_b(x, **fit_result)
            + cov[2, 2] * partial_l(x, **fit_result) * partial_l(x, **fit_result)
            + 2. * cov[0, 1] * partial_a(x, **fit_result) * partial_b(x, **fit_result)
            + 2. * cov[0, 2] * partial_a(x, **fit_result) * partial_l(x, **fit_result)
            + 2. * cov[1, 2] * partial_b(x, **fit_result) * partial_l(x, **fit_result)
    )


# Vectorize function
vec_error = np.vectorize(error)

# %%%%%%%%%%%%
# %%% PLOT %%%
# %%%%%%%%%%%%

# %%%%%%%%%%%%%%%%%%%%%%
# %%% Best fit curve %%%

# Best-fit
plt.plot(age,
         hazard_gm(x=age, a=fit_result['a'], b=fit_result['b'], l=fit_result['l']),
         color='blue', linewidth=1, label=r'Best-fit', zorder=1)

upper_band = hazard_gm(x=age, **fit_result) + 1.96 * vec_error(x=age)
lower_band = hazard_gm(x=age, **fit_result) - 1.96 * vec_error(x=age)

# 95% confidence band
plt.fill_between(age,
                 upper_band,
                 np.maximum(
                     np.asarray([1.e-5 for j in range(0, len(age))]),
                     np.asarray(lower_band)
                 ),
                 facecolor='orange', alpha=0.5, linewidth=0.0, label=r'95% confidence')

# Because we are using log scale, we cannot have a negative lower band
# The trick is to take max("a small number", *)

plt.scatter(age,
            hazard,
            color='black', marker='.', zorder=2, s=30, alpha=0.6, label='ELT16 life table')

plt.xlim([1., 109.])
plt.ylim([8.e-5, 1.])

plt.xlabel(r'Age (years)')
plt.ylabel(r'Death rate')

plt.title(
    rf"$\alpha = {fit_result['a'].value:.3e}".replace("e-", r"\times 10^{-")
    + r"}\,\mathrm{year}^{-1}$" + "\n"
    + rf"$\beta = {fit_result['b'].value:.3e}".replace("e-", r"\times 10^{-")
    + r"}\,\mathrm{year}^{-1}$" + "\n"
    + rf"$\lambda = {fit_result['l'].value:.3e}".replace("e-", r"\times 10^{-")
    + r"}\,\mathrm{year}^{-1}$"
)

plt.legend(loc='upper left', frameon=True, numpoints=1, ncol=1)

plt.yscale('log')
plt.tight_layout()
plt.savefig('./plots/plot_hazard.pdf')
plt.savefig('./plots/plot_hazard.png')
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
                                     limits=((1.e-5, 1.e-4), (0.08, 0.095))
                                     )

plt.contourf(1.e5 * cx, 1.e2 * cy, grid, [0., 0.68, 0.95, 0.99], cmap='inferno')

plt.xlabel(r'$\alpha \times 10^5 \,\, (\mathrm{year}^{-1})$')
plt.colorbar()
plt.ylabel(r'$\beta \times 10^2 \,\, (\mathrm{year}^{-1})$')

plt.tight_layout()
plt.savefig('./plots/plot_contours_alpha_beta.pdf')
plt.savefig('./plots/plot_contours_alpha_beta.png')
plt.close()

# %%% b versus l %%%

cx, cy, grid = lmfit.conf_interval2d(mini,
                                     out2,
                                     'l', 'b',
                                     nx=60, ny=60,
                                     limits=((1.e-6, 1.e-4), (0.08, 0.095))
                                     )

plt.contourf(1.e5 * cx, 1.e2 * cy, grid, [0., 0.68, 0.95, 0.99], cmap='inferno')

plt.xlabel(r'$\lambda \times 10^5 \,\, (\mathrm{year}^{-1})$')
plt.colorbar()
plt.ylabel(r'$\beta \times 10^2 \,\, (\mathrm{year}^{-1})$')

plt.tight_layout()
plt.savefig('./plots/plot_contours_lambda_beta.pdf')
plt.savefig('./plots/plot_contours_lambda_beta.png')
plt.close()

# %%% a versus l %%%

cx, cy, grid = lmfit.conf_interval2d(mini,
                                     out2,
                                     'l', 'a',
                                     nx=60, ny=60,
                                     limits=((1.e-6, 1.e-4), (1.e-5, 1.e-4))
                                     )

plt.contourf(1.e5 * cx, 1.e5 * cy, grid, [0., 0.68, 0.95, 0.99], cmap='inferno')

plt.xlabel(r'$\lambda \times 10^5 \,\, (\mathrm{year}^{-1})$')
plt.colorbar()
plt.ylabel(r'$\alpha \times 10^5 \,\, (\mathrm{year}^{-1})$')

plt.tight_layout()
plt.savefig('./plots/plot_contours_alpha_lambda.pdf')
plt.savefig('./plots/plot_contours_alpha_lambda.png')
plt.close()
