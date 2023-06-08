#########################################################################
# Same as basic py-unfold example script but adds a non-uniform cut efficiency
# between "truth" and "reconstructed" data. This efficiency is accounted for
# by the unfolding procedure.
#########################################################################
from pyunfold import iterative_unfold
from pyunfold.callbacks import Logger

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['lines.markeredgewidth'] = 2

bins = np.linspace(-3, 3, 51)
num_bins = len(bins) - 1
bin_width = np.diff(bins)[0]

num_samples = int(1e5)
mc_true_samples = np.random.normal(loc=0.0, scale=1.0, size=num_samples)
data_true_samples = np.random.normal(loc=1.0, scale=0.3, size=num_samples)

# Fake cut condition
def cut_probability(x):
    if x < 0:
        return 0.0
    return min(1., x/3.0)

hist_mc_true, _ = np.histogram(mc_true_samples, bins=bins)
norm = max(hist_mc_true)

random_noise = np.random.normal(loc=0.3, scale=0.5, size=num_samples)
mc_observed_samples = mc_true_samples + random_noise

temp = np.array([[x, y] for x, y in zip(mc_observed_samples, mc_true_samples) if np.random.rand() > cut_probability(x)])
mc_observed_samples = temp[:, 0]
mc_true_selected_samples = temp[:, 1]

mc_observed, _ = np.histogram(mc_observed_samples, bins=bins)
hist_mc_selected, _ = np.histogram(mc_true_selected_samples, bins=bins)

eff_vals = [1-cut_probability(x+bin_width/2.) for x in bins[:-1]]

fig, ax = plt.subplots()
ax.step(bins[:-1]+bin_width/2., hist_mc_true/norm, where='mid', lw=3,
        alpha=0.7, label='True distribution')
ax.step(bins[:-1]+bin_width/2., hist_mc_selected/norm, where='mid', lw=3,
        alpha=0.7, label=' True distribution after cuts')
ax.step(bins[:-1]+bin_width/2., mc_observed/norm, where='mid', lw=3,
        alpha=0.7, label='\'reconstructed\' distribution')

ax.plot(bins[:-1]+bin_width/2., eff_vals,  linestyle='--', color='r', label='Efficiency') 

ax.set(xlabel='X bins', ylabel='Counts')
ax.legend()
plt.show()


efficiencies = hist_mc_selected/hist_mc_true
efficiencies_err = 2*efficiencies/np.sqrt(hist_mc_selected)

response_hist, _, _ = np.histogram2d(mc_observed_samples, mc_true_selected_samples, bins=bins)
response_hist_err = np.sqrt(response_hist)
fig, ax = plt.subplots()
im = ax.imshow(response_hist, origin='lower')
cbar = plt.colorbar(im, label='Counts')
ax.set(xlabel='Cause bins', ylabel='Effect bins')
plt.show()

column_sums = response_hist.sum(axis=0)
normalization_factor = efficiencies / column_sums

response = response_hist * normalization_factor
response_err = response_hist_err * normalization_factor

response.sum(axis=0)

fig, ax = plt.subplots()
im = ax.imshow(response, origin='lower')
cbar = plt.colorbar(im, label='$P(E_i|C_{\mu})$')
ax.set(xlabel='Cause bins', ylabel='Effect bins',
       title='Normalized response matrix')
plt.show()

unfolded_results = iterative_unfold(data=mc_observed,
                                    data_err=np.sqrt(mc_observed),
                                    response=response,
                                    response_err=response_err,
                                    efficiencies=efficiencies,
                                    efficiencies_err=efficiencies_err,
                                    return_iterations=True,
                                    ts_stopping=0.01,
                                    callbacks=[Logger()])

num_its = max(unfolded_results['num_iterations'].keys())

fig, ax = plt.subplots()
ax.step(np.arange(num_bins), hist_mc_true, where='mid', lw=3,
        alpha=0.7, label='True distribution')
ax.step(np.arange(num_bins), mc_observed, where='mid', lw=3,
        alpha=0.7, label='Observed distribution')
ax.errorbar(np.arange(num_bins), unfolded_results['unfolded'][num_its],
            yerr=unfolded_results['sys_err'][num_its],
            alpha=0.7,
            elinewidth=3,
            capsize=4,
            ls='None', marker='.', ms=10, 
            label='Unfolded distribution')

ax.set(xlabel='X bins', ylabel='Counts')
plt.legend()
plt.show()
