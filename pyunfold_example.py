###################################################
# Py-Unfold example.
# This script will produce a fake "MC" dataset and a fake "real" dataset, both
# are just different gaussian distributions.  Then a "detector response" is
# added by applying a guassian shift & smear to both data sets.  A detector
# response matrix is formed from the MC "truth" and "reconstructed" datasets.
# That detector response is then used to perform unfolding on the "real reconstructed"
# dataset. Finally the unfolded result is compared to the "real truth" data.
###################################################

from pyunfold import iterative_unfold
from pyunfold.callbacks import Logger

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2)
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['lines.markeredgewidth'] = 2

bins = np.linspace(-3, 4.5, 51)
num_bins = len(bins) - 1

# Simulated data statistics
num_samples = int(1e6)

# Create an MC and an "detector" data set that are different from each other
mc_true_samples = np.random.normal(loc=0.0, scale=1.0, size=num_samples)
data_true_samples = np.random.normal(loc=1.0, scale=0.6, size=num_samples)

# Histogram the truth mc & data distributions
hist_mc_true, _ = np.histogram(mc_true_samples, bins=bins)
hist_data_true, _ = np.histogram(data_true_samples, bins=bins)

# Perform "reconstruction" which just applies a gaussian shift & smear
random_noise = np.random.normal(loc=0.3, scale=0.7, size=num_samples)
random_noise2 = np.random.normal(loc=0.3, scale=0.7, size=num_samples)
mc_observed_samples = mc_true_samples + random_noise
data_observed_samples = data_true_samples + random_noise2


# Histogram the reconstructed distributions
mc_observed, _ = np.histogram(mc_observed_samples, bins=bins)
data_observed, _ = np.histogram(data_observed_samples, bins=bins)

# show the truth & reconstructed distributions
fig, ax = plt.subplots()
ax.step(np.arange(num_bins), hist_mc_true, where='mid', lw=3,
        alpha=0.7, label='MC \'true\' distribution')
ax.step(np.arange(num_bins), mc_observed, where='mid', lw=3,
        alpha=0.7, label='MC reconstructed distribution')
ax.step(np.arange(num_bins), hist_data_true, where='mid', lw=3,
        alpha=0.7, label='Physical \'true\' distribution')
ax.step(np.arange(num_bins), data_observed, where='mid', lw=3,
        alpha=0.7, label='Detector reconstructed distribution')
ax.set(xlabel='X bins', ylabel='Counts')
ax.legend()
plt.show()

# Fake efficiency 
efficiencies = np.ones_like(data_observed, dtype=float)
efficiencies_err = np.full_like(efficiencies, 0.01, dtype=float)

# Create the detector response matrix by correlating the MC truth & reconstructed data
response_hist, _, _ = np.histogram2d(mc_observed_samples, mc_true_samples, bins=bins)
response_hist_err = np.sqrt(response_hist)

fig, ax = plt.subplots()
im = ax.imshow(response_hist, origin='lower')
cbar = plt.colorbar(im, label='Counts')
ax.set(xlabel='Cause bins', ylabel='Effect bins')
plt.show()

# Normalize the detector response matrix
column_sums = response_hist.sum(axis=0)
normalization_factor = efficiencies / column_sums
response = response_hist * normalization_factor
response_err = response_hist_err * normalization_factor

fig, ax = plt.subplots()
im = ax.imshow(response, origin='lower')
cbar = plt.colorbar(im, label='$P(E_i|C_{\mu})$')
ax.set(xlabel='Cause bins', ylabel='Effect bins',
       title='Normalized response matrix')
plt.show()

data_observed_err = np.sqrt(data_observed)

unfolded_results = iterative_unfold(data=data_observed,
                                    data_err=data_observed_err,
                                    response=response,
                                    response_err=response_err,
                                    efficiencies=efficiencies,
                                    efficiencies_err=efficiencies_err,
                                    return_iterations=True,
                                    ts_stopping=0.001,
                                    callbacks=[Logger()])

num_its = max(unfolded_results['num_iterations'].keys())

# Plot the results
fig, ax = plt.subplots()
ax.step(np.arange(num_bins), hist_mc_true, where='mid', lw=3,
        alpha=0.7, label='MC Truth')
ax.step(np.arange(num_bins), mc_observed, where='mid', lw=3,
        alpha=0.7, label='MC Reconstructed')
ax.step(np.arange(num_bins), hist_data_true, where='mid', lw=3,
        alpha=0.7, label='Physical \'true\' distribution')
ax.step(np.arange(num_bins), data_observed, where='mid', lw=3,
        alpha=0.7, label='Detector reconstructed distribution')
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
