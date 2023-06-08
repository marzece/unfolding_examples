#########################################################################
# Same as basic py-unfold example script but adds a systematic uncertainty.
# Now the detector response "MC" and "real" datasets are slighly different.
# This causes the unfolding to produce an incorrect result, to account for this
# there's a systematic uncertainty added on the detector response smearing
# value.  The smearing value is varied randomly, unfolding re-done with each
# variation, resulting in many different unfolded distributions.  The spread of
# thoses unfolded distributions forms the systematic error bar on the final
# result.
# This script takes a long time to run ~20 minutes. Reduce NSYS to speed up
#########################################################################

from pyunfold import iterative_unfold
from pyunfold.callbacks import Logger
import ROOT

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['lines.markeredgewidth'] = 2

bins = np.linspace(-3, 4, 51)
num_bins = len(bins) - 1

# Number of systematic variations to perform
NSYS = 1000

num_samples = int(5e6)
mc_true_samples = np.random.normal(loc=0.0, scale=1.0, size=num_samples)
data_true_samples = np.random.normal(loc=1.0, scale=0.6, size=num_samples)

hist_mc_true, _ = np.histogram(mc_true_samples, bins=bins)
hist_data_true, _ = np.histogram(data_true_samples, bins=bins)


#
#fig, ax = plt.subplots()
#ax.step(np.arange(num_bins), hist_mc_true, where='mid', lw=3,
#        alpha=0.7, label='MC \'true\' distribution')
#ax.step(np.arange(num_bins), hist_data_true, where='mid', lw=3,
#        alpha=0.7, label='Physical \'true\' distribution')
#ax.set(xlabel='X bins', ylabel='Counts')
#ax.legend()
#plt.show()

# DATA Smearing
random_noise2 = np.random.normal(loc=0.3, scale=0.7, size=num_samples)
data_observed_samples = data_true_samples + random_noise2
data_observed, _ = np.histogram(data_observed_samples, bins=bins)
data_observed_err = np.sqrt(data_observed)


results = []
stash = None
stash2 = None
adjustments = np.concatenate([[0], np.random.normal(loc=0, scale=0.2, size=NSYS)])
the_closest = adjustments[np.argmin(np.abs(adjustments-0.1))]

for adjust in adjustments:
    print(len(results))
    # "MC" SMEARING
    efficiencies_err = None
    efficiencies = None
    response_hist = None
    response_hist_err = None
    normalization_factor = None
    loop_count = 0
    while True:
        random_noise = np.random.normal(loc=0.3, scale=np.abs(0.6+adjust), size=num_samples)
        mc_observed_samples = mc_true_samples + random_noise

        mc_observed, _ = np.histogram(mc_observed_samples, bins=bins)

        efficiencies = np.ones_like(data_observed, dtype=float)

        efficiencies_err = efficiencies/np.sqrt(mc_observed)

        response_hist, _, _ = np.histogram2d(mc_observed_samples, mc_true_samples, bins=bins)
        response_hist_err = np.sqrt(response_hist)

        column_sums = response_hist.sum(axis=0)
        normalization_factor = efficiencies / column_sums
        if not np.any( column_sums == 0):
            break
        print("WOOOPS %0.3f" % adjust)
        loop_count+=1
        if(loop_count > 10):
            adjust = np.random.choice(adjustments)
            loop_count = 0

    response = response_hist * normalization_factor
    response_err = response_hist_err * normalization_factor



    unfolded_results = iterative_unfold(data=data_observed,
                                        data_err=data_observed_err,
                                        response=response,
                                        response_err=response_err,
                                        efficiencies=efficiencies,
                                        efficiencies_err=efficiencies_err,
                                        ts_stopping=0.0001)


    unfolded_dist = unfolded_results['unfolded']
    if(stash is None):
        stash = unfolded_results

    if(adjust == the_closest):
        stash2 = unfolded_dist


    results.append(unfolded_dist)

cv_result = results[0]
results = np.array(results[1:])
diffs = results-cv_result



idxs = np.argsort(np.abs(diffs), axis=0)
lows = np.min(np.take_along_axis(diffs,idxs, axis=0)[:int(NSYS*0.68),:],axis=0)
highs = np.max(np.take_along_axis(diffs,idxs, axis=0)[:int(NSYS*0.68),:],axis=0)

h = ROOT.TH1D("h_unfold", "", num_bins, 0, num_bins)
h_true = ROOT.TH1D("h_true", "", num_bins, 0, num_bins)
gerr = ROOT.TGraphAsymmErrors()
cov = np.cov(results, rowvar=False)

for i, v in enumerate(highs-lows):
    h.SetBinContent(i+1, cv_result[i])
    gerr.SetPoint(i, i+0.5, cv_result[i])
    gerr.SetPointEXhigh(i, 0)
    gerr.SetPointEXlow(i, 0)
    gerr.SetPointEYhigh(i, np.sqrt(cov[i,i])/2.0)
    gerr.SetPointEYlow(i, np.sqrt(cov[i,i])/2.0)
    #h.SetBinError(i+1, np.sqrt(v**2 + stash['sys_err'][i]**2))

for i, v in enumerate(hist_data_true):
    h_true.SetBinContent(i+1, v)
h_true.Draw("sameHist")
h_true.SetLineWidth(2)
h_true.SetLineColor(1)
h.Draw("SAMEHIST")
gerr.Draw("SAMEPE")
h_true.SetStats(0)
h_true.GetXaxis().SetTitle("Bin #")
h_true.GetYaxis().SetTitle("Counts")
h_true.SetLineColor(2)
leg = ROOT.TLegend(0.1, 0.4, 0.4, 0.8)
leg.AddEntry(h_true,"True Distribution", "l")
leg.AddEntry(h,"Unfolded Distribution", "lpe")
leg.Draw("same")
