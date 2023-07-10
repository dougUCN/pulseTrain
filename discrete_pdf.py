#!/usr/bin/env python
"""
Generate pulsetrain signal from dat file
Each event should have 30 - 40 photons
First create probability distribution from histogram
"""

import scipy.stats as st
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_table("coinc_4200_5441.dat")
df = df.rename(columns={df.columns[0]: "cumulative"})
cdf = df["cumulative"].to_numpy()  # cumulative distribution function
pdf = cdf[1:] - cdf[:-1]  # probability distrubtion function arises from differential
time = np.arange(len(cdf) - 1) * 800e-12  # seconds

plt.figure()
plt.plot(time[0:2000] * 1e6, pdf[0:2000], label="pdf")
plt.ylim(bottom=0)
plt.xlabel("time (micro-s)")
plt.ylabel("normalized probability over 0.8 ns")
plt.legend()
plt.savefig("prob_dist.pdf")
plt.close()

# Generate probability distribution from histogram
custom_dist = st.rv_histogram((pdf[0:2000], time[0:2001]))

# Sanity check
hist, bin_edges = np.histogram(
    custom_dist.rvs(size=100000), range=[0, 1.6e-6], bins=1000
)
centers = (bin_edges[1:] + bin_edges[:-1]) / 2

plt.figure()
plt.plot(centers * 1e6, hist, label="generated")
plt.ylim(bottom=0)
plt.xlabel("time (micro-s)")
plt.ylabel("events")
plt.legend()
plt.savefig("generated_dist.pdf")
plt.close()
