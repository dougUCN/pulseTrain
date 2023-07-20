#!/usr/bin/env python
"""
Generate pulsetrain signal from dat file
Plots distribution of photons from UCN detection event
"""

import scipy.stats as st
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
import pandas as pd

import sys

from src.globals import get_project_root

ROOT_DIR = get_project_root()

pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_table(str(ROOT_DIR / "in" / "coinc_4200_5441.dat"))
df = df.rename(columns={df.columns[0]: "cumulative"})
cdf = df["cumulative"].to_numpy()  # cumulative distribution function
pdf = cdf[1:] - cdf[:-1]  # probability distribution function arises from differential
time = np.arange(len(cdf) - 1) * 800e-12  # seconds

plt.figure()
plt.plot(time[0:3000] * 1e6, pdf[0:3000], label="pdf")
plt.ylim(bottom=0)
plt.xlabel("time (micro-s)")
plt.ylabel("normalized probability over 0.8 ns")
plt.title("Probability distribution of photons from UCN event", fontsize=10)
plt.legend()
plt.savefig(str(ROOT_DIR / "figures" / "prob_dist.pdf"))
plt.close()

# Generate probability distribution from histogram
custom_dist = st.rv_histogram((pdf[0:2400], time[0:2401]))

# Sanity check
fake_photons = 100000
hist, bin_edges = np.histogram(
    custom_dist.rvs(size=fake_photons), range=[0, 2e-6], bins=2000
)  # Rebin to 1 ns bins
centers = (bin_edges[1:] + bin_edges[:-1]) / 2

plt.figure()
plt.plot(centers * 1e6, hist, label="generated")
plt.ylim(bottom=0)
plt.xlabel("time (micro-s)")
plt.ylabel("events")
plt.legend()
plt.title(f"Generated photons (n={fake_photons})", fontsize=10)
plt.savefig(str(ROOT_DIR / "figures" / "generated_dist.pdf"))
plt.close()
