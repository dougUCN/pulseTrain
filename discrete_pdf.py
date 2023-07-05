#!/usr/bin/env python
# coding: utf-8

# ## Generate pulsetrain signal from dat file
#
# Each event should have 30 - 40 photons
#
# First create probability distribution from histogram

# In[1]:


import scipy.stats as st
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.pyplot as plt
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'


# In[2]:


get_ipython().run_line_magic("matplotlib", "widget")


# In[3]:


df = pd.read_table("coinc_4200_5441.dat")
df = df.rename(columns={df.columns[0]: "cumulative"})
cdf = df["cumulative"].to_numpy()  # cumulative distribution function
pdf = cdf[1:] - cdf[:-1]  # probability distrubtion function arises from differential
time = np.arange(len(cdf) - 1) * 800e-12  # seconds


# In[8]:


plt.figure()
plt.plot(time[0:2000] * 1e6, pdf[0:2000], label="pdf")
plt.ylim(bottom=0)
plt.xlabel("time (micro-s)")
plt.ylabel("normalized probability over 0.8 ns")
plt.legend()


# In[5]:


# Generate probability distribution from histogram
custom_dist = st.rv_histogram((pdf[0:2000], time[0:2001]))


# In[7]:


# Sanity check
hist, bin_edges = np.histogram(
    custom_dist.rvs(size=100000), range=[0, 1.6e-6], bins=1000
)
centers = (bin_edges[1:] + bin_edges[:-1]) / 2

plt.figure()
plt.plot(centers * 1e6, hist, label="generated")
plt.ylim(bottom=0)
plt.xlabel("time (micro-s)")
plt.ylabel("normalized probability over 0.8 ns")
plt.legend()


# In[ ]:
