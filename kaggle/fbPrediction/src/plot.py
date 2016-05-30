import readData
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

df_train = loadData('training')
df_train["hour"]       = (df_train["time"]%(60*24))//60.
df_train["dayofweek"]  = np.ceil((df_train["time"]%(60*24*7))//(60.*24))
df_train["day"]  = np.ceil((df_train["time"]/(60*24)))
df_train["week"] = np.ceil((df_train["time"]/(60*24*7)))
df_test["hour"]        = (df_test["time"]%(60*24))//60.
df_test["dayofweek"]   = np.ceil((df_test["time"]%(60*24*7))//(60.*24))
df_test["day"]   = np.ceil((df_test["time"]/(60*24)))
df_test["week"]  = np.ceil((df_test["time"]/(60*24*7)))




# Pull out timestamps
times = np.squeeze(df_train.as_matrix(columns=['time']))

n_events = times.size
n_samples = n_events
hist_range = (-100000.0, 100000.0)
n_bins = 100000 # One bin per 2.0 time units
n_loops = 100

# Estimate autocorrelations
# Randomly pull timestamps and subtract from each other. Get histogram of these values to estimate autocorrelation
print('Estimating autocorrelation')
all_autocorrs_global = np.zeros((n_bins, n_loops))
for loop_n in range(n_loops):
  hist_vals, bin_edges = np.histogram(np.random.choice(times, size=n_samples, replace=True) - \
                                      np.random.choice(times, size=n_samples, replace=True), bins=n_bins, range=hist_range)
  all_autocorrs_global[:, loop_n] = hist_vals

# Plot the autocorrelation and fft of autocorrelation
fig, axs = plt.subplots(2,1)

axs[0].plot(bin_edges[:-1], all_autocorrs_global)
axs[0].set_xlim([-20000.0, 20000.0])
axs[0].set_title('Autocorrelation of all timestamps')
axs[0].set_xlabel('time units')

# Plot the fft of the autocorrelation
f, psd = scipy.signal.welch(all_autocorrs_global, nperseg=25000, noverlap=20000, return_onesided=True, axis=0)

# Adjust the X axis to be in time points instead of 1/F
f /= 2.0  # Remember that there is one bin per 4.0 time units
f = 1.0/f # Go back to time points

# Plot the fft of the autocorrelation (The PSD of the timestamps)
axs[1].plot(f, np.log(psd))
axs[1].set_title('Log FFT of global autocorrelations')
axs[1].set_xlabel('time units')
axs[1].set_xlim([0.0, 20000.0])
axs[1].set_xticks(np.arange(0.0, 20000.0, 1000))
axs[1].grid(True)

fig.tight_layout()

plt.show()


