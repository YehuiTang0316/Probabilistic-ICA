"""
Code for Section 2.3 ICA Blind Source Separation.
"""
import numpy as np
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
from scipy import signal

# construct sample wave
np.random.seed(1)
n_samples = 2000
t = np.linspace(0, 10, n_samples)

s1 = np.cos(3 * t)  # cosine wave
s2 = signal.square(2 * np.pi * t)     # square wave
s3 = signal.sawtooth(np.pi * t)  # sawtooth wave

S = np.c_[s1, s2, s3]
S += 0.1 * np.random.normal(size=S.shape)   # add random noise
S /= S.std(axis=0)  # normalize data

# mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])    # mixing matrix
X = np.dot(S, A.T)  # generated blind source

ica = FastICA(n_components=3)
S_est = ica.fit_transform(X)  # estimated source signals

# plot figure
plt.figure()
models = [X, S, S_est]
names = ['Observation(mixed signal)',
         'True Source',
         'ICA Recovered Source']

colors = ['red', 'green', 'blue']

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(3, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.show()


