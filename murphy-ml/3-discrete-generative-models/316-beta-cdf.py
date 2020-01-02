import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

mean = 0.15
lower = 0.05
upper = 0.3

a1s = np.arange(2, 10, 0.001)
pms = np.zeros(a1s.shape)
for i, a1 in enumerate(a1s):
    a2 = a1 * (1-mean)/mean
    cdfs = beta.cdf([lower, upper], a1, a2)
    pms[i] = cdfs[1] - cdfs[0]
errors = (pms-0.95)**2
a1 = a1s[np.argmin(errors)]
a2 = a1 * (1-mean)/mean
print(a1, a2, pms[np.argmin(errors)])
plt.plot(a1s, errors)
plt.show()
