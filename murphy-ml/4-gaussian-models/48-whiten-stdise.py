import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# get height/weight data for males
with open('../data/heightWeightData.txt', 'r') as data_file:
    data = np.array([list(map(int, line.split(","))) for line in data_file if line])
mdata = data[data[:,0]==1][:,1:]
fig = plt.figure(figsize=(9,3))

# a. plot raw data
ax = fig.add_subplot(131)
ax.set_title('raw')
plt.scatter(mdata[:,0], mdata[:,1])
plt.scatter(mdata[0,0], mdata[0,1], c='red')
plt.scatter(mdata[1,0], mdata[1,1], c='orange')
plt.xlim([60, 80])
plt.ylim([60, 290])

X, Y = np.meshgrid(np.arange(60, 80, 0.1), np.arange(80, 280, 0.1))
pos = np.empty(X.shape + (2,))
pos[:,:,0] = X
pos[:,:,1] = Y
cov = np.cov(mdata.T)
Z = multivariate_normal.pdf(pos, np.mean(mdata, axis=0), cov)
plt.contour(X, Y, Z, levels=[1e-4])

# b. plot standardised data (mean = 0, s1 = s2 = 1)
mdata_zeroed = mdata - np.mean(mdata, axis=0)
mdata_std = (mdata_zeroed) / np.std(mdata, axis=0)
ax = fig.add_subplot(132)
ax.set_title('standardised')
plt.scatter(mdata_std[:,0], mdata_std[:,1])
plt.scatter(mdata_std[0,0], mdata_std[0,1], c='red')
plt.scatter(mdata_std[1,0], mdata_std[1,1], c='orange')
plt.xlim([-3, 3])
plt.ylim([-3, 3])

X, Y = np.meshgrid(np.arange(-3, 3, 0.1), np.arange(-3, 3, 0.1))
pos = np.empty(X.shape + (2,))
pos[:,:,0] = X
pos[:,:,1] = Y
cov = np.cov(mdata_std.T)
Z = multivariate_normal.pdf(pos, np.mean(mdata_std, axis=0), cov)
plt.contour(X, Y, Z, levels=[1e-2])

# c. plot whitened data (mean = 0, cov = I)
N = mdata.shape[0]
cov = mdata_zeroed.T @ mdata_zeroed / N
l, v = np.linalg.eig(cov)
mdata_white = np.array([v.T@x/np.sqrt(l) for x in mdata_zeroed])
ax = fig.add_subplot(133)
ax.set_title('whitened')
plt.scatter(mdata_white[:,0], mdata_white[:,1])
plt.scatter(mdata_white[0,0], mdata_white[0,1], c='red')
plt.scatter(mdata_white[1,0], mdata_white[1,1], c='orange')
plt.xlim([-3, 3])
plt.ylim([-3, 3])

cov = np.cov(mdata_white.T)
Z = multivariate_normal.pdf(pos, np.mean(mdata_white, axis=0), cov)
plt.contour(X, Y, Z, levels=[1e-2])

plt.savefig('48-whiten-stdise')
plt.show()
