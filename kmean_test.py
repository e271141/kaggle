from __future__ import division
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

UNET_INPUT = '/mnt/filsystem1/code/dsb2017/code/ZJJ/3d_resnet/3d_preprocess/prob_0.9_train/'
#POS_SAMPLE = UNET_INPUT + '2b861ff187c8ff2977d988f3d8b08d87.npy'
POS_SAMPLE = UNET_INPUT + '398208da1bcb6a88e11a7314065f13ff.npy'
#POS_SAMPLE = UNET_INPUT + '118be21b7e0c3058b29a524686391c66.npy'

#POS_SAMPLE = UNET_INPUT + '2e8bb42ed99b2bd1d9cd3ffaf5129e4c.npy'

NEG_SAMPLE = UNET_INPUT + '750680cce371800ea26576b694b32dc8.npy'
THE = 0.8
sum_shape = 0

pos_sample = np.load(POS_SAMPLE)
neg_sample = np.load(NEG_SAMPLE)

pos = np.asarray(pos_sample, dtype='float32')
neg = np.asarray(neg_sample, dtype='float32')

pos = pos.reshape(pos.shape[0], pos.shape[1])
#pos = np.transpose(pos)
print pos.shape
#y_pred = KMeans(pos, 2, random_state=0)
y_pred = KMeans(n_clusters=2, random_state=0).fit_predict(pos)

neg = neg.reshape(neg.shape[0], neg.shape[1])
#pos = np.transpose(pos)
print neg.shape
#y_pred = KMeans(pos, 2, random_state=0)
y_pred_neg = KMeans(n_clusters=2, random_state=0).fit_predict(neg)
# km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
# y_pred = km.fit(pos)
# print y_pred

fig = plt.figure()
ax = plt.axes(projection='3d')
x = np.concatenate([pos[:,0], neg[:,0]])
y = np.concatenate([pos[:,1], neg[:,1]])
z = np.concatenate([pos[:,2], neg[:,2]])


ax.scatter(x, y, z)
ax.set_xlabel('XY PROB')
ax.set_ylabel('XZ PROB')
ax.set_zlabel('YZ PROB')

plt.show()