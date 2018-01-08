from __future__ import division
import os
import numpy as np
from sklearn.cluster import KMeans

UNET_INPUT = '/mnt/filsystem1/code/dsb2017/code/ZJJ/3d_resnet/3d_preprocess/prob_0.9_train/'
#POS_SAMPLE = UNET_INPUT + '2b861ff187c8ff2977d988f3d8b08d87.npy'
#POS_SAMPLE = UNET_INPUT + '398208da1bcb6a88e11a7314065f13ff.npy'
#POS_SAMPLE = UNET_INPUT + '118be21b7e0c3058b29a524686391c66.npy'

#POS_SAMPLE = UNET_INPUT + '733205c5d0bbf19f5c761e0c023bf9a0.npy'
#POS_SAMPLE = UNET_INPUT + '198d3ff4979a9a89f78ac4b4a0fe0638.npy'
POS_SAMPLE = UNET_INPUT + '2e8bb42ed99b2bd1d9cd3ffaf5129e4c.npy'

NEG_SAMPLE = UNET_INPUT + 'ac366a2168a4d04509693b7e5bcf3cce.npy'
THE = 0.8
sum_shape = 0

pos_sample = np.load(POS_SAMPLE)
neg_sample = np.load(NEG_SAMPLE)

pos = np.asarray(pos_sample, dtype='float32')
neg = np.asarray(neg_sample, dtype='float32')

print len(pos), len(neg)

pos_mean = np.mean(pos_sample)
neg_mean = np.mean(neg_sample)

pos_effect = 0
neg_effect = 0
for n in range(len(pos)):
	if np.sum(pos[n])>THE:
		pos_effect += 1

for n in range(len(neg)):
	if np.sum(neg[n])>THE:
		neg_effect += 1

pos_percent = pos_effect/len(pos)
neg_percent = neg_effect/len(neg)


pos = pos.reshape(pos.shape[0], pos.shape[1])
#pos = np.transpose(pos)
print pos.shape
y_pred = KMeans(pos, 2, random_state=0)
# km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1)
# y_pred = km.fit(pos)
print y_pred

# print 'pos_sample: %.6f, %.6f' %(pos_mean, pos_percent)
# print 'neg_sample: %.6f, %.6f' %(neg_mean, neg_percent)