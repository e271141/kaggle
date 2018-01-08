from __future__ import division
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree 

LABELS_PATH = '/mnt/filsystem1/code/dsb2017/stage1_labels/stage1_labels.csv'
ORI_PATH = '/mnt/filsystem1/code/dsb2017/code/ZJJ/3d_resnet/'
THD_PATH = ORI_PATH + '3d_preprocess/'
PROB_TRAIN_PATH = THD_PATH + 'prob_0.9_train/'
CANCER_TRAIN_PATH = THD_PATH + 'cancer_0.9_train/'

UNET_INPUT = PROB_TRAIN_PATH
labels_path = '/mnt/filsystem1/code/dsb2017/stage1_labels/stage1_labels.csv'
REGRESSION = THD_PATH + 'regression/'
KD_TREE = REGRESSION + 'kd_tree_select/'

files = os.listdir(UNET_INPUT)

df_train = pd.read_csv(labels_path)

POINT = np.array([1,1,1],dtype='float32')
KNN = 20
#print df_train
ax = plt.figure().add_subplot(111, projection = '3d') 

train_set=np.zeros((1,3),dtype='float32')
train_label=np.zeros((1),dtype='float32')

for filename in files:
  sample = np.load(UNET_INPUT+filename)
  sample = np.asarray(sample, dtype='float32')
  sample = sample.reshape(sample.shape[0], sample.shape[1])
  [name, sss] = filename.split('.')
  index = list(df_train.id).index(name)
  cancer = df_train.cancer[index]
  tree = KDTree(sample, leaf_size=2) 
  if sample.shape[0]>KNN:
    dist, ind = tree.query(POINT, k=KNN)
    sample = sample[ind]

    label_temp = np.zeros((1),dtype='float32')
    if cancer:
      label_temp = np.ones(KNN)
    else:
      label_temp = np.zeros(KNN)
    temp = sample.reshape(sample.shape[1], sample.shape[2])

    train_set = np.concatenate([train_set,temp])
    train_label = np.concatenate([train_label, label_temp])

#   if cancer:
#     ax.scatter(sample[:,0],sample[:,1],sample[:,2], c = 'r', marker = 'o')

#   else:
#     ax.scatter(sample[:,0],sample[:,1],sample[:,2], c = 'b', marker = 'o')
# plt.show() 
np.save('train_set.npy', train_set[1:])
np.save('train_label.npy', train_label[1:])
 


#pos = np.asarray(pos_sample, dtype='float32')
#neg = np.asarray(neg_sample, dtype='float32')
#
#pos = pos.reshape(pos.shape[0], pos.shape[1])
##pos = np.transpose(pos)
##y_pred = KMeans(pos, 2, random_state=0)
#y_pred = KMeans(n_clusters=2, random_state=0).fit_predict(pos)
#
#neg = neg.reshape(neg.shape[0], neg.shape[1])
##pos = np.transpose(pos)
#y_pred_neg = KMeans(n_clusters=2, random_state=0).fit_predict(neg)