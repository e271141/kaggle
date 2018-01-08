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
PROB_TRAIN_PATH = THD_PATH + 'probability_0.9/'
CANCER_TRAIN_PATH = THD_PATH + 'cancer_0.9_train/'

UNET_INPUT = PROB_TRAIN_PATH
labels_path = '/mnt/filsystem1/code/dsb2017/stage1_labels/stage1_labels.csv'
REGRESSION = ORI_PATH + 'regression/'
KD_TREE = REGRESSION + 'kd_tree_select/'

files = os.listdir(UNET_INPUT)

POINT = np.array([1,1,1],dtype='float32')
KNN = 20
#print df_train

test_set=np.zeros((1,3),dtype='float32')

FILE = THD_PATH + 'nodule_0.9.txt'
doc = open(FILE, 'r')
#doc = open(TEMP_TXT, 'r')
doc_lines = doc.readlines()
doc.close()
id_list=[]

for line in doc_lines:
  temp = line.strip('\n')
  id_temp = temp.split(':')[0]
  id_list.append(id_temp)

for filename in id_list:
  filename = filename + '.npy'
  if not os.path.exists(UNET_INPUT+filename):
    # temp = np.zeros((1,3),dtype='float32')
    # np.save(KD_TREE+filename, temp)
    continue

  sample = np.load(UNET_INPUT+filename)
  sample = np.asarray(sample, dtype='float32')
  sample = sample.reshape(sample.shape[0], sample.shape[1])
  tree = KDTree(sample, leaf_size=2)
  temp = sample 
  if sample.shape[0]>KNN:
    dist, ind = tree.query(POINT, k=KNN)
    sample = sample[ind]
    temp = sample.reshape(sample.shape[1], sample.shape[2])
  np.save(KD_TREE+filename, temp)




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