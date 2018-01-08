from __future__ import division
import os
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

LABELS_PATH = '/mnt/filsystem1/code/dsb2017/stage1_labels/stage1_labels.csv'
ORI_PATH = '/mnt/filsystem1/code/dsb2017/code/ZJJ/3d_resnet/'
THD_PATH = ORI_PATH + '3d_preprocess/'
PROB_TRAIN_PATH = THD_PATH + 'prob_0.9_train/'
CANCER_TRAIN_PATH = THD_PATH + 'cancer_0.9_train/'

NODULE_9_TXT = THD_PATH + 'nodule_0.9_train.txt'
CANCER_9_TXT = THD_PATH + 'cancer_0.9_train.txt'
NODULE_8_TXT = THD_PATH + 'nodule_0.8_train.txt'
CANCER_8_TXT = THD_PATH + 'cancer_0.8_train.txt'
TEMP_TXT = THD_PATH + 'temp.txt'

NODULE_TXT = TEMP_TXT
CANCER_TXT = CANCER_9_TXT
MEAN_THRESHOLD = 0.02
PROB_SUM_THRESHOLD = 0.03

def cross_entrophy(N, truth, predict):

	N_sum = 0
	for n in range(N):
		y = truth[n]; yi = predict[n]
		#print y, yi, id_list[n]
		if yi == 0:
			yi += 0.01
		elif yi == 1:
			yi -= 0.01
		e = y*math.log(yi) + (1-y)*math.log(1-yi)
		N_sum += e

	return -N_sum/N

def prob_sum(prob_list):
	pos_effect = 0
	for n in range(len(prob_list)):
		if np.sum(prob_list[n])>PROB_SUM_THRESHOLD:
			pos_effect += 1

	pos_percent = pos_effect/len(prob_list)
	return pos_percent
	
if __name__=='__main__':

	df_train = pd.read_csv(LABELS_PATH)
	train_names = df_train.id
	train_labels = df_train.cancer

	doc = open(NODULE_TXT, 'r')
	doc_lines = doc.readlines()
	doc.close()
	
	id_list=[]
	prob_list=[]
	ground_truth=[]

	for line in doc_lines:
		temp = line.strip('\n')
		id_temp = temp.split(':')[0]
		prob_temp = temp.split(':')[1]
		id_list.append(id_temp)
		prob_list.append(prob_temp)

	predict=[]

	pos_sample=np.zeros((1,3),dtype='float32')
	neg_sample=np.zeros((1,3),dtype='float32')

	for n in range(len(id_list)):

		res = train_names[train_names==id_list[n]]
		id_name = str(res.index).split('[')[1]
		id_index = id_name.split(']')[0]
		label = train_labels[int(id_index)]
		ground_truth.append(label)

		mean_path = PROB_TRAIN_PATH+id_list[n]+'.npy'
		pre_temp = 0
		if os.path.exists(mean_path):
			temp = np.load(mean_path)
			mean = np.mean(temp)
			prob_per = prob_sum(temp)
			
			temp = temp.reshape(temp.shape[0], temp.shape[1])
			temp = np.asarray(temp, dtype='float32')

			if label:
				pos_sample = np.concatenate([pos_sample,temp])
			else:
				neg_sample = np.concatenate([neg_sample,temp])
			#print mean, id_list[n]
			if mean >= 0.021 and mean <=0.025 and prob_per > 0.02 and prob_per <0.06:
				pre_temp = prob_list[n]
			else:
				pre_temp = 0
		predict.append(pre_temp)

	truth = np.asarray(ground_truth, dtype='float32')
	predict = np.asarray(predict, dtype='float32')
	prob_list = np.asarray(prob_list, dtype='float32')

	cross = cross_entrophy(len(id_list), truth, prob_list)
	cross_mean = cross_entrophy(len(id_list), truth, predict)

	pos = 0; neg = 0
	for n in range(len(id_list)):
		if truth[n]==1:
			pos += 1
		else: neg += 1

	print cross_mean, pos, neg

	
	print 'pos_samples',len(pos_sample)
	print 'neg_samples',len(neg_sample)

	fig = plt.figure()
	ax = plt.axes(projection='3d')
	x = neg_sample[:,0]
	y = neg_sample[:,1]
	z = neg_sample[:,2]
	ax.scatter(x, y, z)
	ax.set_xlabel('XY PROB')
	ax.set_ylabel('XZ PROB')
	ax.set_zlabel('YZ PROB')
	plt.show()



