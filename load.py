import numpy as np
import os

LABELS_PATH = '/mnt/filsystem1/code/dsb2017/stage1_labels/stage1_labels.csv'
ORI_PATH = '/mnt/filsystem1/code/dsb2017/code/ZJJ/3d_resnet/'
THD_PREPROCESS = ORI_PATH+'3d_preprocess/'
PROB_FOLDER = THD_PREPROCESS + 'prob_0.9_train/'
CANCER_FOLDER = THD_PREPROCESS + 'cancer_0.9_train/'
TEMP_TXT = THD_PREPROCESS + 'temp.txt'
LOAD_PATH = PROB_FOLDER
RESULT_CSV = THD_PREPROCESS + 'submission.csv'
FILE = THD_PREPROCESS + 'nodule_0.9.txt'
REGRESSION = '/mnt/filsystem1/code/dsb2017/code/ZJJ/3d_resnet/regression/'

if __name__ == "__main__":

    doc = open(FILE, 'r')
    doc_lines = doc.readlines()
    doc.close()

    id_list=[]
    sub_list = []

    for line in doc_lines:
        temp = line.strip('\n')
        id_temp = temp.split(':')[0]
        id_list.append(id_temp)

    sample=np.zeros((1,3),dtype='float32')

    for ids in id_list:
        prob_path = PROB_FOLDER+ids+'.npy'
        if os.path.exists(prob_path):
            print prob_path
            temp = np.load(prob_path)
            temp = temp.reshape(temp.shape[0], temp.shape[1])
            sample = np.concatenate([sample,temp])
            sub_list.append(ids)

    x = np.load(REGRESSION+'train_set.npy')
    y = np.load(REGRESSION+'train_label.npy')
    print x.shape,y.shape

    print sample.shape
