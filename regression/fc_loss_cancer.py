import scipy.io as scio
from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import BatchNormalization
from keras.models import Model
import keras.backend as K
import numpy as np
from keras.utils.visualize_util import plot
import pandas as pd
import os

np.random.seed(0)

LABELS_PATH = '/mnt/filsystem1/code/dsb2017/stage1_labels/stage1_labels.csv'
ORI_PATH = '/mnt/filsystem1/code/dsb2017/code/ZJJ/3d_resnet/'
THD_PREPROCESS = ORI_PATH+'3d_preprocess/'
PROB_FOLDER = THD_PREPROCESS + 'prob_0.9_train/'
CANCER_FOLDER = THD_PREPROCESS + 'cancer_0.9_train/'
TEMP_TXT = THD_PREPROCESS + 'temp.txt'

LOAD_PATH = CANCER_FOLDER

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# #config.gpu_options.per_process_gpu_memory_fraction = 0.3
# set_session(tf.Session(config=config))
 
def test_model(active='relu', initial='glorot_uniform'):
    input_vector = Input(shape=(3,))
    
    x = Dense(32, init=initial, activation=active)(input_vector)
    x = Dense(64, init=initial, activation=active)(x)
    x = Dense(128, init=initial, activation=active)(x)
    x = Dense(256, init=initial, activation=active)(x)
    x = Dense(512, init=initial, activation=active)(x)
    x = Dense(1024, init=initial, activation=active)(x)
    x = Dense(512, init=initial, activation=active)(x)
    x = Dense(256, init=initial, activation=active)(x)
    x = Dense(128, init=initial, activation='linear')(x)
    
    
    loss1 = Dense(1, init=initial, activation='sigmoid', name='loss1')(x)
    #loss2 = Dense(1, init=initial, activation=active, name='loss2')(x)
    #loss3 = Dense(1, init=initial, activation=active, name='loss3')(x)
    #loss4 = Dense(1, init=initial, activation=active, name='loss4')(x)
    #loss5 = Dense(1, init=initial, activation=active, name='loss5')(x)
    #loss6 = Dense(1, init=initial, activation=active, name='loss6')(x)
    
    model = Model(input_vector, [loss1])
    return model

    
def reliability(y_true, y_pred):
    par=0.1
    return K.mean(K.lesser(K.abs((y_pred - y_true)/y_true),par))

def preprocess():
    df_train = pd.read_csv(LABELS_PATH)
    train_names = df_train.id
    train_labels = df_train.cancer

    doc = open(TEMP_TXT, 'r')
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

    sample=np.zeros((1,3),dtype='float32')
    label_sample=np.zeros((1),dtype='float32')

    for n in range(len(id_list)):

        res = train_names[train_names==id_list[n]]
        id_name = str(res.index).split('[')[1]
        id_index = id_name.split(']')[0]
        label = train_labels[int(id_index)]
        ground_truth.append(label)

        prob_path = LOAD_PATH+id_list[n]+'.npy'
        if os.path.exists(prob_path):
            temp = np.load(prob_path)
            temp = temp.reshape(temp.shape[0], temp.shape[1])
            sample = np.concatenate([sample,temp])
            if label:
                add = np.ones(3*len(temp))
                sample = np.concatenate([sample,temp])
                sample = np.concatenate([sample,temp])
            else: 
                add = np.zeros(len(temp))

            label_sample = np.concatenate([label_sample, add])

    x = sample[1:]
    y = label_sample[1:]

    from sklearn.utils import shuffle
    x, y = shuffle(x, y, random_state=0)

    return x, y
      
if __name__ == "__main__":

    x, y = preprocess()

    print x.shape,y.shape
    
    model = test_model()
    
    from keras.optimizers import Adam, SGD, RMSprop
    adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop = RMSprop(lr=1e-3, rho=0.9, epsilon=1e-08, decay=0.0)
    
    model.compile(loss={'loss1':'binary_crossentropy'}, #loss_weights={'loss1':1,'loss2':0,'loss3':0,'loss4':0,'loss5':0,'loss6':0},
    optimizer=adam, metrics=[reliability])#'accuracy',
    #model.load_weights('loss1.h5')
    plot(model, to_file='fc.png',show_shapes=True) 
    
    nb_epoch = 100
    batch_size = 256
    
    hist=model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=0.1)
    model.save_weights('fc_loss_cancer.h5', overwrite=True)
