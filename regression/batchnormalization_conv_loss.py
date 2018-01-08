import scipy.io as scio
from keras.layers import merge, Input
from keras.layers import Dense, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import BatchNormalization
from keras.models import Model
import keras.backend as K
import numpy as np
from keras.utils.visualize_util import plot

np.random.seed(0)

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# set_session(tf.Session(config=config))

LABELS_PATH = '/mnt/filsystem1/code/dsb2017/stage1_labels/stage1_labels.csv'
ORI_PATH = '/mnt/filsystem1/code/dsb2017/code/ZJJ/3d_resnet/'
THD_PREPROCESS = ORI_PATH+'3d_preprocess/'
PROB_FOLDER = THD_PREPROCESS + 'prob_0.9_train/'
CANCER_FOLDER = THD_PREPROCESS + 'cancer_0.9_train/'
TEMP_TXT = THD_PREPROCESS + 'temp.txt'
LOAD_PATH = PROB_FOLDER
    
def test_model(active='relu', initial='glorot_uniform'):
    input_vector = Input(shape=(3,))
    conv1 = Convolution1D(nb_filter=64, filter_length=9, border_mode='same', name='conv1',activation=active)(input_vector)
    conv1 = BatchNormalization(axis=2 )(conv1)
        
    conv2 = Convolution1D(nb_filter=128, filter_length=1, border_mode='same', name='conv2',activation=active)(conv1)
    conv2 = BatchNormalization(axis=2 )(conv2)
    conv3 = Convolution1D(nb_filter=128, filter_length=7, border_mode='same', name='conv3',activation=active)(conv2)
    conv3 = BatchNormalization(axis=2 )(conv3)
    conv4 = Convolution1D(nb_filter=128, filter_length=1, border_mode='same', name='conv4',activation=active)(conv3)
    conv4 = BatchNormalization(axis=2 )(conv4)
    shortcut1 = Convolution1D(nb_filter=128, filter_length=1, border_mode='same', name='shortcut1',activation=active)(conv1)
    shortcut1 = BatchNormalization(axis=2 )(shortcut1)
    x = merge([conv4, shortcut1], mode='sum')
    x = Activation(active)(x)
    #pool1 = MaxPooling1D(pool_length=1, stride=None, border_mode='valid')(x)
    
    conv5 = Convolution1D(nb_filter=256, filter_length=1, border_mode='same', name='conv5',activation=active)(x)
    conv5 = BatchNormalization(axis=2 )(conv5)
    conv6 = Convolution1D(nb_filter=256, filter_length=5, border_mode='same', name='conv6',activation=active)(conv5)
    conv6 = BatchNormalization(axis=2 )(conv6)
    conv7 = Convolution1D(nb_filter=256, filter_length=1, border_mode='same', name='conv7',activation=active)(conv6)
    conv7 = BatchNormalization(axis=2 )(conv7)
    shortcut2 = Convolution1D(nb_filter=256, filter_length=1, border_mode='same', name='shortcut2',activation=active)(x)
    shortcut2 = BatchNormalization(axis=2 )(shortcut2)
    x = merge([conv7, shortcut2], mode='sum')
    x = Activation(active)(x)
    pool2 = MaxPooling1D(pool_length=2, stride=None, border_mode='valid')(x)
    
    conv8 = Convolution1D(nb_filter=512, filter_length=1, border_mode='same', name='conv8',activation=active)(pool2)
    conv8 = BatchNormalization(axis=2 )(conv8)
    conv9 = Convolution1D(nb_filter=512, filter_length=3, border_mode='same', name='conv9',activation=active)(conv8)
    conv9 = BatchNormalization(axis=2 )(conv9)
    conv10 = Convolution1D(nb_filter=512, filter_length=1, border_mode='same', name='conv10',activation=active)(conv9)
    conv10 = BatchNormalization(axis=2 )(conv10)
    shortcut3 = Convolution1D(nb_filter=512, filter_length=1, border_mode='same', name='shortcut3',activation=active)(pool2)
    shortcut3 = BatchNormalization(axis=2 )(shortcut3)
    x = merge([conv10, shortcut3], mode='sum')
    x = Activation(active)(x)
    pool3 = MaxPooling1D(pool_length=2, stride=None, border_mode='valid')(x)
     
    x = Flatten()(pool3)
    x = Dense(512, init=initial, activation='linear')(x)
    
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

def preprocess(data):
    x = np.asarray(data['input_data'])
    y = np.asarray(data['output_data'])
    #from sklearn.utils import shuffle
    #x, y = shuffle(x, y, random_state=0)
    x = x.reshape(x.shape[0],x.shape[1],1)
    y = y.reshape(y.shape[0],y.shape[1],1) 
    train_x = x[:-50000]
    train_y = y[:-50000]
    test_x = x[-50000:]
    test_y = y[-50000:]
    print train_x.shape,train_y.shape,test_x.shape,test_y.shape
    return train_x, train_y, test_x, test_y
      
if __name__ == "__main__":
    
    x, y = preprocess()
    
    model = test_model()
    
    from keras.optimizers import Adam, SGD, RMSprop
    adam = Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    rmsprop = RMSprop(lr=1e-3, rho=0.9, epsilon=1e-08, decay=0.0)
    
    model.compile(loss={'loss1':'binary_crossentropy'}, #loss_weights={'loss1':1,'loss2':0,'loss3':0,'loss4':0,'loss5':0,'loss6':0},
    optimizer=adam, metrics=[reliability])#'accuracy',
    #model.load_weights('batchnormalization_conv_loss1.h5')
    plot(model, to_file='residual_conv.png',show_shapes=True)
 
    nb_epoch = 50
    batch_size = 256

    hist=model.fit(x, y, batch_size=batch_size, nb_epoch=nb_epoch, validation_data=(test_x,[test_y[:,0]]))
    model.save_weights('batchnormalization_conv_loss1.h5', overwrite=True)
