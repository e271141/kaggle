from __future__ import division
import sys
import numpy as np
import os
import skimage.io
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
import lasagne
sys.path.append('/mnt/filsystem1/code/luna/znet/luna16/src/deep/resnet')
sys.path.append('/mnt/filsystem1/code/luna/znet/luna16/')
sys.path.append('/mnt/filsystem1/code/luna/znet/luna16/src/')
sys.path.append('/mnt/filsystem1/code/luna/znet/luna16/src/deep/')
import util
from parallel import ParallelBatchIterator
from functools import partial
from tqdm import tqdm
from glob import glob
import resnet
import pandas as pd
import augment
from normalize import normalize
import params

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

OUTPUT_SIZE = 64
model_folder = '/mnt/filsystem1/code/luna/znet/luna16/models/'

model_folder = os.path.join(model_folder, '1488525711_OWN2_resnet56_0')

#Overwrite params, ugly hack for now
params.params = params.Params(['/mnt/filsystem1/code/luna/znet/luna16/config/default.ini'] + [os.path.join(model_folder, 'config.ini')])
from params import params as P

def preprocess(image):
    image = normalize(image)
    if P.ZERO_CENTER:
        image -= P.MEAN_PIXEL
    return image

def load_resnet():
    print "Defining network"

    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    network = resnet.ResNet_FullPre_Wide(input_var, P.DEPTH, P.BRANCHING_FACTOR)

    epoch = '218'

    model_save_file = os.path.join(model_folder, P.MODEL_ID+"_epoch"+epoch+'.npz')

    print "Loading saved model", model_save_file
    with np.load(model_save_file) as f:
         param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    print "Defining updates.."
    train_fn, val_fn, l_r = resnet.define_updates(network, input_var, target_var)
    return val_fn

def resnet_predict(image, val_fn):
    image = preprocess(image).astype('float32')
    image = image.reshape(1,1,OUTPUT_SIZE,OUTPUT_SIZE)
    # target0 = np.array([0],dtype='int32')
    target1 = np.array([1],dtype='int32')
    # err0, l2_loss0, acc0, predictions0, predictions_raw0 = val_fn(image, target0)
    err1, l2_loss1, acc1, predictions1, predictions_raw1 = val_fn(image, target1)
    return predictions_raw1, acc1

# if __name__ == "__main__":   
    # val_fn = load_resnet()

    # in_pattern = '/mnt/filsystem1/code/dsb2017/code/ZJJ/3d_resnet/test/*.npy'
    # filenames = glob(in_pattern)

    # print "Predicting {} patches".format(len(filenames))

    # all_probabilities = []
    # all_filenames = []

    # n_batches = 0
    # err_total = 0
    # acc_total = 0

    # for filename in filenames:
    #     inputs, targets = load_images([filename])
    #     print inputs.shape, targets.shape
    #     for input_sample in inputs:
    #         input_sample = input_sample.reshape(1,1,64,64)
    #         target_sample = np.array([1],dtype='int32')
    #         err, l2_loss, acc, predictions, predictions_raw = val_fn(input_sample, target_sample)
    #         err_total += err
    #         acc_total += acc
    #         print "err: %.6f, l2_loss: %.6f, acc: %.6f, predictions: %.6f, predictions_raw: %.6f" % (err, l2_loss, acc, predictions, predictions_raw)
    #         # if acc>0:
    #         #     print "acc > 0"
    #             # plt.imshow(input_sample.reshape(64,64), cmap=plt.cm.gray)
    #             # plt.show()