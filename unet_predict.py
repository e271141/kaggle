from __future__ import division
import skimage.morphology
import cv2
import sys
sys.path.append('/mnt/filsystem1/code/luna/znet/luna16/')
sys.path.append('/mnt/filsystem1/code/luna/znet/luna16/src/')
sys.path.append('/mnt/filsystem1/code/luna/znet/luna16/src/deep/unet/')
sys.path.append('/mnt/filsystem1/code/luna/znet/luna16/src/deep/')
import params
import numpy as np
import os
import skimage.io
import matplotlib.pyplot as plt
import loss_weighting
import normalize

import theano
import theano.tensor as T
import lasagne

import unet
import util
from unet import INPUT_SIZE, OUTPUT_SIZE
from glob import glob

model_folder = '/mnt/filsystem1/code/luna/znet/luna16/models/'

model_folder = os.path.join(model_folder, '1488527877_unet_split01')

#Overwrite params, ugly hack for now
params.params = params.Params(['/mnt/filsystem1/code/luna/znet/luna16/config/default.ini'] + [os.path.join(model_folder, 'config.ini')])
from params import params as P
P.RANDOM_CROP = 0
P.INPUT_SIZE = 512
#P.INPUT_SIZE = 0

def get_image(lung, deterministic):
    lung[lung==-2000] = 0
    #lung = lung - 1024
    truth = np.zeros_like(lung)
    outside = np.where(lung==0,1,0)
    #######################

    outside = np.array(outside, dtype=np.float32)

    truth = np.array(np.round(truth),dtype=np.int64)
    outside = np.array(np.round(outside),dtype=np.int64)

    #Set label of outside pixels to -10
    truth = truth - (outside*10)

    lung = lung*(1-outside)
    lung = lung-outside*3000

    if P.INPUT_SIZE > 0:
        lung = crop_or_pad(lung, INPUT_SIZE, -3000)
        truth = crop_or_pad(truth, OUTPUT_SIZE, 0)
        outside = crop_or_pad(outside, OUTPUT_SIZE, 1)
    else:
        out_size = output_size_for_input(lung.shape[1], P.DEPTH)
        #lung = crop_or_pad(lung, INPUT_SIZE, -1000)
        truth = crop_or_pad(truth, out_size, 0)
        outside = crop_or_pad(outside, out_size, 1)

    lung = normalize.normalize(lung)
    lung = np.expand_dims(np.expand_dims(lung, axis=0),axis=0)

    if P.ZERO_CENTER:
        lung = lung - P.MEAN_PIXEL

    truth = np.array(np.expand_dims(np.expand_dims(truth, axis=0),axis=0),dtype=np.int64)

    return lung, truth

def crop_or_pad(image, desired_size, pad_value):
    if image.shape[0] < desired_size:
        offset = int(np.ceil((desired_size-image.shape[0])/2))
        image = np.pad(image, offset, 'constant', constant_values=pad_value)

    if image.shape[0] > desired_size:
        offset = (image.shape[0]-desired_size)//2
        image = image[offset:offset+desired_size,offset:offset+desired_size]

    return image

def load_images(images, deterministic=True):
    slices = [get_image(image, deterministic) for image in images]
    lungs, truths = zip(*slices)

    l = np.array(np.concatenate(lungs,axis=0), dtype=np.float32)
    t = np.concatenate(truths,axis=0)

    # Weight the loss by class balancing, classes other than 0 and 1
    # get set to 0 (the background is -10)
    w = loss_weighting.weight_by_class_balance(t, classes=[0,1])

    #Set -1 labels back to label 0
    t = np.clip(t, 0, 100000)

    return l, t, w

def load_unet():
    input_var = T.tensor4('inputs')

    print "Defining network"
    net_dict = unet.define_network(input_var)
    network = net_dict['out']

    epoch = '143'
    image_size = OUTPUT_SIZE**2

    model_save_file = os.path.join(model_folder, P.MODEL_ID+"_epoch"+epoch+'.npz')

    print "Loading saved model", model_save_file
    with np.load(model_save_file) as f:
         param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    predict_fn = unet.define_predict(network, input_var)
    return predict_fn


def unet_predict(images,predict_fn):
    # images = [cv2.resize(img,(324,324)) for img in images]

    inputs, _, weights = load_images(images)
    # print inputs.shape
    predictions = [predict_fn(img.reshape(1,1,512,512))[0] for img in inputs]

    saved_predictions = np.empty((len(predictions),1,324,324),dtype='float32')
    for n, img in enumerate(images):
        out_size = unet.output_size_for_input(inputs.shape[3], P.DEPTH)
        image_size = out_size**2
        image = predictions[n][0:image_size][:,1].reshape(out_size,out_size)

        #Remove parts outside a few pixels from the lungs
        image = image * np.where(weights[n,0,:,:]==0,0,1)
        saved_predictions[n] = image

    return saved_predictions


