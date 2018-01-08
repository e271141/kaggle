import numpy as np
import os
import matplotlib.pyplot as plt

ORI_PATH = '/mnt/filsystem1/code/dsb2017/code/ZJJ/3d_resnet/'
THD_PREPROCESS = ORI_PATH+'3d_preprocess/unet_input/'

LOAD = '2eb92d17ca91b393765e8acf069763a6.npy'

unet_input = np.load(THD_PREPROCESS + LOAD)
print unet_input.shape

plt.figure(22)

plt.subplot(221)
plt.imshow(unet_input[1], cmap='gray')

plt.subplot(222)
plt.imshow(unet_input[100], cmap='gray')

plt.subplot(223)
plt.imshow(unet_input[200], cmap='gray')

plt.subplot(224)
plt.imshow(unet_input[300], cmap='gray')

plt.show()