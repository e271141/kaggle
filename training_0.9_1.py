import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import SimpleITK as sitk

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from unet_predict import unet_predict, load_unet, crop_or_pad
from resnet_predict import resnet_predict, load_resnet
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import label,regionprops, perimeter
from scipy import ndimage

os.environ["CUDA_VISIBLE_DEVICES"]="3"

PREDICTIONS = ['id','cancer']

INPUT_FOLDER = '/mnt/filsystem1/code/dsb2017/stage1_data/stage1/'
labels_path = '/mnt/filsystem1/code/dsb2017/stage1_labels/stage1_labels.csv'
ORI_PATH = '/mnt/filsystem1/code/dsb2017/code/ZJJ/3d_resnet/'
THD_PREPROCESS = ORI_PATH+'3d_preprocess/'
UNET_INPUT = THD_PREPROCESS +'unet_input_train/'
RESNET_PADDING = THD_PREPROCESS + 'resnet_padding_train/'
PROB_FOLDER = THD_PREPROCESS + 'prob_0.9_train/'
CANCER_FOLDER = THD_PREPROCESS + 'cancer_0.9_train/'
ACCURACY_FOLDER = THD_PREPROCESS + 'accuracy_0.9_train/'
CANCER_TXT = THD_PREPROCESS + 'cancer_0.9_train_2.txt'
NODULE_TXT = THD_PREPROCESS + 'nodule_0.9_train_2.txt'
ACCURACY_TXT = THD_PREPROCESS + 'accuracy_0.9_train.txt'
RESULT1_CSV = THD_PREPROCESS + 'result1_0.8.csv'
RESULT_CSV = THD_PREPROCESS + 'result_0.8.csv'
TEMP = THD_PREPROCESS + 'temp/'


# OUTPUT_SPACING in mm
OUTPUT_SPACING = [0.5, 0.5, 0.5]
# Output image with have shape [3,OUTPUT_DIM,OUTPUT_DIM]
OUTPUT_DIM = 64
#max area portion when segment lung 
LUNG_PORTION = 0.15
# UNET predition threshold
THRESHOLD = 0.9

MIN_BOUND = -1000.0
MAX_BOUND = 400.0
PIXEL_MEAN = 0.25

def world_2_voxel(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def voxel_2_world(voxelCoord, origin, spacing):
    stretchedVoxelCoord = voxelCoord * spacing
    worldCoord = stretchedVoxelCoord + origin
    return worldCoord

def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs((slices[0].ImagePositionPatient[2] - slices[2].ImagePositionPatient[2])/2)
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
    
    for s in slices:
        s.SliceThickness = slice_thickness
        # print s.ImagePositionPatient
        
    return slices

if __name__=='__main__':
    
    offset = OUTPUT_DIM/2

    predict_fn = load_unet()
    val_fn = load_resnet()

    result = []   #  max of all
    result1 = []    #  max of acc==1

    for u_input in os.listdir(UNET_INPUT):

        if os.path.exists(TEMP+u_input):

            continue

        name = u_input.split('.')[0]
        unet_input = np.load(UNET_INPUT+u_input)
        print RESNET_PADDING+u_input
        resnet_padding = np.load(RESNET_PADDING + u_input)

        maxAllowedArea = LUNG_PORTION*324*324
        nodule_centers = []   # without padding

        np.save(TEMP+u_input, 'null')

        first_patient = load_scan(INPUT_FOLDER + u_input.split('.')[0])
        origin = first_patient[0].ImagePositionPatient

        for slice_index, image in enumerate(unet_input):

            thresh_img = np.where(image<0, 1.0, 0.)
            regions = regionprops(thresh_img.astype(int))
            
            for region in regions:

                if region.area > maxAllowedArea:
                    prediction = unet_predict([image],predict_fn)
                    prediction[prediction<THRESHOLD] = 0
                    prediction[prediction>0] = 1
                    prediction = prediction.reshape(324,324)

                    # localize the center of nodule
                    if np.amax(prediction)>0:
                        centers = []
                        selem = morphology.disk(1)
                        image_eroded = morphology.binary_erosion(prediction,selem=selem)

                        label_im, nb_labels = ndimage.label(image_eroded)
                        for i in xrange(1,nb_labels+1):
                            blob_i = np.where(label_im==i,1,0)
                            mass = center_of_mass(blob_i)
                            centers.append([mass[1],mass[0]])

                        for center in centers:
                            world_coords = voxel_2_world(np.asarray([center[0],center[1],slice_index]),np.asarray(origin),np.asarray([1,1,1]))
                            resnet_coords = world_2_voxel(np.asarray(world_coords),np.asarray(origin),np.asarray(OUTPUT_SPACING))
                            resnet_coords = np.floor(resnet_coords).astype('int16')
                            nodule_centers.append(resnet_coords)
                    continue

        probability = []
        cancer = []
        accuracy = []
        for i, coords in enumerate(nodule_centers):
            # print coords
            coords = coords + offset
            # print coords
            # Create xy, xz, yz
            if (coords[0]+offset)>=resnet_padding.shape[0] or (coords[1]+offset)>=resnet_padding.shape[1] or (coords[2]+offset)>=resnet_padding.shape[2]:
                continue
            xy_slice = np.transpose(resnet_padding[coords[0]-offset:coords[0]+offset,coords[1]-offset:coords[1]+offset,coords[2]])
            xz_slice = np.rot90(resnet_padding[coords[0]-offset:coords[0]+offset,coords[1],coords[2]-offset:coords[2]+offset])
            yz_slice = np.rot90(resnet_padding[coords[0],coords[1]-offset:coords[1]+offset,coords[2]-offset:coords[2]+offset])
            
            # Create output
            if ((xy_slice.shape == (OUTPUT_DIM, OUTPUT_DIM)) and (xz_slice.shape == (OUTPUT_DIM, OUTPUT_DIM)) and (yz_slice.shape == (OUTPUT_DIM, OUTPUT_DIM))):
                xy_prob, xy_acc = resnet_predict(xy_slice,val_fn)
                xz_prob, xz_acc = resnet_predict(xz_slice,val_fn)
                yz_prob, yz_acc = resnet_predict(yz_slice,val_fn)
                #print "xy_prob: %.6f, xz_prob: %.6f, yz_prob: %.6f" % (xy_prob, xz_prob, yz_prob)
                #print "xy_acc: %.6f, xz_acc: %.6f, yz_acc: %.6f" % (xy_acc, xz_acc, yz_acc)

                probability.append([xy_prob, xz_prob, yz_prob])
                accuracy.append([xy_acc, xz_acc, yz_acc])
                if xy_acc>0 or xz_acc>0 or yz_acc>0:
                    cancer.append([xy_prob, xz_prob, yz_prob])
        

        if len(accuracy)>0:
            np.save(ACCURACY_FOLDER+u_input, accuracy)
            print ACCURACY_FOLDER+name+ ' saved!'            
        if len(probability)>0:
            prob = np.array(probability, dtype='float32')
            max_prob = np.amax(prob)
            print prob.shape, max_prob
            np.save(PROB_FOLDER+u_input, prob)
            print PROB_FOLDER+u_input+' saved!'
            result.append([name, max_prob])
            content = str(name)+':'+str(max_prob)
        else:
            result.append([name, 0])
            content = str(name)+':'+'0'
            
        fout = open(NODULE_TXT,'a')
        fout.writelines(content+'\n')
        fout.close()

        if len(cancer)>0:
            prob = np.array(cancer, dtype='float32')
            max_prob = np.amax(prob)
            print prob.shape, max_prob
            result1.append([name, max_prob])
            np.save(CANCER_FOLDER+u_input, prob)
            print CANCER_FOLDER+u_input+' saved!'
            content = str(name)+':'+str(max_prob)
        else:
            result1.append([name, 0])
            content = str(name)+':'+'0'
        f2out = open(CANCER_TXT, 'a')
        f2out.writelines(content+'\n')
        f2out.close()

    # df = pd.DataFrame(result,columns=PREDICTIONS)
    # df.to_csv(RESULT_CSV, index=False)

    # df1 = pd.DataFrame(result1,columns=PREDICTIONS)
    # df1.to_csv(RESULT1_CSV, index=False)