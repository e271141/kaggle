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

os.environ["CUDA_VISIBLE_DEVICES"]="1"

INPUT_FOLDER = '/mnt/filsystem1/code/dsb2017/stage1_data/stage1/'
labels_path = '/mnt/filsystem1/code/dsb2017/stage1_labels/stage1_labels.csv'
SAVE_FOLDER = '/mnt/filsystem1/code/dsb2017/code/ZJJ/3d_resnet/unet+resnet/'

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

# Load the scans in given folder path
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
    
def get_pixels_hu(slices): #Hounsfield Unit
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0 
    # The intercept is usually -1024, so air is approximately 0
    image[image == np.amin(image)] = 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)
    

def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = np.array([scan[0].SliceThickness] + scan[0].PixelSpacing, dtype=np.float32)
    
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing
    

def plot_3d(image, threshold=-300):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces = measure.marching_cubes(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()
    

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image

def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image
    
def zero_center(image):
    image = image - PIXEL_MEAN
    return image

def save_itk(image, origin, spacing, filename):
    itkimage = sitk.GetImageFromArray(image, isVector=False)
    itkimage.SetSpacing(spacing)
    itkimage.SetOrigin(origin)
    sitk.WriteImage(itkimage, filename, True)

def load_itk(filename):
    itkimage = sitk.ReadImage(filename)
    image = np.transpose(sitk.GetArrayFromImage(itkimage))
    origin = np.array(itkimage.GetOrigin())
    spacing = np.array(itkimage.GetSpacing())
    return image, origin, spacing

def world_2_voxel(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord

def voxel_2_world(voxelCoord, origin, spacing):
    stretchedVoxelCoord = voxelCoord * spacing
    worldCoord = stretchedVoxelCoord + origin
    return worldCoord

if __name__=='__main__':
    patients = os.listdir(INPUT_FOLDER)

    df_train = pd.read_csv(labels_path)
    train_names = df_train.id
    train_labels = df_train.cancer

    predict_fn = load_unet()

    val_fn = load_resnet()

    for order, patient in enumerate(patients):
        if os.path.isfile(SAVE_FOLDER + patient + '.npy'):
            # print SAVE_FOLDER + patient + '.mhd' + '  exists!'
            continue
        
        else:
            # print order, patient
            res = train_names[train_names==patient]
            # print res.index,res.values
            if res.values:
                print order, patient, int(train_labels[res.index].values)
                first_patient = load_scan(INPUT_FOLDER + patient)
                origin = first_patient[0].ImagePositionPatient

                first_patient_pixels = get_pixels_hu(first_patient)
                
                pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])
                # print("Shape before resampling\t", first_patient_pixels.shape)
                # print("Shape after resampling\t", pix_resampled.shape)        

                segmented_lungs_fill = segment_lung_mask(pix_resampled, True)

                unet_input = segmented_lungs_fill*pix_resampled
                unet_input = [crop_or_pad(img, 324, 0) for img in unet_input]

                
                resnet_sampled, resnet_spacing = resample(first_patient_pixels, first_patient, OUTPUT_SPACING)  # 0.5mm*0.5mm*0.5mm
                print("resnet shape after resampling\t", resnet_sampled.shape)
                
                # save_itk(resnet_sampled, origin, resnet_spacing, SAVE_FOLDER+patient+'.mhd')

                maxAllowedArea = LUNG_PORTION*324*324          
                nodule_centers = []   # without padding

                # select candidate region of interest
                for slice_index, image in enumerate(unet_input):
                    # lung segment
                    # print "slice index: %d"  %  slice_index
                    thresh_img = np.where(image<0, 1.0, 0.)
                    regions = regionprops(thresh_img.astype(int))
                    for region in regions:
                        # select slices where lung area > maxAllowedArea
                        if region.area > maxAllowedArea:

                            prediction = unet_predict([image],predict_fn)  # 1mm*1mm*1mm
                            prediction[prediction<THRESHOLD] = 0   # thresholding
                            prediction[prediction>0] = 1
                            prediction = prediction.reshape(324,324)

                            # localize the center of nodule
                            if np.amax(prediction)>0:
                                centers = []
                                #erosion
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
                                    # print world_coords, resnet_coords

                                    nodule_centers.append(resnet_coords)

                                # f, axarr = plt.subplots(2,2)
                                # axarr[0,0].imshow(prediction.reshape(324,324), cmap=plt.cm.gray)
                                # axarr[0,1].imshow(image, cmap=plt.cm.gray)
                                # axarr[1,0].imshow(image_eroded, cmap=plt.cm.gray)
                                # axarr[1,1].imshow(label_im, cmap=plt.cm.gray)
                                # plt.show()
                                
                            continue


                offset = OUTPUT_DIM/2
                resnet_padding = np.pad(resnet_sampled,offset,'constant',constant_values=0)
                resnet_padding = np.transpose(resnet_padding)
                print("resnet shape after padding\t", resnet_padding.shape)
                print "nodule numbers: %d "  %  len(nodule_centers)

                data = []

                for i, coords in enumerate(nodule_centers):
                    # print coords
                    coords = coords + offset
                    # print coords
                    # Create xy, xz, yz
                    xy_slice = np.transpose(resnet_padding[coords[0]-offset:coords[0]+offset,coords[1]-offset:coords[1]+offset,coords[2]])
                    xz_slice = np.rot90(resnet_padding[coords[0]-offset:coords[0]+offset,coords[1],coords[2]-offset:coords[2]+offset])
                    yz_slice = np.rot90(resnet_padding[coords[0],coords[1]-offset:coords[1]+offset,coords[2]-offset:coords[2]+offset])
                    # f, axarr = plt.subplots(2,2)
                    # axarr[0,0].imshow(xy_slice, cmap=plt.cm.gray)
                    # axarr[0,1].imshow(xz_slice, cmap=plt.cm.gray)
                    # axarr[1,0].imshow(yz_slice, cmap=plt.cm.gray)
                    # plt.show()
                    # assert xy_slice.shape == (OUTPUT_DIM, OUTPUT_DIM)
                    # assert xz_slice.shape == (OUTPUT_DIM, OUTPUT_DIM)
                    # assert yz_slice.shape == (OUTPUT_DIM, OUTPUT_DIM)
                    # Create output
                    if ((xy_slice.shape == (OUTPUT_DIM, OUTPUT_DIM)) and (xz_slice.shape == (OUTPUT_DIM, OUTPUT_DIM)) and (yz_slice.shape == (OUTPUT_DIM, OUTPUT_DIM))):
                        xy_prob, xy_acc = resnet_predict(xy_slice,val_fn)
                        xz_prob, xz_acc = resnet_predict(xz_slice,val_fn)
                        yz_prob, yz_acc = resnet_predict(yz_slice,val_fn)
                        print "xy_prob: %.6f, xz_prob: %.6f, yz_prob: %.6f" % (xy_prob, xz_prob, yz_prob)
                        print "xy_acc: %.6f, xz_acc: %.6f, yz_acc: %.6f" % (xy_acc, xz_acc, yz_acc)

                        # if xy_acc>0 or xz_acc>0 or yz_acc>0:
                        #     output = np.zeros([3,OUTPUT_DIM,OUTPUT_DIM])
                        #     output[0,:,:] = xy_slice
                        #     output[1,:,:] = xz_slice
                        #     output[2,:,:] = yz_slice
                        #     data.append(output)
                        
                if len(data)>0:
                    np.save(SAVE_FOLDER+patient+'.npy', np.asarray(data))
                    print SAVE_FOLDER+patient+'.npy     saved!'



