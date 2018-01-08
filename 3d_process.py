import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import SimpleITK as sitk

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage.measurements import center_of_mass
from skimage.measure import label,regionprops, perimeter
from scipy import ndimage

from joblib import Parallel, delayed

os.environ["CUDA_VISIBLE_DEVICES"]="1"

PREDICTIONS = ['id','cancer']

INPUT_FOLDER = '/mnt/filsystem1/code/dsb2017/stage1_data/stage1/'
labels_path = '/mnt/filsystem1/code/dsb2017/stage1_labels/stage1_labels.csv'
SAVE_FOLDER = '/mnt/filsystem1/code/dsb2017/code/ZJJ/3d_resnet/probability/'
THD_PREPROCESS = '/mnt/filsystem1/code/dsb2017/code/ZJJ/3d_resnet/3d_preprocess/'
UNET_INPUT = THD_PREPROCESS +'unet_input_train/'
RESNET_PADDING = THD_PREPROCESS + 'resnet_padding_train/'

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
def crop_or_pad(image, desired_size, pad_value):
    if image.shape[0] < desired_size:
        offset = int(np.ceil((desired_size-image.shape[0])/2))
        image = np.pad(image, offset, 'constant', constant_values=pad_value)

    if image.shape[0] > desired_size:
        offset = (image.shape[0]-desired_size)//2
        image = image[offset:offset+desired_size,offset:offset+desired_size]

    return image

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
    return worldCoor

def preprocess(patient, train_names, order):
    res = train_names[train_names==patient]
    name = patient + '.npy'
    # print res.index,res.values
    # if res.values exist do preprocess to training set
    if res.values:
        if not os.path.exists(RESNET_PADDING+name):
            print order, patient#, int(train_labels[res.index].values)
            first_patient = load_scan(INPUT_FOLDER + patient)
            origin = first_patient[0].ImagePositionPatient

            first_patient_pixels = get_pixels_hu(first_patient)
                
            pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])    

            segmented_lungs_fill = segment_lung_mask(pix_resampled, True)

            unet_input = segmented_lungs_fill*pix_resampled
            unet_input = [crop_or_pad(img, 324, 0) for img in unet_input]
            np.save(UNET_INPUT+name, unet_input)
            print 'unet_input save:',name
                
            resnet_sampled, resnet_spacing = resample(first_patient_pixels, first_patient, OUTPUT_SPACING)  # 0.5mm*0.5mm*0.5mm         
            offset = OUTPUT_DIM/2
            resnet_padding = np.pad(resnet_sampled,offset,'constant',constant_values=0)
            resnet_padding = np.transpose(resnet_padding)
            np.save(RESNET_PADDING+name, resnet_padding)
            print 'resnet_padding save:',name

if __name__=='__main__':
    # = os.listdir(INPUT_FOLDER)

    df_train = pd.read_csv(labels_path)
    train_names = df_train.id
    train_labels = df_train.cancer

    patient = '781cf9cc66e59f4586d9338630b95c1a'

    #Parallel(n_jobs=12)(delayed(preprocess)(patient, train_names, order) for order, patient in enumerate(patients))


    
    # patient = '6faabf4152bf0ebfd91f686bc37a1f16'

    name = patient + '.npy'
    # print res.index,res.values
    # if res.values exist do preprocess to training set

    
    first_patient = load_scan(INPUT_FOLDER + patient)
    origin = first_patient[0].ImagePositionPatient

    first_patient_pixels = get_pixels_hu(first_patient)
                
    pix_resampled, spacing = resample(first_patient_pixels, first_patient, [1,1,1])

    segmented_lungs_fill = segment_lung_mask(pix_resampled, True)
    segmented_lungs = segment_lung_mask(pix_resampled, False)

    plot_3d(segmented_lungs)

    unet_input = segmented_lungs_fill*pix_resampled
    unet_input = [crop_or_pad(img, 324, 0) for img in unet_input]
    print 'end crop_or_pad'
    np.save(UNET_INPUT+name, unet_input)
    print 'unet_input save:',name
                
    resnet_sampled, resnet_spacing = resample(first_patient_pixels, first_patient, OUTPUT_SPACING)  # 0.5mm*0.5mm*0.5mm         
    offset = OUTPUT_DIM/2
    resnet_padding = np.pad(resnet_sampled,offset,'constant',constant_values=0)
    resnet_padding = np.transpose(resnet_padding)
    np.save(RESNET_PADDING+name, resnet_padding)
    print 'resnet_padding save:',name
