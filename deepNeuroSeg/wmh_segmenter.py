import os
import scipy
import logging
import numpy as np
import SimpleITK as sitk
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras import backend as K
from .segmentation_types import AbstractSegmenter

rows_standard = 200  #the input size 
cols_standard = 200
os.environ["CUDA_VISIBLE_DEVICES"]="1" ## select which gpu to use; if using CPU, just comment this.


class WMHSegmentation(AbstractSegmenter):
    wmh_dict = {"pretrained_FLAIR_only": {'0.h5':'57A6FFA5FD700FDB&resid=57A6FFA5FD700FDB%21110&authkey=ANvFSyNpSSjh3NQ',
                                          '1.h5':'57A6FFA5FD700FDB&resid=57A6FFA5FD700FDB%21111&authkey=APGrgxpxl_4OBHE',
                                          '2.h5':'57A6FFA5FD700FDB&resid=57A6FFA5FD700FDB%21109&authkey=AONWUbE5llZpbS4'}, 
                "pretrained_FLAIR_T1": {'0.h5':'57A6FFA5FD700FDB&resid=57A6FFA5FD700FDB%21113&authkey=ALAHNnoGtEi7tl4',
                                        '1.h5':'57A6FFA5FD700FDB&resid=57A6FFA5FD700FDB%21114&authkey=AAl5nwEj0AgyAcQ',
                                        '2.h5':'57A6FFA5FD700FDB&resid=57A6FFA5FD700FDB%21112&authkey=ALZ8G_0zUgLo-ro'}}

    def __init__(self, FLAIR_path, T1_path=None):
        self.FLAIR_path = FLAIR_path
        self.T1_path = T1_path

    def get_FLAIR_path(self):
        return self.FLAIR_path

    def get_T1_path(self):
        return self.T1_path

    def perform_segmentation(self, outputPath=None):
        """Performs segmentation by loading three required models from ./~deepNeuroSeg cache directory.

        Args:
            outputPath (str, optional): the desired directory path where the resulting mask will be saved under the name out_mask.nii.gz. Defaults to None meaning not saved.

        Returns:
            numpy.ndarray: the predicted mask.
        """
        img_shape, imgs_test, model_dir, FLAIR_array, real_FLAIR_dim = read_data(self.FLAIR_path, self.T1_path)
        original_pred = load_model(img_shape, imgs_test, model_dir, FLAIR_array)

        if real_FLAIR_dim[1]<rows_standard:
            original_pred = original_pred[:,:real_FLAIR_dim[1],:]
        if real_FLAIR_dim[2]<cols_standard:
            original_pred = original_pred[:,:,:real_FLAIR_dim[2]]
            
        if outputPath:
            self.save_segmentation(original_pred, outputPath)

        return original_pred

    def save_segmentation(self, mask, outputPath):
        """Saves provided mask as out_mask.nii.gz in the given directory.

        Args:
            mask (numpy.ndarray): the mask.
            outputPath ([type]): the desired directory path where the resulting mask will be saved under the name out_mask.nii.gz
        """
        if os.path.isdir(outputPath):
            if not os.path.exists(outputPath):
                os.mkdir(outputPath)
            filename_resultImage = os.path.join(outputPath,'out_mask.nii.gz')
        else:
            if outputPath.endswith('nii.gz'):
                filename_resultImage = outputPath
            else:
                raise NameError('Invalide file expension. Must end with .nii.gz')
        FLAIR_image = sitk.ReadImage(self.FLAIR_path)
        img_out = sitk.GetImageFromArray(mask)
        img_out.CopyInformation(FLAIR_image) #copy the meta information (voxel size, etc.) from the input raw image
        sitk.WriteImage(img_out, filename_resultImage)

    def _get_links(self):
        if self.T1_path:
            return 'pretrained_FLAIR_T1', WMHSegmentation.wmh_dict['pretrained_FLAIR_T1']
        else:
            return 'pretrained_FLAIR_only', WMHSegmentation.wmh_dict['pretrained_FLAIR_only']

def expand_rows(image):
    updated_image = np.zeros((image.shape[0],rows_standard,image.shape[2]))
    updated_image[:, :image.shape[1], :image.shape[2]] = image
    return updated_image

def expand_columns(image):
    updated_image = np.zeros((image.shape[0],image.shape[1],cols_standard))
    updated_image[:, :image.shape[1], :image.shape[2]] = image
    return updated_image


def read_data(FLAIR_path, T1_path):
    FLAIR_image = sitk.ReadImage(FLAIR_path)
    FLAIR_array = sitk.GetArrayFromImage(FLAIR_image)
    real_FLAIR_dim = FLAIR_array.shape
    if FLAIR_array.shape[1]<rows_standard:
        FLAIR_array = expand_rows(FLAIR_array)
    if FLAIR_array.shape[2]<cols_standard:
        FLAIR_array = expand_columns(FLAIR_array)
    if T1_path is None:
        # single modality as the input
        img_shape=(rows_standard, cols_standard, 1)
        model_dir = os.path.realpath(os.path.expanduser('~/.deepNeuroSeg/pretrained_FLAIR_only'))
        T1_array = []
        imgs_test = preprocessing(np.float32(FLAIR_array), np.float32(T1_array))
    else:
        img_shape=(rows_standard, cols_standard, 2)
        model_dir = os.path.realpath(os.path.expanduser('~/.deepNeuroSeg/pretrained_FLAIR_T1'))
        T1_image = sitk.ReadImage(T1_path)
        T1_array = sitk.GetArrayFromImage(T1_image)
        if T1_array.shape[1]<rows_standard:
            T1_array = expand_rows(T1_array)
        if T1_array.shape[2]<cols_standard:
            T1_array = expand_columns(T1_array)
        imgs_test = preprocessing(np.float32(FLAIR_array), np.float32(T1_array))
    return img_shape, imgs_test, model_dir, FLAIR_array, real_FLAIR_dim

def load_model(img_shape, imgs_test, model_dir, FLAIR_array):
    model = get_u_net(img_shape)
    logging.info(model_dir)
    model.load_weights(os.path.join(model_dir,'0.h5'))  # 3 ensemble models
    logging.info('-'*30)
    logging.info('Predicting masks on test data...') 
    pred_1 = model.predict(imgs_test, batch_size=1, verbose=1)
    model.load_weights(os.path.join(model_dir, '1.h5')) 
    pred_2 = model.predict(imgs_test, batch_size=1, verbose=1)
    model.load_weights(os.path.join(model_dir, '2.h5'))
    pred_3 = model.predict(imgs_test, batch_size=1, verbose=1)
    pred = (pred_1+pred_2+pred_3)/3
    pred[pred[...,0] > 0.45] = 1      #0.45 thresholding 
    pred[pred[...,0] <= 0.45] = 0
    original_pred = postprocessing(FLAIR_array, pred) # get the original size to match
    return original_pred

def dice_coef_for_training(y_true, y_pred, smooth = 1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return - dice_coef_for_training(y_true, y_pred)

def conv_l_relu(nd, k=3, inputs=None):
    conv = Conv2D(nd, k, padding='same')(inputs)
    L_relu = LeakyReLU(alpha=0.01)(conv)
    return L_relu

def get_crop_shape(target, refer):
    # width, the 3rd dimension
    cw = (target.get_shape()[2] - refer.get_shape()[2])#.value
    assert (cw >= 0)
    if cw % 2 != 0:
        cw1, cw2 = int(cw / 2), int(cw / 2) + 1
    else:
        cw1, cw2 = int(cw / 2), int(cw / 2)
    # height, the 2nd dimension
    ch = (target.get_shape()[1] - refer.get_shape()[1])#.value
    assert (ch >= 0)
    if ch % 2 != 0:
        ch1, ch2 = int(ch / 2), int(ch / 2) + 1
    else:
        ch1, ch2 = int(ch / 2), int(ch / 2)
    #return (1, 1), (1, 1)
    return (ch1, ch2), (cw1, cw2)


def get_u_net(img_shape=None):
    concat_axis = -1
    input = Input(shape=img_shape)
    conv1 = conv_l_relu(64, 5, input)
    conv1 = conv_l_relu(64, 5, conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv_l_relu(96, 3, pool1)
    conv2 = conv_l_relu(96, 3, conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv_l_relu(128, 3, pool2)
    conv3 = conv_l_relu(128, 3, conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv_l_relu(256, 3, pool3)
    conv4 = conv_l_relu(256, 4, conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv_l_relu(512, 3, pool4)
    conv5 = conv_l_relu(512, 3, conv5)

    up_conv5 = UpSampling2D(size=(2, 2))(conv5)
    ch, cw = get_crop_shape(conv4, up_conv5)
    crop_conv4 = Cropping2D(cropping=(ch, cw))(conv4)
    up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
    conv6 = conv_l_relu(256, 3, up6)
    conv6 = conv_l_relu(256, 3, conv6)

    up_conv6 = UpSampling2D(size=(2, 2))(conv6)
    ch, cw = get_crop_shape(conv3, up_conv6)
    crop_conv3 = Cropping2D(cropping=(ch, cw))(conv3)
    up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
    conv7 = conv_l_relu(128, 3, up7)
    conv7 = conv_l_relu(128, 3, conv7)

    up_conv7 = UpSampling2D(size=(2, 2))(conv7)
    ch, cw = get_crop_shape(conv2, up_conv7)
    crop_conv2 = Cropping2D(cropping=(ch, cw))(conv2)
    up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
    conv8 = conv_l_relu(96, 3, up8)
    conv8 = conv_l_relu(96, 3, conv8)

    up_conv8 = UpSampling2D(size=(2, 2))(conv8)
    ch, cw = get_crop_shape(conv1, up_conv8)
    crop_conv1 = Cropping2D(cropping=(ch, cw))(conv1)
    up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
    conv9 = conv_l_relu(64, 3, up9)
    conv9 = conv_l_relu(64, 3, conv9)

    ch, cw = get_crop_shape(input, conv9)
    conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
    dice_out = Conv2D(1, 1, activation='sigmoid', padding='same', name='dice_out')(conv9)
    unet = Model(inputs=input, outputs=dice_out)
    return unet

def preprocessing(FLAIR_array, T1_array):
    thresh = 30   # threshold for getting the brain mask
    brain_mask = np.ndarray(np.shape(FLAIR_array), dtype=np.float32)
    brain_mask[FLAIR_array >=thresh] = 1
    brain_mask[FLAIR_array < thresh] = 0
    for iii in range(np.shape(FLAIR_array)[0]):
        brain_mask[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask[iii,:,:])  #fill the holes inside brain
    
    FLAIR_array -=np.mean(FLAIR_array[brain_mask == 1])      #Gaussion Normalization
    FLAIR_array /=np.std(FLAIR_array[brain_mask == 1])
    
    rows_o = np.shape(FLAIR_array)[1]
    cols_o = np.shape(FLAIR_array)[2]
    FLAIR_array = FLAIR_array[:, int((rows_o-rows_standard)/2):int((rows_o-rows_standard)/2)+rows_standard, int((cols_o-cols_standard)/2):int((cols_o-cols_standard)/2)+cols_standard]
    
    if len(T1_array)>0:
        T1_array -=np.mean(T1_array[brain_mask == 1])      #Gaussion Normalization
        T1_array /=np.std(T1_array[brain_mask == 1])
        T1_array = T1_array[:, int((rows_o-rows_standard)/2):int((rows_o-rows_standard)/2)+rows_standard, int((cols_o-cols_standard)/2):int((cols_o-cols_standard)/2)+cols_standard]
    
        imgs_two_channels = np.concatenate((FLAIR_array[..., np.newaxis], T1_array[..., np.newaxis]), axis = 3)
        return imgs_two_channels
    else: 
        return FLAIR_array[..., np.newaxis]


def postprocessing(FLAIR_array, pred):
    per = 0.125
    start_slice = int(np.shape(FLAIR_array)[0]*per)
    num_o = np.shape(FLAIR_array)[1]  # original size
    rows_o = np.shape(FLAIR_array)[1]
    cols_o = np.shape(FLAIR_array)[2]
    original_pred = np.zeros(np.shape(FLAIR_array), dtype=np.float32)
    original_pred[:,int((rows_o-rows_standard)/2):int((rows_o-rows_standard)/2)+rows_standard,int((cols_o-cols_standard)/2):int((cols_o-cols_standard)/2)+cols_standard] = pred[:,:,:,0]
    original_pred[0: start_slice, ...] = 0
    original_pred[(num_o-start_slice):num_o, ...] = 0
    return original_pred