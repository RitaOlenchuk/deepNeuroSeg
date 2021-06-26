import os
import scipy
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, Activation
from tensorflow.keras.layers import concatenate
from tensorflow.keras import backend as K
import SimpleITK as sitk
from .segmentation_types import AbstractSegmenter

smooth=1.0
img_shape = (180, 180, 1)

class ClaustrumSegmentation(AbstractSegmenter):
    c_dict = {"pretrained_T1_claustrum":{'axial_0.h5':'57A6FFA5FD700FDB&resid=57A6FFA5FD700FDB%21120&authkey=AM4S6ZzpCEY4B0g',
                                         'axial_1.h5':'57A6FFA5FD700FDB&resid=57A6FFA5FD700FDB%21121&authkey=AEDWiapn9bksT94',
                                         'axial_2.h5':'57A6FFA5FD700FDB&resid=57A6FFA5FD700FDB%21122&authkey=ANEGp1S5oXy75-o',
                                         'coronal_0.h5':'57A6FFA5FD700FDB&resid=57A6FFA5FD700FDB%21123&authkey=AG-qBBe1ENeIov8',
                                         'coronal_1.h5':'57A6FFA5FD700FDB&resid=57A6FFA5FD700FDB%21125&authkey=AC9C3n98z8J6F_Q',
                                         'coronal_2.h5':'57A6FFA5FD700FDB&resid=57A6FFA5FD700FDB%21124&authkey=AGjLrH6ZhQXWyJk'}}

    def __init__(self, T1_path):
        self.T1_path = T1_path

    def get_T1_path(self):
        return self.T1_path

    def _get_links(self):
        return 'pretrained_T1_claustrum', ClaustrumSegmentation.c_dict['pretrained_T1_claustrum']

    def perform_segmentation(self, outputPath=None, check_orientation=False):
        """Performs claustrum segmentation by loading required models from ./~deepNeuroSeg/pretrained_T1_claustrum cache directory.

        Args:
            outputPath (str, optional): the desired directory path where the resulting mask will be saved under the name out_mask.nii.gz. Defaults to None meaning not saved.

        Returns:
            numpy.ndarray: the predicted mask.
        """
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(self.T1_path))
        norm_image_array = self.z_score_normalization(image_array)
        # transform/project the original array to axial and coronal views
        corona_array = np.transpose(norm_image_array, (1, 0, 2))
        axial_array = norm_image_array

        #original size and the orientations 
        ori_size_c = np.asarray(np.shape(corona_array))
        ori_size_a = np.asarray(np.shape(axial_array))
        orient_c = [1, 0, 2]
        orient_a = [0, 1, 2]

        #pre-processing, crop or pad them to a standard size [N, 180, 180]
        corona_array =  self.pre_processing(corona_array, img_shape)
        axial_array =  self.pre_processing(axial_array, img_shape)

        if check_orientation:
            #this is to check the orientation of your images is right or not. Please check /images/coronal and /images/axial
            cache_dir = os.path.realpath(os.path.expanduser('~/.deepNeuroSeg'))
            image_path = os.path.join(cache_dir,'images')
            direction_1 = 'coronal'
            direction_2 = 'axial'
            if not os.path.exists(image_path):
                os.makedirs(image_path)
            if not os.path.exists(os.path.join(image_path, direction_1)):
                os.makedirs(os.path.join(image_path, direction_1))
            if not os.path.exists(os.path.join(image_path, direction_2)):
                os.makedirs(os.path.join(image_path, direction_2))
            for ss in range(np.shape(corona_array)[0]):
                slice_ = 255*(corona_array[ss] - np.min(corona_array[ss]))/(np.max(corona_array[ss]) - np.min(corona_array[ss]))
                np_slice_ = np.squeeze(slice_, axis=2)
                im = Image.fromarray(np.uint8(np_slice_))
                im.save(os.path.join(image_path, direction_1, str(ss)+'.png'))
            
            for ss in range(np.shape(axial_array)[0]):
                slice_ = 255*(axial_array[ss] - np.min(axial_array[ss]))/(np.max(axial_array[ss]) - np.min(axial_array[ss]))
                np_slice_ = np.squeeze(slice_, axis=2)
                im = Image.fromarray(np.uint8(np_slice_))
                im.save(os.path.join(image_path, direction_2, str(ss)+'.png')) 

        pred_a, pred_c = self.predict(axial_array, corona_array, self.get_unet(img_shape))

        # transform them to their original size and orientations
        pred_1_post = self.post_processing(pred_c, ori_size_c, orient_c)
        pred_2_post = self.post_processing(pred_a, ori_size_a, orient_a)

        # ensemble of two views
        pred = (pred_1_post+pred_2_post)/2
        pred[pred > 0.40] = 1.
        pred[pred <= 0.40] = 0.

        if outputPath:
            #save the masks
            self.save_segmentation(pred, outputPath)
        
        return pred

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

        img_out = sitk.GetImageFromArray(mask)
        sitk.WriteImage(img_out, filename_resultImage)

    def predict(self, axial_array, corona_array, model):
        #do inference on different views 
        direction_1 = 'coronal'
        direction_2 = 'axial'
        model_dir = os.path.realpath(os.path.expanduser('~/.deepNeuroSeg/pretrained_T1_claustrum'))
        model_path_1 = os.path.join(model_dir,direction_1+'_0.h5')
        model.load_weights(model_path_1)  
        pred_1c = model.predict(corona_array, batch_size=1, verbose=True)
        model_path_2 = os.path.join(model_dir,direction_1+'_1.h5')
        model.load_weights(model_path_2)  
        pred_2c = model.predict(corona_array, batch_size=1, verbose=True)
        model_path_3 = os.path.join(model_dir,direction_1+'_2.h5')
        model.load_weights(model_path_3)  
        pred_3c = model.predict(corona_array, batch_size=1, verbose=True)
        # ensemble 
        pred_c = (pred_1c+pred_2c+pred_3c)/3
        
        model_path_1 = os.path.join(model_dir,direction_2+'_0.h5')
        model.load_weights(model_path_1)  
        pred_1a = model.predict(axial_array, batch_size=1, verbose=True)
        model_path_2 = os.path.join(model_dir,direction_2+'_1.h5')
        model.load_weights(model_path_2)  
        pred_2a = model.predict(axial_array, batch_size=1, verbose=True)
        model_path_3 = os.path.join(model_dir,direction_2+'_2.h5')
        model.load_weights(model_path_3)  
        pred_3a = model.predict(axial_array, batch_size=1, verbose=True)
        
        pred_a = (pred_1a+pred_2a+pred_3a)/3
        return pred_a, pred_c

    def z_score_normalization(self, image, thresh=10):    
        # z-score normalization
        brain_mask_T1 = np.zeros(np.shape(image), dtype = 'float32')
        brain_mask_T1[image >=10] = 1
        brain_mask_T1[image < 10] = 0
        for iii in range(np.shape(image)[0]):
            brain_mask_T1[iii,:,:] = scipy.ndimage.morphology.binary_fill_holes(brain_mask_T1[iii,:,:])  #fill the holes inside br
        image_array = image - np.mean(image[brain_mask_T1 == 1])
        image_array /= np.std(image_array[brain_mask_T1 == 1])
        return image_array

    def dice_coef_for_training(self, y_true, y_pred):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def dice_coef_loss(self, y_true, y_pred):
        return 1.0 - self.dice_coef_for_training(y_true, y_pred)

    def conv_bn_relu(self, nd, k=3, inputs=None):
        conv = Conv2D(nd, k, padding='same')(inputs)
        relu = Activation('relu')(conv)
        return relu

    def get_crop_shape(self, target, refer):
            # width, the 3rd dimension
            cw = (target.get_shape()[2] - refer.get_shape()[2])
            assert (cw >= 0)
            if cw % 2 != 0:
                cw1, cw2 = int(cw//2), int(cw//2) + 1
            else:
                cw1, cw2 = int(cw//2), int(cw//2)
            # height, the 2nd dimension
            ch = (target.get_shape()[1] - refer.get_shape()[1])
            assert (ch >= 0)
            if ch % 2 != 0:
                ch1, ch2 = int(ch//2), int(ch//2) + 1
            else:
                ch1, ch2 = int(ch//2), int(ch//2)

            return (ch1, ch2), (cw1, cw2)

    def get_unet(self, img_shape = None, first5=False):
            inputs = Input(shape = img_shape)
            concat_axis = -1

            if first5: filters = 5
            else: filters = 3
            conv1 = self.conv_bn_relu(32, filters, inputs)
            conv1 = self.conv_bn_relu(32, filters, conv1)
            pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
            conv2 = self.conv_bn_relu(64, 3, pool1)
            conv2 = self.conv_bn_relu(64, 3, conv2)
            pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
            conv3 = self.conv_bn_relu(96, 3, pool2)
            conv3 = self.conv_bn_relu(96, 3, conv3)
            pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

            conv4 = self.conv_bn_relu(128, 3, pool3)
            conv4 = self.conv_bn_relu(128, 4, conv4)
            pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

            conv5 = self.conv_bn_relu(256, 3, pool4)
            conv5 = self.conv_bn_relu(256, 3, conv5)

            up_conv5 = UpSampling2D(size=(2, 2))(conv5)
            ch, cw = self.get_crop_shape(conv4, up_conv5)
            crop_conv4 = Cropping2D(cropping=(ch,cw))(conv4)
            up6 = concatenate([up_conv5, crop_conv4], axis=concat_axis)
            conv6 = self.conv_bn_relu(128, 3, up6)
            conv6 = self.conv_bn_relu(128, 3, conv6)

            up_conv6 = UpSampling2D(size=(2, 2))(conv6)
            ch, cw = self.get_crop_shape(conv3, up_conv6)
            crop_conv3 = Cropping2D(cropping=(ch,cw))(conv3)
            up7 = concatenate([up_conv6, crop_conv3], axis=concat_axis)
            conv7 = self.conv_bn_relu(96, 3, up7)
            conv7 = self.conv_bn_relu(96, 3, conv7)

            up_conv7 = UpSampling2D(size=(2, 2))(conv7)
            ch, cw = self.get_crop_shape(conv2, up_conv7)
            crop_conv2 = Cropping2D(cropping=(ch,cw))(conv2)
            up8 = concatenate([up_conv7, crop_conv2], axis=concat_axis)
            conv8 = self.conv_bn_relu(64, 3, up8)
            conv8 = self.conv_bn_relu(64, 3, conv8)

            up_conv8 = UpSampling2D(size=(2, 2))(conv8)
            ch, cw = self.get_crop_shape(conv1, up_conv8)
            crop_conv1 = Cropping2D(cropping=(ch,cw))(conv1)
            up9 = concatenate([up_conv8, crop_conv1], axis=concat_axis)
            conv9 = self.conv_bn_relu(32, 3, up9)
            conv9 = self.conv_bn_relu(32, 3, conv9)

            ch, cw = self.get_crop_shape(inputs, conv9)
            conv9 = ZeroPadding2D(padding=(ch, cw))(conv9)
            conv10 = Conv2D(1, 1, activation='sigmoid', padding='same')(conv9) #, kernel_initializer='he_normal'
            model = Model(inputs=inputs, outputs=conv10)
            opt = keras.optimizers.Adam(learning_rate=2e-4)
            model.compile(optimizer=opt, loss=self.dice_coef_loss)

            return model

    def crop_or_pad(self, input_array, ori_size, per_=0.2):
    # dim_1 = np.shape(input_array)[0]
        dim_2 = np.shape(input_array)[1]
        dim_3 = np.shape(input_array)[2]
        rows = ori_size[1]
        cols = ori_size[2]
        array_1 = np.zeros(ori_size, dtype = 'float32')
        array_1[...] = np.min(input_array)
        
        if dim_2 <=rows and dim_3<=cols: 
            array_1[int(ori_size[0]*per_): (ori_size[0] -int(ori_size[0]*per_)), int((rows - dim_2)/2):(int((rows - dim_2)/2)+ dim_2), int((cols - dim_3)/2):(int((cols - dim_3)/2)+dim_3)] = input_array[:, :, :, 0]
        elif dim_2>=rows and dim_3>=cols: 
            array_1[int(ori_size[0]*per_): (ori_size[0] -int(ori_size[0]*per_)), :, :] = input_array[:, int((dim_2 -rows)/2):(int((dim_2-rows)/2)+ rows), int((dim_3-cols)/2):(int((dim_3-cols)/2)+cols), 0]
        elif dim_2>=rows and dim_3<=cols: 
            array_1[int(ori_size[0]*per_): (ori_size[0] -int(ori_size[0]*per_)), :, int((cols-dim_3)/2):(int((cols-dim_3)/2)+dim_3)] = input_array[:, int((dim_2 -rows)/2):(int((dim_2-rows)/2)+ rows), :, 0]
        elif dim_2<=rows and dim_3>=cols: 
            array_1[int(ori_size[0]*per_): (ori_size[0] -int(ori_size[0]*per_)), int((rows-dim_2)/2):(int((rows-dim_2)/2)+ dim_2), :] = input_array[:, :, int((dim_3 -cols)/2):(int((dim_3 -cols)/2)+cols), 0]
        return array_1

    def pre_processing(self, volume, ref_size, per_ = 0.2):
        rows, cols = ref_size[0], ref_size[1]
        dim_1 = np.shape(volume)[0]
        orig_rows, orig_cols = np.shape(volume)[1], np.shape(volume)[2]
        cropped_volume = []
        for nn in range(np.shape(volume)[0]):
            min_value = np.min(volume)
            if orig_rows >= rows and orig_cols >= cols:
                cropped_volume.append(volume[nn, int((orig_rows - rows) / 2): int((orig_rows - rows) / 2) + rows,
                                    int((orig_cols - cols) / 2): int((orig_cols - cols) / 2) + cols])
            elif orig_rows >= rows and cols >= orig_cols:
                norm_slice = np.zeros((rows, cols))
                norm_slice[...] = min_value
                norm_slice[:, int((cols - orig_cols) / 2): int((cols - orig_cols) / 2) + orig_cols] = volume[nn, 
                                                        int((orig_rows - rows) / 2): int((orig_rows - rows) / 2) + rows, :]
                cropped_volume.append(norm_slice)
            elif rows >= orig_rows and orig_cols >= cols:
                norm_slice = np.zeros((rows, cols))
                norm_slice[...] = min_value
                norm_slice[int((rows - orig_rows) / 2): int((rows - orig_rows) / 2) + orig_rows, :] = volume[nn, :, int((orig_cols - cols) / 2): int((orig_cols - cols) / 2) + cols]
                cropped_volume.append(norm_slice)
            elif rows >= orig_rows and cols >= orig_cols:
                norm_slice = np.zeros((rows, cols))
                norm_slice[...] = min_value
                norm_slice[int((rows - orig_rows) / 2): int((rows - orig_rows) / 2) + orig_rows, int((cols - orig_cols) / 2): int((cols - orig_cols) / 2) + orig_cols] = volume[nn, :, :]
                cropped_volume.append(norm_slice)
        cropped_volume = np.asarray(cropped_volume)
        cropped_volume = cropped_volume[int(dim_1*per_): (dim_1 -int(dim_1*per_))]
        return cropped_volume[..., np.newaxis]

    def inverse_orient(self, orient_):
        inv_orient = []
        if orient_ == [0, 1, 2]:
            inv_orient = (0, 1, 2)
        elif orient_ == [1, 0, 2]:
            inv_orient = (1, 0, 2)
        elif orient_ == [1, 2, 0]:
            inv_orient = (2, 0, 1)
        elif orient_ == [2, 1, 0]:
            inv_orient = (2, 1, 0)
        return inv_orient

    def post_processing(self, input_array, ori_size, orient_1):
        output_array = self.crop_or_pad(input_array, ori_size)
        inv_orient = self.inverse_orient(orient_1)
        output_array = np.transpose(output_array, inv_orient)
        return output_array